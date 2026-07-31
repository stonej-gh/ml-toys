# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Gymnasium single-agent duel environment over the physics core.

The learning agent flies ship 0; the opponent is any callable
``policy(arena, ship_idx) -> (turn, thrust, fire)`` (default: a passive
coaster). That callable contract is also the self-play interface: a frozen
policy can fly seat 1 through the same mirrored observation (see
agents/league_duel.py's NetOpponent and orbitduel.netpilot.NetPilot).

Observation is 19 floats (egocentric, rotation-normalized; contract below).
Action is Discrete(12): turn {-1,0,1} x thrust {0,1} x fire {0,1}, flattened.

Rule flags come in named era presets (RULE_PRESETS) so every phase of the
experiment, including the ones the agent exploited, stays reproducible:

    v1-freewalls  walls are free bounces, no fuel, fixed spawn lanes,
                  straight lasers      -> the wall-riding + spawn-ambush era
    v3-rude       walls cost (spin + penalty), still no fuel, fixed lanes
                  -> the burn-to-hover spawn-ambush era
    v4-honest     fuel budget + spawn-phase randomization, straight lasers
    v6-full       everything: curved lasers, hit-spin, fuel, phase, cone

Pass ``rules="v6-full"`` (the default) or override any constructor knob
explicitly; explicit kwargs beat the preset.
"""

import math
import random

import gymnasium as gym
import numpy as np

from . import physics as P

ACTIONS = [(t, th, f) for t in (-1, 0, 1) for th in (0, 1) for f in (0, 1)]


def passive_coaster(arena, idx):
    """Do-nothing opponent: coasts its spawn orbit."""
    return (0, 0, 0)


def obs_from(arena, idx, fuel=1.0):
    """The 19-float egocentric observation from ship `idx`'s seat.

    Module-level so a frozen policy net can fly EITHER seat (self-play), and
    so any port of a trained policy can rebuild the vector element by
    element. THIS LIST IS THE CONTRACT: any reimplementation (the deploy/
    bundle's reference forward, a microcontroller port, a shader) must match
    order, units, and signs exactly, or the policy plays subtly drunk.
    Elements (all roughly unit-scale):
      0 r/h   1 vr/vc   2 vt/vc          (vc = SPAWN_V = 298 pt/s, h = 768 pt;
      3 sin(nose-tangent) 4 cos(...)      velocities in PHYSICS-time pt/s)
      5 sin(bearing to foe, nose frame) 6 cos(...)
      7 foe alive   8 edge clearance/h
      9 dist/h  10 closing speed/vc  11 foe r/h
     12 own damage/5  13 foe damage/5
     14 fire-cooldown fraction remaining (0 = ready)
     15 nearest THREAT laser dist/h (1.0 = none) 16 sin(bearing) 17 cos(bearing)
     18 own fuel-tank level (each seat ticks its own FuelTank)"""
    s = arena.ships[idx]
    o = arena.ships[1 - idx]
    vc = P.SPAWN_V
    r = s.radius() or 1e-9
    ux, uy = s.x / r, s.y / r                                 # radial unit
    tx, ty = -uy, ux                                          # CCW tangent unit
    vr = s.vx * ux + s.vy * uy
    vt = s.vx * tx + s.vy * ty
    nx, ny = s.nose()
    h_rel = math.atan2(tx * ny - ty * nx, tx * nx + ty * ny)  # nose vs CCW tangent
    dx, dy = o.x - s.x, o.y - s.y
    dist = math.hypot(dx, dy) or 1e-9
    bear = math.atan2(dy, dx) - math.atan2(ny, nx)
    closing = -((dx * (o.vx - s.vx) + dy * (o.vy - s.vy)) / dist)
    lz_d, lz_bear = 1.0, 0.0
    for lz in arena.lasers:
        if lz.owner == idx:
            continue
        d = math.hypot(lz.x - s.x, lz.y - s.y) / P.H
        if d < lz_d:
            lz_d = d
            lz_bear = math.atan2(lz.y - s.y, lz.x - s.x) - math.atan2(ny, nx)
    edge = min(P.PLAY_W / 2 - abs(s.x), P.H / 2 - abs(s.y)) / P.H
    vec = [r / P.H, vr / vc, vt / vc,
           math.sin(h_rel), math.cos(h_rel),
           math.sin(bear), math.cos(bear),
           1.0 if o.alive else 0.0, edge,
           dist / P.H, closing / vc, o.radius() / P.H,
           s.damage / P.MAX_DAMAGE, o.damage / P.MAX_DAMAGE,
           s.cooldown / P.FIRE_COOLDOWN if P.FIRE_COOLDOWN else 0.0,
           lz_d, math.sin(lz_bear), math.cos(lz_bear), fuel]
    return np.asarray(vec, dtype=np.float32)


FUEL_BURN = 0.67    # tank per physics-s lit
FUEL_REGEN = 0.035  # tank per physics-s coasting
FUEL_REARM = 0.10   # empty tank must regen past this to re-arm


class FuelTank:
    """A hard actuator budget: a full tank is ~1.5 physics-s (~3 wall-s) of
    continuous burn, then thrust is DEAD until the tank regenerates past
    FUEL_REARM (~3 physics-s of forced coast). The burst limit that a
    per-second reward tax could not express."""

    def __init__(self):
        self.level = 1.0
        self.armed = True

    def gate(self, thrust, frames=1):
        """Apply the budget to a thrust request over `frames` physics frames;
        returns the permitted thrust and ticks the tank."""
        if thrust and not self.armed:
            thrust = 0
        dt = frames * P.DT
        if thrust:
            self.level = max(0.0, self.level - FUEL_BURN * dt)
            if self.level <= 0.0:
                self.armed = False
        else:
            self.level = min(1.0, self.level + FUEL_REGEN * dt)
            if self.level >= FUEL_REARM:
                self.armed = True
        return thrust


FIRE_REACH = P.LASER_SPEED * P.LASER_TTL * 1.25  # nominal shot reach + margin
                                                 # for a closing target


def gate_fire(arena, idx, f, cone):
    """Fire-discipline: the trigger only works inside the bow cone (radians),
    within plausible reach (no clearly-impossible long shots), AND with a
    clear line past the hole."""
    if not f or cone is None:
        return 0 if not f else f
    s0 = arena.ships[idx]
    o0 = arena.ships[1 - idx]
    nx, ny = s0.nose()
    dx, dy = o0.x - s0.x, o0.y - s0.y
    off = math.atan2(nx * dy - ny * dx, nx * dx + ny * dy)
    if not o0.alive or abs(off) > cone:
        return 0
    if math.hypot(dx, dy) > FIRE_REACH:
        return 0

    # hole-in-the-way check on the shot's TRUE path (ship velocity + muzzle
    # along the nose): closed-form nearest approach for straight lasers, a
    # forward-sim under the field when the arena's lasers curve.
    px, py = s0.x + nx * SHIP_R_GUARD, s0.y + ny * SHIP_R_GUARD
    vx = s0.vx + nx * P.LASER_SPEED
    vy = s0.vy + ny * P.LASER_SPEED
    range_ = math.hypot(dx, dy)
    if arena.gravity_on_lasers:
        dt = 1 / 30
        travelled = 0.0
        for _ in range(int(P.LASER_TTL / dt)):
            ax, ay = P.gravity_accel(px, py)
            vx += ax * dt
            vy += ay * dt
            px += vx * dt
            py += vy * dt
            if math.hypot(px, py) < P.HOLE_R * 1.05:
                return 0
            travelled += math.hypot(vx, vy) * dt
            if travelled >= range_:
                return f
        return f
    v2 = vx * vx + vy * vy
    t_target = range_ / math.sqrt(v2) if v2 > 0 else 0.0
    t_star = max(0.0, min(min(P.LASER_TTL, t_target),
                          -(px * vx + py * vy) / v2 if v2 > 0 else 0.0))
    cx, cy = px + vx * t_star, py + vy * t_star
    if math.hypot(cx, cy) < P.HOLE_R * 1.05:
        return 0
    return f


SHIP_R_GUARD = P.SHIP_R  # conservative muzzle offset for the hole-shot guard


# Named era rule-sets (see module docstring). Values are constructor kwargs;
# explicit constructor arguments override the preset.
RULE_PRESETS = {
    "v1-freewalls": dict(spin_kick=0.0, hit_spin=0.0, wall_penalty=0.0,
                         fuel=False, spawn_phase=False, spawn_jitter=0.08,
                         gravity_on_lasers=False),
    "v3-rude":      dict(hit_spin=0.0, wall_penalty=0.8,
                         fuel=False, spawn_phase=False, spawn_jitter=0.08,
                         gravity_on_lasers=False),
    "v4-honest":    dict(hit_spin=0.0, wall_penalty=0.8,
                         fuel=True, spawn_phase=True, spawn_jitter=0.08,
                         gravity_on_lasers=False),
    "v6-full":      dict(wall_penalty=0.8,
                         fuel=True, spawn_phase=True, spawn_jitter=0.08,
                         gravity_on_lasers=True),
}

_UNSET = object()


class OrbitDuelEnv(gym.Env):
    """One learning ship vs a pluggable opponent. Discrete(12) actions.

    info dict carries grader-facing counters, cumulative per episode:
    wall_touches, thrust_frames, frames, longest_burn_s (physics s), and on
    termination outcome ("win"/"loss") and cause ("laser"/"blackhole").
    """

    metadata = {"render_modes": []}

    def __init__(self, opponent=passive_coaster, rules="v6-full",
                 action_repeat=4, max_wall_seconds=120.0,
                 gravity_on_lasers=_UNSET, hit_reward=0.0,
                 spawn_jitter=_UNSET, spawn_phase=_UNSET, fuel=_UNSET,
                 spin_kick=_UNSET, hit_spin=_UNSET, seed=None, record=False,
                 pot_coef=0.0, time_cost=0.0, gamma=0.995,
                 fire_cone_deg=15.0, wall_penalty=_UNSET, thrust_cost=0.0):
        preset = dict(RULE_PRESETS[rules]) if rules else {}

        def pick(name, explicit, fallback):
            if explicit is not _UNSET:
                return explicit
            return preset.get(name, fallback)

        gravity_on_lasers = pick("gravity_on_lasers", gravity_on_lasers, True)
        self.spawn_jitter = pick("spawn_jitter", spawn_jitter, 0.0)
        self.spawn_phase = pick("spawn_phase", spawn_phase, True)
        self.fuel = pick("fuel", fuel, True)
        self.wall_penalty = pick("wall_penalty", wall_penalty, 0.0)
        spin_kick = pick("spin_kick", spin_kick, P.SPIN_KICK)
        hit_spin = pick("hit_spin", hit_spin, P.HIT_SPIN)

        self.opponent = opponent
        self.action_repeat = action_repeat
        self.max_frames = int(max_wall_seconds * P.FPS)
        self.arena = P.Arena(gravity_on_lasers=gravity_on_lasers,
                             spin_kick=spin_kick, hit_spin=hit_spin)
        self.hit_reward = hit_reward    # shaping: +/- per hit dealt/taken
        self.pot_coef = pot_coef        # potential-based distance shaping
        self.time_cost = time_cost      # per wall-second cost (anti-camping)
        self.gamma = gamma              # discount used by the shaping term
        self.fire_cone = math.radians(fire_cone_deg) if fire_cone_deg else None
        self.thrust_cost = thrust_cost  # per wall-second of burn
        self.rng = random.Random(seed)
        self.record = record            # keep a replay of the episode
        self.replay = None
        self.tank = FuelTank()          # ship 0's burst budget
        self._frames = 0
        self._info = {}
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(-4.0, 4.0, shape=(19,),
                                                dtype=np.float32)

    # observation
    def _obs(self):
        return obs_from(self.arena, 0, fuel=self.tank.level)

    # gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.arena.reset()
        self.tank = FuelTank()
        if self.spawn_phase:
            # random spawn PHASE: rotate the whole start configuration so there
            # is no fixed lane to camp (kills the spawn-ambush exploit)
            phi = self.rng.uniform(0, 2 * math.pi)
            cphi, sphi = math.cos(phi), math.sin(phi)
            for ship in self.arena.ships:
                x, y = ship.x, ship.y
                vx, vy = ship.vx, ship.vy
                ship.x, ship.y = x * cphi - y * sphi, x * sphi + y * cphi
                ship.vx, ship.vy = vx * cphi - vy * sphi, vx * sphi + vy * cphi
                ship.heading += phi
        if self.spawn_jitter > 0:
            for ship in self.arena.ships:
                k = 1 + self.rng.uniform(-self.spawn_jitter, self.spawn_jitter)
                ship.vx *= k
                ship.vy *= k
                ship.heading += self.rng.uniform(-math.pi, math.pi) * self.spawn_jitter
        self._frames = 0
        self._info = {"wall_touches": 0, "thrust_frames": 0, "frames": 0,
                      "longest_burn_s": 0.0}
        self._burn_frames = 0
        self.replay = {"frames": [], "events": []} if self.record else None
        return self._obs(), dict(self._info)

    def step(self, action):
        t, th, f = ACTIONS[int(action)]

        # fire-discipline constraint (shared with any net opponent on the
        # other seat; see gate_fire)
        f = gate_fire(self.arena, 0, f, self.fire_cone)
        if self.fuel:
            th = self.tank.gate(th, frames=self.action_repeat)
        act0 = (t * P.TURN_RATE, th, f)      # discrete turn -> signed rate
        act1 = self.opponent(self.arena, 1)  # one opponent decision per block
        reward, terminated = 0.0, False
        a0, b0 = self.arena.ships
        d_before = math.hypot(b0.x - a0.x, b0.y - a0.y)
        for _ in range(self.action_repeat):
            events = self.arena.step((act0, act1))
            self._frames += 1
            self._info["frames"] = self._frames
            if th:
                self._info["thrust_frames"] += 1
                self._burn_frames += 1
                self._info["longest_burn_s"] = max(
                    self._info["longest_burn_s"], self._burn_frames * P.DT)
            else:
                self._burn_frames = 0
            for ev in events:
                if ev[0] == 'death':
                    reward += -1.0 if ev[1] == 0 else 1.0
                    terminated = True
                    self._info["outcome"] = "loss" if ev[1] == 0 else "win"
                    self._info["cause"] = ev[2]
                    if self.replay is not None:
                        self.replay["events"].append(
                            {"f": len(self.replay["frames"]), "ev": "death",
                             "ship": ev[1], "cause": ev[2]})
                elif ev[0] == 'hit':
                    reward += self.hit_reward if ev[1] == 1 else -self.hit_reward
                    if self.replay is not None:
                        self.replay["events"].append(
                            {"f": len(self.replay["frames"]), "ev": "hit",
                             "ship": ev[1]})
                elif ev[0] == 'wall':
                    if ev[1] == 0:
                        reward -= self.wall_penalty
                        self._info["wall_touches"] += 1
                    if self.replay is not None:
                        self.replay["events"].append(
                            {"f": len(self.replay["frames"]), "ev": "wall",
                             "ship": ev[1]})
            if self.replay is not None and self._frames % 3 == 0:
                a, b = self.arena.ships
                self.replay["frames"].append([
                    round(a.x, 1), round(a.y, 1), round(a.heading, 3), int(a.alive),
                    round(b.x, 1), round(b.y, 1), round(b.heading, 3), int(b.alive),
                    int(a.thrusting), int(b.thrusting),
                    [[round(lz.x), round(lz.y)] for lz in self.arena.lasers]])
            if terminated:
                break

        # potential-based shaping (gamma*phi(s') - phi(s), phi = -c*dist/h):
        # rewards approach without changing which policy is optimal
        if self.pot_coef and self.arena.ships[1].alive:
            a1, b1 = self.arena.ships
            d_after = math.hypot(b1.x - a1.x, b1.y - a1.y)
            reward += (self.gamma * (-self.pot_coef * d_after / P.H)
                       - (-self.pot_coef * d_before / P.H))
        reward -= self.time_cost * self.action_repeat / P.FPS
        if th:
            reward -= self.thrust_cost * self.action_repeat / P.FPS
        truncated = self._frames >= self.max_frames
        return self._obs(), reward, terminated, truncated, dict(self._info)
