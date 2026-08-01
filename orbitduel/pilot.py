# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The scripted opponent ladder: a hand-tuned robot pilot, levels 1-10.

A behavioral port of the original game's per-level bot specification (the
numbers come from its difficulty table, the code is written fresh for this
arena). What's kept:

- The two-competency split: flight/survival is difficulty-INDEPENDENT and
  always preempts dueling; only dueling scales with level.
- The priority ladder: P1 peel off a wall -> P2 fit the free orbit to the
  arena (forward coast prediction, lift periapsis / pull off the wall) ->
  P3 orbital-phasing chase (off at L<=4, ramping in to co-orbital by L10) ->
  else coast and fight.
- One action per beat with a level-scaled action gap; rate-capped aim with a
  held wobble (refreshed every 0.45 wall s); level-scaled fire cooldown,
  lead-aim blend, and 0.5h range gate; hole-shot guard (never fire a shot
  that would curve into the hole).

The pilot returns per-frame actions (turn rad/phys s, thrust, fire) for
Arena.step; call it every frame, one instance per ship.
"""

import math

from . import physics as P

# spec table anchors (wall-time numbers converted to physics time where used)
FLIGHT_TURN = 2 * math.pi / P.GAME_SPEED  # survival slew ~1 rev/wall s
AIM_TURN_L1, AIM_TURN_L10 = 0.48, 1.00    # aim cap, rev/wall s
AIM_ERR_L1 = math.radians(31.5)           # held wobble band at t=0
COOLDOWN_L1, COOLDOWN_L10 = 1.30, 0.30    # wall s
GAP_L1, GAP_L10 = 0.220, 0.005            # action-switch gap, wall s
WOBBLE_REFRESH = 0.45                     # wall s
RANGE_GATE = 0.5 * P.H
AIM_TOL = 0.12                            # rad: within this counts as aimed
HOLE_CLEAR = P.HOLE_R * 2.4               # periapsis must clear this
PREDICT_STEPS = 240                       # 4.0 physics s of coast lookahead
PREDICT_DT = 1 / 60                       # ...at the game's own step size
GUARD_STEPS = 80                          # ~1.3 physics s of shot lookahead
GUARD_DT = 1 / 60
CHASE_STANDOFF = 0.30                     # early-band standoff fraction
CHASE_K_PHASE = 0.55                      # r* offset per radian of phase error
CHASE_MAX_OFF = 0.45                      # clamp on the phase-driven offset
EASE_ZONE = 1.4                           # rad: arc a slew decelerates across
ORBIT_CAP = 0.38                          # safe orbit radius, as a fraction of H
MIGRATE_GAIN = 1.0                        # how hard a chase burn pulls r to r*
VR_CLAMP = 200.0                          # radial-speed ceiling (raw pt/s)

# Burn magnitudes and margins the game states in RAW scene units, so each one
# is divided by FIELD_SCALE where it is used.
WALL_INSET = 8.0                          # safe-box inset beyond the hull
PEEL_BURN = 260.0                         # P1: outward-facing burn off a wall
DEEP_BURN = 230.0                         # P2: already inside the clearance ring
LIFT_BURN = 80.0                          # P2: lifting a threatened periapsis
WALL_PULL_GAIN = 1.4                      # inward pull when the orbit hits a wall


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _slew(err, cap):
    """Turn-rate command for a quadratic ease-out slew onto a heading.

    The robot decelerates INTO its target over EASE_ZONE rather than holding
    the rate cap until it arrives: x is 1 far out and 0 on target, and x(2-x)
    is the taper. Without it a slew reads as mechanical flicking.

    The game applies a per-FRAME angular step; this arena wants a rate in rad
    per physics second, and the two differ by exactly DT, so the never-overshoot
    clamp that reads min(step, mag) there reads min(rate, mag / DT) here."""
    mag = abs(err)
    x = min(mag / EASE_ZONE, 1.0)
    return math.copysign(min(cap * x * (2 - x), mag / P.DT), err)


def _safe_box(ship_r):
    """Half-extents of the arena inset by the hull plus the game's own margin.
    Outside this box the robot counts as pinned on a wall."""
    m = ship_r + WALL_INSET / P.FIELD_SCALE
    return P.PLAY_W / 2 - m, P.H / 2 - m


def lead_direction(dx, dy, vrx, vry, s):
    """Unit direction to fire so a shot meets a coasting target at (dx, dy).

    Solves |D + vRel*tau| = s*tau for the smallest positive tau. The shot's
    speed s is measured RELATIVE to the shooter, because the round inherits the
    ship's velocity, which is why the caller passes a relative target velocity
    rather than the target's own. Gravity curvature is deliberately ignored:
    the shots really do bend, and reading that bend is the aiming edge a human
    keeps over the bot.

    Returns None when no positive-time intercept exists, for instance when the
    target outruns the round; the caller then aims at where the target is."""
    a = vrx * vrx + vry * vry - s * s
    b = 2.0 * (dx * vrx + dy * vry)
    c = dx * dx + dy * dy
    if abs(a) < 1e-6:                     # target speed ~= shot speed: linear, not quadratic
        if abs(b) < 1e-6:
            return None
        tau = -c / b
    else:
        disc = b * b - 4.0 * a * c
        if disc < 0.0:                    # no real intercept
            return None
        sq = math.sqrt(disc)
        roots = [r for r in ((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)) if r > 0.0]
        if not roots:
            return None
        tau = min(roots)
    if tau <= 0.0:
        return None
    fx, fy = dx + vrx * tau, dy + vry * tau
    m = math.hypot(fx, fy)
    if m <= 1e-6:
        return None
    return fx / m, fy / m


def predict_coast(x, y, vx, vy, g, ship_r, dt=PREDICT_DT, steps=PREDICT_STEPS):
    """Free-orbit lookahead: (closest approach to the hole, stays in the box).

    The robot does NOT integrate the true inverse-square field here. It coasts
    under a CONSTANT-magnitude inward pull, the one it has just measured at its
    own radius, because on hardware it cannot assume a field law. That
    approximation is kept deliberately: it decides when the bot panics, and an
    exact predictor would make this pilot more far-sighted than the real one.

    ship_r is the flying hull's own collision radius, so pass the pilot's
    ship.r rather than a module constant: the two hulls differ by 10.4 pt and
    both the wall test and the hole test are radius-sensitive."""
    half_w, half_h = _safe_box(ship_r)
    min_r, inside = math.hypot(x, y), True
    for _ in range(steps):
        r = max(math.hypot(x, y), 0.001)
        vx += -x / r * g * dt                 # semi-implicit Euler: pull, then move
        vy += -y / r * g * dt
        x += vx * dt
        y += vy * dt
        rr = math.hypot(x, y)
        min_r = min(min_r, rr)
        if abs(x) > half_w or abs(y) > half_h:
            inside = False                    # would clip a wall
            break
        if rr < HOLE_CLEAR:
            break                             # already doomed, stop early
    return min_r, inside


def would_enter_hole(ship, heading=None, guard=1.5, g=None,
                     steps=GUARD_STEPS, dt=GUARD_DT):
    """Forward-sim the candidate shot; True if it passes inside guard x holeR.

    Three details here are the robot's approximations rather than the arena's
    truth, and all three are deliberate. It flies the round from its own CENTRE,
    not from the muzzle. It bends the round under the same constant-magnitude
    field its coast predictor uses, rather than the exact one. And it always
    simulates a fixed 80 steps of 1/60, about 1.3 physics seconds, which is
    close to twice a phone round's actual lifetime, so the guard is more
    cautious than the round it is checking.

    Checked against the heading the bot WANTS, wobble included, which is the
    shot it is about to take and not merely where its nose points now.

    Difficulty-independent, like survival flight: a beginner does not waste
    shots into the hole either."""
    if guard <= 0:
        return False
    if heading is None:
        nx, ny = ship.nose()
    else:
        nx, ny = -math.sin(heading), math.cos(heading)
    x, y = ship.x, ship.y
    if g is None:
        r0 = max(math.hypot(x, y), 1e-9)
        g = P.GM / (r0 * r0)
    kill = P.HOLE_R * guard
    vx, vy = ship.vx + nx * P.LASER_SPEED, ship.vy + ny * P.LASER_SPEED
    for _ in range(steps):
        r = max(math.hypot(x, y), 0.001)
        if r < kill:
            return True
        vx += -x / r * g * dt
        vy += -y / r * g * dt
        x += vx * dt
        y += vy * dt
    return False


class ScriptedPilot:
    """One instance per ship; call pilot(arena, idx) every frame."""

    def __init__(self, level=1, seed=0):
        self.level = max(1, level)
        skill = max(1, self.level - 1)    # displayed -> duel skill
        t = min(1.0, (skill - 1) / 9.0)
        self.passive = self.level <= 1    # displayed 1 never fires
        self.aim_turn = (AIM_TURN_L1 + (AIM_TURN_L10 - AIM_TURN_L1) * t) \
            * 2 * math.pi / P.GAME_SPEED  # rev/wall s -> rad/phys s
        self.aim_err = AIM_ERR_L1 * (1 - t)
        self.cooldown = (COOLDOWN_L1 + (COOLDOWN_L10 - COOLDOWN_L1) * t) * P.GAME_SPEED
        self.gap = (GAP_L1 + (GAP_L10 - GAP_L1) * t) * P.GAME_SPEED
        self.lead = t
        self.chase = max(0.0, (t - 0.33) / 0.67)

        # deterministic per-instance wobble stream (no globals, replayable)
        self._rand_state = (seed * 2654435761 + 12345) & 0xFFFFFFFF
        self.wobble = 0.0
        self.wobble_t = -1e9
        self.fire_t = -1e9

        # one action a beat: 1 aim, 2 throttle, 3 fire/idle. A switch between
        # them cannot happen until the dwell deadline passes.
        self.mode = 0
        self.deadline = 0.0
        self._pred_t = -1e9  # coast-prediction cache
        self._pred = (1e9, True)

    def _rand(self):         # xorshift, uniform [0,1)
        x = self._rand_state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._rand_state = x
        return x / 0xFFFFFFFF

    # the per-frame brain: ONE action a beat, aim or throttle or fire
    def __call__(self, arena, idx):
        me = arena.ships[idx]
        foe = arena.ships[1 - idx]
        now = arena.time
        if not me.alive:
            return (0.0, 0, 0)
        r = me.radius() or 1e-9

        # the pull the robot would measure right here: it reads dv/dt while
        # coasting, which at this radius is exactly this magnitude
        g_est = P.GM / (r * r)
        half_w, half_h = _safe_box(me.r)
        r_wall_safe = min(HOLE_CLEAR * 1.6, P.H * ORBIT_CAP)
        near_wall = abs(me.x) > half_w or abs(me.y) > half_h

        if now - self.wobble_t > WOBBLE_REFRESH * P.GAME_SPEED:
            self.wobble = (self._rand() * 2 - 1) * self.aim_err
            self.wobble_t = now

        # The orbit the thruster should be shaping, expressed as the velocity
        # this ship WANTS. Default is its current velocity, which means coast.
        # Survival outranks the chase, and the hole outranks the wall.
        des_vx, des_vy = me.vx, me.vy
        orbit_unsafe = False
        chasing = False
        if near_wall:
            # P1: a slow ship hugging an edge traces too short a path for the
            # predictor to flag it, so this hard override is what peels it off.
            des_vx, des_vy = self._orbit_velocity(
                me, r, -PEEL_BURN / P.FIELD_SCALE,
                math.sqrt(max(g_est, 1e-4) * r_wall_safe))
            orbit_unsafe = True
        else:
            if now - self._pred_t > 0.05:   # re-predict every 3 frames
                self._pred = predict_coast(me.x, me.y, me.vx, me.vy, g_est, me.r)
                self._pred_t = now
            min_r, inside = self._pred
            if min_r < HOLE_CLEAR:
                # P2a: lift a periapsis that threatens the hole, harder when
                # the ship is already inside the clearance ring
                deep = r < HOLE_CLEAR * 1.15
                v_c = math.sqrt(g_est * r)
                des_vx, des_vy = self._orbit_velocity(
                    me, r, (DEEP_BURN if deep else LIFT_BURN) / P.FIELD_SCALE,
                    max(self._tangential(me, r), v_c * (1.30 if deep else 1.10)))
                orbit_unsafe = True
            elif not inside:
                # P2b: pull in to a radius whose circle actually fits the arena
                pull = max(-VR_CLAMP / P.FIELD_SCALE,
                           min(0.0, -WALL_PULL_GAIN * (r - r_wall_safe)))
                des_vx, des_vy = self._orbit_velocity(
                    me, r, pull, math.sqrt(g_est * r_wall_safe))
                orbit_unsafe = True
            elif self.chase > 0.0 and foe.alive:
                # P3: close on the foe by orbital phasing
                r_star = self._chase_radius(me, foe, r_wall_safe)
                migrate = 0.4 + MIGRATE_GAIN * self.chase
                lim = VR_CLAMP / P.FIELD_SCALE
                des_vx, des_vy = self._orbit_velocity(
                    me, r, max(-lim, min(lim, -migrate * (r - r_star))),
                    math.sqrt(g_est * max(r_star, 1.0)))
                chasing = True

        ex, ey = des_vx - me.vx, des_vy - me.vy
        deadband = (16.0 if me.thrusting else 40.0) / P.FIELD_SCALE
        need_thrust = (orbit_unsafe or chasing) and math.hypot(ex, ey) > deadband

        dx, dy = foe.x - me.x, foe.y - me.y
        dist = max(math.hypot(dx, dy), 1.0)
        aim = self._enemy_heading(me, foe, dx, dy, dist)

        # one goal this beat: fix the orbit if it needs fixing, else aim
        goal = math.atan2(-ex, ey) if need_thrust else aim
        turn_cap = FLIGHT_TURN if need_thrust else self.aim_turn
        gap = 0.0 if need_thrust else self.gap
        can_fire = (not orbit_unsafe and not self.passive and foe.alive
                    and r > P.HOLE_R * 1.8 and not near_wall
                    and dist < RANGE_GATE)

        # a little extra tolerance once already burning, so the bot does not
        # stutter between aiming and throttling on small heading drift
        tol = AIM_TOL * (2.5 if self.mode == 2 else 1.0)
        d_goal = _wrap(goal - me.heading)
        desired = 1 if abs(d_goal) >= tol else (2 if need_thrust else 3)

        # a brief gap between SWITCHING actions: coast through it
        if desired != self.mode:
            if now < self.deadline:
                return (0.0, 0, 0)
            self.mode = desired
            self.deadline = now + gap

        if self.mode == 1:                  # AIM: turn, no throttle, no shot
            return (_slew(d_goal, turn_cap), 0, 0)
        if self.mode == 2:                  # THROTTLE: burn along the heading
            return (0.0, 1, 0)

        fire = 0                            # FIRE / idle: lined up, so shoot
        if can_fire and now - self.fire_t >= self.cooldown \
           and not would_enter_hole(me, aim, g=g_est):
            fire = 1
            self.fire_t = now
        return (0.0, 0, fire)

    # geometry helpers
    def _orbit_velocity(self, me, r, v_r, v_t):
        """Build a desired velocity from radial and tangential components,
        keeping the ship's current sense of sweep around the hole."""
        ux, uy = me.x / r, me.y / r
        sense = 1 if (me.x * me.vy - me.y * me.vx) >= 0 else -1
        tx, ty = -uy * sense, ux * sense      # keep the current sweep sense
        return ux * v_r + tx * v_t, uy * v_r + ty * v_t

    def _tangential(self, me, r):
        """This ship's circulation speed about the hole."""
        ux, uy = me.x / r, me.y / r
        sense = 1 if (me.x * me.vy - me.y * me.vx) >= 0 else -1
        return me.vx * -uy * sense + me.vy * ux * sense

    def _chase_radius(self, me, foe, r_wall_safe):
        """Target radius from the SIGNED phase to the foe.

        Positive phase means the foe is ahead in our direction of travel, so
        drop inside its orbit to catch up (a smaller orbit sweeps angularly
        faster); negative means fall back by rising outside. The offset is
        clamped so the chase can never fling the orbit across the arena, and
        the floor and ceiling mean safety always outranks the chase."""
        r_foe = max(foe.radius(), 1.0)
        sense = 1 if (me.x * me.vy - me.y * me.vx) >= 0 else -1
        phase = sense * _wrap(math.atan2(foe.y, foe.x) - math.atan2(me.y, me.x))
        standoff = CHASE_STANDOFF * (1 - self.chase)
        off = max(-CHASE_MAX_OFF,
                  min(CHASE_MAX_OFF, -CHASE_K_PHASE * phase)) * self.chase
        return max(HOLE_CLEAR * 1.25,
                   min(r_wall_safe, r_foe * (1 + standoff) * (1 + off)))

    def _enemy_heading(self, me, foe, dx, dy, dist):
        """Where to point to hit the foe, wobble included.

        Difficulty phases the lead in by blending the two UNIT directions,
        aim-at-current toward full intercept, so t=0 shoots where the foe is
        and t=1 takes the whole solution."""
        fx, fy = dx / dist, dy / dist
        ld = lead_direction(dx, dy, foe.vx - me.vx, foe.vy - me.vy, P.LASER_SPEED)
        if ld is not None:
            fx = fx * (1 - self.lead) + ld[0] * self.lead
            fy = fy * (1 - self.lead) + ld[1] * self.lead
        return math.atan2(-fx, fy) + self.wobble   # nose 0 = +y
