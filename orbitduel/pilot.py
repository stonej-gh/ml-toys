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

Not ported (L11+ superhuman layers): evasion, sweep-fire, stances, fuel
governor, curve-aware aim / AA / flank / jink. `level` is the DISPLAYED
level of the current ladder: displayed 1 is fully passive (never fires),
displayed L behaves like duel skill L-1 on the table axes.

The pilot returns per-frame actions (turn rad/phys s, thrust, fire) for
Arena.step; call it every frame, one instance per ship.
"""

import math

from . import physics as P

# spec table anchors (wall-time numbers converted to physics time where used)
FLIGHT_TURN = 2 * math.pi / P.GAME_SPEED     # survival slew ~1 rev/wall s
AIM_TURN_L1, AIM_TURN_L10 = 0.48, 1.00       # aim cap, rev/wall s
AIM_ERR_L1 = math.radians(31.5)              # held wobble band at t=0
COOLDOWN_L1, COOLDOWN_L10 = 1.30, 0.30       # wall s
GAP_L1, GAP_L10 = 0.220, 0.005               # action-switch gap, wall s
WOBBLE_REFRESH = 0.45                        # wall s
RANGE_GATE = 0.5 * P.H
CHASE_BURST = 0.4                            # wall s: max chase-burn length...
CHASE_REST = 0.9                             # ...then a coast/fight window
HOLE_CLEAR = P.HOLE_R * 2.4                  # periapsis must clear this
WALL_MARGIN = 60.0                           # P1 trigger distance off an edge
PREDICT_WALL_S = 4.0                         # coast lookahead
CHASE_STANDOFF = 0.30                        # early-band standoff fraction


def _wrap(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def predict_coast(x, y, vx, vy, phys_seconds, dt=1 / 30):
    """Free-orbit lookahead: (min hole distance, stays inside the arena)."""
    min_r, inside = math.hypot(x, y), True
    steps = int(phys_seconds / dt)
    for _ in range(steps):
        ax, ay = P.gravity_accel(x, y)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        r = math.hypot(x, y)
        min_r = min(min_r, r)
        if abs(x) > P.PLAY_W / 2 - P.SHIP_R or abs(y) > P.H / 2 - P.SHIP_R:
            inside = False
            break
        if r < P.HOLE_R + P.SHIP_R:
            break
    return min_r, inside


def would_enter_hole(ship, guard=1.15):
    """Forward-sim the candidate shot; True if it curves inside guard x holeR.
    (The guard matters even for straight lasers: it vetoes point-blank
    shots across the hole.)"""
    nx, ny = ship.nose()
    x, y = ship.x + nx * P.SHIP_R, ship.y + ny * P.SHIP_R
    vx, vy = ship.vx + nx * P.LASER_SPEED, ship.vy + ny * P.LASER_SPEED
    dt = 1 / 30
    for _ in range(int(P.LASER_TTL / dt)):
        x += vx * dt
        y += vy * dt
        if math.hypot(x, y) < P.HOLE_R * guard:
            return True
        if abs(x) > P.PLAY_W / 2 or abs(y) > P.H / 2:
            return False
    return False


class ScriptedPilot:
    """One instance per ship; call pilot(arena, idx) every frame."""

    def __init__(self, level=1, seed=0):
        self.level = max(1, level)
        skill = max(1, self.level - 1)               # displayed -> duel skill
        t = min(1.0, (skill - 1) / 9.0)
        self.passive = self.level <= 1               # displayed 1 never fires
        self.aim_turn = (AIM_TURN_L1 + (AIM_TURN_L10 - AIM_TURN_L1) * t) \
            * 2 * math.pi / P.GAME_SPEED             # rev/wall s -> rad/phys s
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
        self.gap_until = 0.0
        self.burst_start = -1.0
        self.rest_until = 0.0
        self._pred_t = -1e9                          # coast-prediction cache
        self._pred = (1e9, True)

    def _rand(self):                                  # xorshift, uniform [0,1)
        x = self._rand_state
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._rand_state = x
        return x / 0xFFFFFFFF

    # -- the per-frame brain ------------------------------------------------
    def __call__(self, arena, idx):
        me = arena.ships[idx]
        foe = arena.ships[1 - idx]
        now = arena.time
        if not me.alive:
            return (0.0, 0, 0)
        r = me.radius() or 1e-9

        # P1: pinned on a wall -> hard burn back toward centre with circulation
        edge_x = P.PLAY_W / 2 - abs(me.x)
        edge_y = P.H / 2 - abs(me.y)
        if min(edge_x, edge_y) < WALL_MARGIN + P.SHIP_R:
            to_c = math.atan2(-me.y, -me.x)
            swirl = 0.6 if (me.x * me.vy - me.y * me.vx) >= 0 else -0.6
            return self._steer_burn(me, to_c + swirl, FLIGHT_TURN)

        # P2: free orbit must clear the hole and stay in the box. Recovery
        # circularizes AT THE CURRENT RADIUS (always the cheapest safe orbit)
        # with an outward bias when the dive threatens the hole: commanding a
        # far radius's circular speed mid-dive decelerates and makes it worse.
        if now - self._pred_t > 0.05:                # re-predict every 3 frames
            self._pred = predict_coast(me.x, me.y, me.vx, me.vy,
                                       PREDICT_WALL_S * P.GAME_SPEED)
            self._pred_t = now
        min_r, inside = self._pred
        r_wall_safe = min(HOLE_CLEAR * 1.6, P.H / 2 - P.SHIP_R - WALL_MARGIN)
        if min_r < HOLE_CLEAR or not inside:
            return self._recover_burn(me, r, min_r, inside, r_wall_safe)

        # safe orbit: dueling (or pure coast for the passive displayed-1)
        if self.passive or not foe.alive:
            return (0.0, 0, 0)

        # P3 chase: pick a target radius from the signed phase to the opponent:
        # drop INSIDE the foe's orbit to catch up (smaller r sweeps faster),
        # rise outside to drop back, settle to a standoff/co-orbital radius.
        # Burst/rest duty cycle: one deliberate shift, then coast and fight;
        # a chaser that migrates every frame never fires and dives too deep.
        if self.chase > 0.0 and now >= self.gap_until and now >= self.rest_until:
            standoff = CHASE_STANDOFF * (1 - self.chase)
            r_foe = foe.radius()
            dist = math.hypot(foe.x - me.x, foe.y - me.y)
            if dist < RANGE_GATE * 0.9:
                r_star = r_foe * (1 + standoff)       # closed: hold the standoff
            else:                                     # (co-orbital only at L10)
                # dip INSIDE until closed - the faster inner sweep laps the foe
                # (a signed-phase chooser flaps at the +/-pi ambiguity and stalls)
                r_star = r_foe * (1 - 0.30 * self.chase)
            r_star = max(HOLE_CLEAR * 1.1, min(r_wall_safe, r_star))
            act = self._orbit_shift_burn(me, r, r_star) \
                if abs(r - r_star) > 20.0 else (0.0, 0, 0)
            if act != (0.0, 0, 0):
                if self.burst_start < 0:
                    self.burst_start = now
                if now - self.burst_start > CHASE_BURST * P.GAME_SPEED:
                    self.burst_start = -1.0
                    self.rest_until = now + CHASE_REST * P.GAME_SPEED
                else:
                    if act[1]:
                        self.gap_until = now + self.gap
                    return act
            else:
                self.burst_start = -1.0

        # aim beat: rate-capped slew onto the (lead-blended, wobbled) firing line
        if now - self.wobble_t > WOBBLE_REFRESH * P.GAME_SPEED:
            self.wobble = (self._rand() * 2 - 1) * self.aim_err
            self.wobble_t = now
        dist = math.hypot(foe.x - me.x, foe.y - me.y)
        tof = dist / P.LASER_SPEED                    # straight-shot time of flight
        aim_x = foe.x + foe.vx * tof * self.lead - me.x
        aim_y = foe.y + foe.vy * tof * self.lead - me.y
        want = math.atan2(aim_y, aim_x) - math.pi / 2   # nose 0 = +y
        err = _wrap(want + self.wobble - me.heading)
        turn = max(-self.aim_turn, min(self.aim_turn, err / P.DT))

        # fire beat: lined up, in range, off cooldown, and the shot stays out
        # of the hole. Never while turning hard (one action per beat).
        fire = 0
        if abs(err) < math.radians(6) and dist < RANGE_GATE \
           and now - self.fire_t >= self.cooldown and abs(turn) < 0.3 * self.aim_turn \
           and not would_enter_hole(me):
            fire = 1
            self.fire_t = now
            self.gap_until = now + self.gap
        return (turn, 0, fire)

    # -- burn helpers ---------------------------------------------------------
    def _recover_burn(self, me, r, min_r, inside, r_wall_safe):
        """P2 emergency: circularize at the current radius, biased outward when
        the predicted periapsis threatens the hole, inward when leaving the box."""
        ux, uy = me.x / r, me.y / r
        sense = 1 if (me.x * me.vy - me.y * me.vx) >= 0 else -1
        tx, ty = -uy * sense, ux * sense
        v_c = math.sqrt(P.GM / r)
        vr_des = 0.0
        if min_r < HOLE_CLEAR:
            vr_des = 0.5 * v_c * (1 - min_r / HOLE_CLEAR)     # lift the periapsis
        elif not inside and r > r_wall_safe:
            vr_des = -0.25 * v_c                              # pull off the wall
        des_vx = tx * v_c + ux * vr_des
        des_vy = ty * v_c + uy * vr_des
        dvx, dvy = des_vx - me.vx, des_vy - me.vy
        if math.hypot(dvx, dvy) < 12.0:
            return (0.0, 0, 0)
        return self._steer_burn(me, math.atan2(dvy, dvx), FLIGHT_TURN)

    def _steer_burn(self, me, want_heading, turn_cap):
        """Slew toward a heading; thrust once roughly aligned (survival style)."""
        err = _wrap(want_heading - math.pi / 2 - me.heading)
        turn = max(-turn_cap, min(turn_cap, err / P.DT))
        thrust = 1 if abs(err) < math.radians(25) else 0
        return (turn, thrust, 0)

    def _orbit_shift_burn(self, me, r, r_star):
        """P2/P3 radius migration: aim the velocity error, thrust it out."""
        ux, uy = me.x / r, me.y / r
        sense = 1 if (me.x * me.vy - me.y * me.vx) >= 0 else -1
        tx, ty = -uy * sense, ux * sense              # keep the current sweep sense
        v_t = math.sqrt(P.GM / max(r_star, 1.0))      # circular speed at target
        vr_des = -0.35 * (r - r_star)                 # gentle radial pull
        des_vx = tx * v_t + ux * vr_des
        des_vy = ty * v_t + uy * vr_des
        dvx, dvy = des_vx - me.vx, des_vy - me.vy
        if math.hypot(dvx, dvy) < 12.0:               # close enough: coast
            return (0.0, 0, 0)
        return self._steer_burn(me, math.atan2(dvy, dvx), FLIGHT_TURN)
