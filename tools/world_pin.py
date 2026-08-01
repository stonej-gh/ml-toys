#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The world pin: a short digest of the world every measured table depends on.

Every "measured" number in the experiment READMEs and exercise walkthroughs
was produced under a particular arena: this physics, this scripted opponent,
these rules. Change any of those and the numbers die silently while the docs
keep quoting them, which happened once: the 2026-07-31 opponent port
invalidated three experiments' tables and nobody noticed for a day. So the
docs now cite the digest of the world they measured under, and
tests/test_world_pin.py recomputes it on every run: if the world moves, the
test fails and NAMES every document that is now due for re-measurement.

The probes need no checkpoint and no torch: a fixed cycling policy flies
seeded duels against the scripted ladder at three rungs, plus seeded survive
rollouts, and the pin is a hash of their outcomes (winner, frames, wall
touches, lifetimes). Outcome-level results are the same granularity the
golden evals assert cross-platform, so the pin is stable everywhere the
suite is.

Run from the repo root to see the current pin and the probe table:
    python tools/world_pin.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orbitduel.env import OrbitDuelEnv
from orbitduel.pilot import ScriptedPilot
from orbitduel.survive import SurviveEnv
from orbitduel import physics as P


def config_rows():
    """The constants themselves, belt and braces: a reward term the behavior
    probes never trigger (a wall price when no probe touches a wall) would
    otherwise sit outside the pin."""
    env = OrbitDuelEnv(opponent=ScriptedPilot(1, seed=0), rules="v6-full",
                       seed=0)
    return [("reward", env.wall_penalty, env.hit_reward, env.pot_coef,
             env.time_cost, env.thrust_cost),
            ("physics", P.FIELD_NAME, round(P.GM, 6), round(P.SPAWN_V, 6),
             round(P.HOLE_R, 6), round(P.SHIP_R_AGENT, 6),
             round(P.SHIP_R_OPP, 6), P.TURN_RATE, round(P.LASER_SPEED, 6),
             round(P.LASER_TTL, 6))]


def probe_rows():
    """-> result tuples from seeded, checkpoint-free probes.

    Two kinds. EPISODE probes run a fixed cycling policy through the env and
    record how the episode ends, which pins the physics, the rules preset,
    and the termination logic. LADDER probes ask every scripted rung, L1
    through L10, for its commands at sampled instants along a live seeded
    trajectory, which pins the opponent's behavior directly: this is the part
    that would have caught the 2026-07-31 opponent port on day one."""
    rows = []

    for seed in (11, 12):
        env = OrbitDuelEnv(opponent=ScriptedPilot(5, seed=seed),
                           rules="v6-full", seed=seed, max_wall_seconds=30.0)
        env.reset(seed=seed)

        # one probe pilot per rung rides along and is queried, never obeyed
        probes = [ScriptedPilot(lv, seed=1000 + lv) for lv in range(1, 11)]
        t, total_r = 0, 0.0
        while True:
            _, r, term, trunc, info = env.step((t // 5) % 12)
            t += 1
            total_r += r
            if t % 20 == 0:
                for lv, p in enumerate(probes, start=1):
                    turn, thrust, fire = p(env.arena, 1)
                    rows.append(("ladder", seed, t, lv, round(turn, 3),
                                 bool(thrust), bool(fire)))
            if term or trunc:
                break

        # the summed reward pins the reward function, which outcomes cannot
        # see: a wall-price or shaping change must move this digest too
        rows.append(("episode", seed, info.get("outcome", "draw"),
                     info["frames"], info["wall_touches"], round(total_r, 4)))

    # survive probes: the same cycling idea on the one-ship task
    for seed in (21, 22, 23):
        env = SurviveEnv(seed=seed, max_wall_seconds=30.0)
        env.reset()
        t, total_r = 0, 0.0
        while True:
            _, r, term, trunc, _ = env.step((t // 3) % 6)
            t += 1
            total_r += r
            if term or trunc:
                break
        rows.append(("survive", seed, "timeout" if trunc else "died",
                     env._frames, round(total_r, 4)))
    return rows


def current_pin():
    """-> (12-hex digest, rows). The digest is what the docs cite."""
    rows = config_rows() + probe_rows()
    digest = hashlib.sha256(repr(rows).encode()).hexdigest()[:12]
    return digest, rows


def main():
    pin, rows = current_pin()
    for row in rows:
        print("  ", row)
    print(f"\nworld pin: {pin}")
    print("cite it next to a measured table as: world pin `" + pin + "`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
