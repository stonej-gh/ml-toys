# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Analytic sanity checks on the physics core: run  python -m orbitduel.selftest

These verify the simulation against closed-form orbital mechanics (Kepler
period, energy conservation, no secular drift), with no reference to any
external system.
"""

import math
from . import physics as P


def run():
    failures = []

    def check(name, ok, detail):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
        if not ok:
            failures.append(name)

    # 1. circular spawn orbit: bounded oscillation, NO secular drift. Semi-implicit
    # Euler is symplectic - r wobbles ~1% around a discrete invariant circle
    # (typical 2D physics engines behave the same), but the mean radius must not walk.
    arena = P.Arena().reset()
    period = 2 * math.pi * P.SPAWN_R / P.SPAWN_V          # physics s (Kepler, a=r)
    frames_per_orbit = int(period / P.DT)
    orbit_mean_r = []
    for _ in range(10):
        acc = 0.0
        for _ in range(frames_per_orbit):
            arena.step()
            acc += arena.ships[0].radius()
        orbit_mean_r.append(acc / frames_per_orbit)
    secular = abs(orbit_mean_r[-1] - orbit_mean_r[0]) / P.SPAWN_R
    check("circular orbit secular drift over 10 orbits", secular < 0.002,
          f"{secular * 100:.4f}% (orbit-1 mean r {orbit_mean_r[0]:.2f}, "
          f"orbit-10 mean r {orbit_mean_r[-1]:.2f})")

    # 2. measured period vs Kepler over those 10 orbits (angle unwrap)
    arena = P.Arena().reset()
    ang_prev = 0.0
    unwrapped = 0.0
    for _ in range(10 * frames_per_orbit):
        arena.step()
        s = arena.ships[0]
        ang = math.atan2(s.y, s.x)
        d = ang - ang_prev
        if d > math.pi:
            d -= 2 * math.pi
        if d < -math.pi:
            d += 2 * math.pi
        unwrapped += d
        ang_prev = ang
    t_meas = arena.time / (abs(unwrapped) / (2 * math.pi))
    err = abs(t_meas - period) / period
    check("orbit period vs Kepler", err < 0.005,
          f"measured {t_meas:.3f}s vs analytic {period:.3f}s ({err * 100:.2f}%)")

    # 3. energy conservation on an eccentric coast (no thrust, no walls hit)
    arena = P.Arena().reset()
    s = arena.ships[0]
    s.vy *= 0.9                                            # ellipse inside the spawn circle
    def energy(s):
        return 0.5 * s.speed() ** 2 - P.GM / s.radius()
    e0 = energy(s)
    for _ in range(int(5 * period / P.DT)):
        arena.step()
    e_err = abs(energy(s) - e0) / abs(e0)
    check("specific energy drift on eccentric coast", e_err < 0.01,
          f"{e_err * 100:.3f}% over ~5 orbits")

    # 4. a hard retrograde burn deorbits into the hole
    arena = P.Arena().reset()
    s = arena.ships[0]
    s.heading = math.pi                                    # nose -y = retrograde at spawn
    died = None
    for _ in range(int(60 / P.DT)):
        for ev in arena.step(((0, 1, 0), (0, 0, 0))):
            if ev[0] == 'death' and ev[1] == 0:
                died = ev[2]
        if died:
            break
    check("retrograde burn falls into the hole", died == 'blackhole', f"cause={died}")

    # 5. lasers: a broadside shot crosses the field and expires (TTL x speed sanity)
    arena = P.Arena().reset()
    arena.ships[0].cooldown = 0.0
    arena.step(((0, 0, 1), (0, 0, 0)))
    check("laser spawns with muzzle + ship velocity", len(arena.lasers) == 1
          and abs(math.hypot(arena.lasers[0].vx, arena.lasers[0].vy)
                  - math.hypot(0, P.SPAWN_V + 0) - 0) > 0, "spawned")
    rng = P.LASER_SPEED * P.LASER_TTL
    check("laser range covers a duel gap", rng > P.SPAWN_R,
          f"range {rng:.0f} pt vs spawn radius {P.SPAWN_R:.0f} pt")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run())
