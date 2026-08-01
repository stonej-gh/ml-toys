#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, stage 0: a gravity sim in about 100 lines.

One body orbits a fixed central mass under inverse-square gravity, stepped
with semi-implicit Euler (velocity first, then position). The script checks
Kepler's third law numerically across several orbit sizes, then shows why the
integrator choice matters: rerun with --euler and the same orbits leak energy
until they are not orbits any more.

Run from the repo root:
    python exercises/stage0-gravity/solution.py           # the working sim
    python exercises/stage0-gravity/solution.py --euler   # break it on purpose

Walkthrough: README.md in this directory.
"""

import math
import sys

MU = 1.0        # G*M of the central mass; units are ours to choose
DT = 1e-3       # timestep, small enough that the wobble stays tiny
ORBITS = 20     # how long the energy-drift check runs


def accel(x, y):
    """Inverse-square pull toward the origin."""
    r = math.hypot(x, y)
    a = -MU / (r * r * r)  # -MU/r^2 along the unit vector (x,y)/r
    return a * x, a * y


def step_semi_implicit(x, y, vx, vy):
    """Velocity first, then position with the NEW velocity. Stable."""
    ax, ay = accel(x, y)
    vx, vy = vx + ax * DT, vy + ay * DT
    return x + vx * DT, y + vy * DT, vx, vy


def step_explicit(x, y, vx, vy):
    """Position with the OLD velocity. Looks almost identical; is not."""
    ax, ay = accel(x, y)
    x, y = x + vx * DT, y + vy * DT
    return x, y, vx + ax * DT, vy + ay * DT


def energy(x, y, vx, vy):
    """Specific orbital energy: kinetic minus potential. Constant on a real orbit."""
    return 0.5 * (vx * vx + vy * vy) - MU / math.hypot(x, y)


def measure_period(r0, step):
    """Start on a circular orbit of radius r0 and time one full trip around.

    The polar angle is accumulated step by step; when it passes 2*pi the trip
    is done, with linear interpolation across the final step for accuracy."""
    x, y = r0, 0.0
    vx, vy = 0.0, math.sqrt(MU / r0)  # circular speed at r0
    swept, t, prev = 0.0, 0.0, math.atan2(y, x)
    while True:
        x, y, vx, vy = step(x, y, vx, vy)
        t += DT
        ang = math.atan2(y, x)
        d = ang - prev
        if d < -math.pi:
            d += 2 * math.pi
        prev = ang
        swept += d
        if swept >= 2 * math.pi:
            return t - DT * (swept - 2 * math.pi) / d


def main():
    step = step_explicit if "--euler" in sys.argv else step_semi_implicit
    name = "explicit Euler" if step is step_explicit else "semi-implicit Euler"
    print(f"integrator: {name}\n")

    # Kepler's third law: T^2 proportional to r^3, so log(T) vs log(r) has slope 3/2
    radii = [0.7, 1.0, 1.4, 2.0]
    periods = [measure_period(r, step) for r in radii]
    for r, T in zip(radii, periods):
        print(f"  r = {r:4.2f}   T = {T:8.4f}   T^2/r^3 = {T * T / r ** 3:.4f}")
    slope = (math.log(periods[-1]) - math.log(periods[0])) \
        / (math.log(radii[-1]) - math.log(radii[0]))
    print(f"log-log slope: {slope:.4f}  (Kepler's third law says 1.5)\n")

    # energy drift over ORBITS trips around: the integrator test that matters.
    # The orbit is deliberately ECCENTRIC (85% of circular speed) because that
    # is where the two integrators tell their stories: the stable one wobbles
    # inside a fixed band and comes home; the broken one drifts and leaves.
    x, y, vx, vy = 1.0, 0.0, 0.0, 0.85 * math.sqrt(MU)
    e0 = energy(x, y, vx, vy)
    lo = hi = e0
    steps = int(ORBITS * 2 * math.pi / DT)
    for _ in range(steps):
        x, y, vx, vy = step(x, y, vx, vy)
        e = energy(x, y, vx, vy)
        lo, hi = min(lo, e), max(hi, e)
    e1 = energy(x, y, vx, vy)
    print(f"energy over {ORBITS} orbit-times (eccentric start): "
          f"start {e0:+.6f}  end {e1:+.6f}")
    print(f"  net drift {100 * (e1 - e0) / abs(e0):+.3f}%   "
          f"wobble band {100 * (hi - lo) / abs(e0):.3f}%")
    print(f"  final radius {math.hypot(x, y):.4f}  (started at 1.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
