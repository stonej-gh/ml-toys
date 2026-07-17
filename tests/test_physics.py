# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The analytic selftest, as pytest: closed-form orbital mechanics checks."""

import math

from orbitduel import selftest
from orbitduel import physics as P


def test_analytic_selftest_passes():
    assert selftest.run() == 0


def test_spawn_orbit_is_circular():
    assert math.isclose(P.SPAWN_V, math.sqrt(P.GM / P.SPAWN_R))


def test_era_physics_knobs():
    free = P.Arena(spin_kick=0.0, hit_spin=0.0)
    assert free.spin_kick == 0.0 and free.hit_spin == 0.0
    modern = P.Arena()
    assert modern.spin_kick == P.SPIN_KICK and modern.hit_spin == P.HIT_SPIN
