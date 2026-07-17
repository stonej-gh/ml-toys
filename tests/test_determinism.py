# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Same seed => bit-identical rollouts. This is a hard promise of the env
(stdlib math throughout the physics core), asserted exactly, not approximately."""

import random

import numpy as np

from orbitduel.env import OrbitDuelEnv
from orbitduel.pilot import ScriptedPilot


def rollout(seed, steps=400):
    env = OrbitDuelEnv(opponent=ScriptedPilot(7, seed=seed), rules="v6-full",
                       seed=seed)
    rng = random.Random(seed)
    obs, _ = env.reset(seed=seed)
    trace = [obs]
    rewards = []
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(rng.randrange(12))
        trace.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return np.stack(trace), rewards, info


def test_same_seed_bit_identical():
    t1, r1, i1 = rollout(123)
    t2, r2, i2 = rollout(123)
    assert np.array_equal(t1, t2)
    assert r1 == r2
    assert i1 == i2


def test_different_seed_diverges():
    t1, _, _ = rollout(123)
    t2, _, _ = rollout(124)
    assert t1.shape != t2.shape or not np.array_equal(t1, t2)


def test_rule_presets_apply():
    free = OrbitDuelEnv(rules="v1-freewalls")
    assert free.arena.spin_kick == 0.0
    assert free.fuel is False and free.spawn_phase is False
    full = OrbitDuelEnv(rules="v6-full")
    assert full.arena.gravity_on_lasers is True and full.fuel is True
    # explicit kwargs beat the preset
    override = OrbitDuelEnv(rules="v6-full", gravity_on_lasers=False)
    assert override.arena.gravity_on_lasers is False
