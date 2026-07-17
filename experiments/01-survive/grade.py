# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Grader for experiment 01 (survive). Run: pytest -m grade_cheap.

Grades runs/survive/final.json when you have retrained, else the shipped
survive_dqn_v2 checkpoint. Pure-Python inference: the shipped checkpoint's
numbers are platform-exact (measured on the reference platform 2026-07-10:
median 60.0 s, survived 42/50, random median 2.025 s)."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402

EPISODES = 50
SURVIVAL_GATE = 0.80
MEDIAN_VS_RANDOM = 10.0


def checkpoint():
    retrain = X.REPO / "runs" / "survive" / "final.json"
    return retrain if retrain.exists() else X.MODELS / "survive_dqn_v2.json"


@pytest.mark.grade_cheap
def test_survive_beats_doom():
    from orbitduel.netpilot import PolicyNet
    ckpt = checkpoint()
    learned = X.survive_stats(PolicyNet(ckpt).act, EPISODES)
    rng = random.Random(2)
    rand = X.survive_stats(lambda o: rng.randrange(6), EPISODES)
    print(f"\n{ckpt.name}: {learned}  |  random: {rand}")
    assert learned["survived_frac"] >= SURVIVAL_GATE, \
        f"survival {learned['survived_frac']:.0%} < {SURVIVAL_GATE:.0%}"
    assert learned["median_s"] >= MEDIAN_VS_RANDOM * rand["median_s"], \
        f"median {learned['median_s']:.1f}s not 10x random {rand['median_s']:.1f}s"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "grade_cheap", "-q", "-s"]))
