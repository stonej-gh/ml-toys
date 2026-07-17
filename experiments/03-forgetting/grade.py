# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Grader for experiment 03 (curriculum forgetting vs league retention).
Run: pytest -m grade_cheap.

All inference is pure Python (orbitduel/netpilot.py), so each cell is an
exact integer on every platform. Reference matrix, 12 episodes/cell, locked
2026-07-10: v1 curriculum 6/7/8/8 across L1/L4/L7/L10; v3 league 12/12/12/12;
v6 champion 12/12/12/10."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402

EPISODES = 12
PANEL = (1, 4, 7, 10)


def row(model, rules):
    return {lv: X.wins(X.duel_stats(X.MODELS / model, lv, EPISODES, rules))
            for lv in PANEL}


@pytest.mark.grade_cheap
def test_curriculum_forgets_its_teachers():
    """The ladder-climber wins FEWER games against the easiest opponent than
    against the hardest: it forgot L1 while specializing on L10."""
    w = row("duel_ppo_v1.json", "v1-freewalls")
    print(f"\nv1 curriculum @ v1-freewalls: {w}")
    assert w[10] >= 7, f"ladder-climber should still hold L10, got {w[10]}/12"
    assert w[1] < w[10], \
        f"forgetting signature gone: L1 {w[1]} !< L10 {w[10]}"


@pytest.mark.grade_cheap
def test_league_retains_every_level():
    """League training holds the whole panel at once."""
    for model, rules, floor in [("duel_ppo_v3_league.json", "v3-rude", 11),
                                ("duel_ppo_v6_final.json", "v6-full", 10)]:
        w = row(model, rules)
        print(f"\n{model} @ {rules}: {w}")
        for lv in PANEL:
            assert w[lv] >= floor, \
                f"{model} slipped vs L{lv}: {w[lv]}/12 < {floor}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "grade_cheap", "-q", "-s"]))
