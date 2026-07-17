# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Grader for experiment 03 (curriculum forgetting vs league retention).
Run: pytest -m grade_cheap.

All inference is pure Python (orbitduel/netpilot.py), so each cell is an
exact integer on every platform. Reference matrix, 12 episodes/cell, re-pinned
2026-07-16 to the current physics: v1 curriculum 2/3/10/9 across L1/L4/L7/L10
(the ladder-climber forgets the easy rungs); v3 league 12/12/12/12 (flat
retention); v6 full champion 12/11/9/6 (strong low, tapering toward L10)."""

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
    """The league model holds the whole panel at once. The full champion is
    strong at the low ladder and tapers toward the hardest rungs."""
    # League training: flat retention across every rung.
    v3 = row("duel_ppo_v3_league.json", "v3-rude")
    print(f"\nduel_ppo_v3_league.json @ v3-rude: {v3}")
    for lv in PANEL:
        assert v3[lv] >= 11, f"league slipped vs L{lv}: {v3[lv]}/12 < 11"
    # v6 full champion: strong at the bottom, a monotone taper toward L10.
    v6 = row("duel_ppo_v6_final.json", "v6-full")
    print(f"duel_ppo_v6_final.json @ v6-full: {v6}")
    assert v6[1] >= 11 and v6[4] >= 10, f"v6 should own the low ladder: {v6}"
    assert v6[10] >= 5, f"v6 should stay net-positive vs L10: {v6[10]}/12"
    assert v6[1] >= v6[4] >= v6[7] >= v6[10], f"v6 taper not monotone: {v6}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "grade_cheap", "-q", "-s"]))
