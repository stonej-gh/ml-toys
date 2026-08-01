# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Grader for experiment 03 (does a league generalize across the opponent
panel where a ladder curriculum does not?). Run: pytest -m grade_cheap.

All inference is pure Python (orbitduel/netpilot.py), so each cell is
deterministic: two runs on one machine match byte for byte.

Measured 2026-07-31 at 48 episodes a cell, under one common era so the two
trainers are compared on identical rules. Across L1/L4/L7/L10 at `v6-full`:
v2 curriculum 45/22/4/2, v6 league 47/43/29/3, v3 league 46/42/36/21. Both
rows of the headline pair touch a wall about zero times an episode, so the
gap is not experiment 02's exploit wearing a different hat.

The n went up from 12 because 12 episodes could not carry these claims. The
12-episode v1 row changed direction three times during the 2026-07 opponent
port, and the same cells at 48 episodes disagree with it, so the old matrix
was reporting sampling noise as a result. Knife-edge episodes still flip
across environments, so the assertions are margins rather than the matrix.

The panel's top rung is the limit of what any shipped checkpoint reaches.
Every model here trained against the pre-port scripted opponent, and the
corrected L10 is an opponent none of them ever met."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402

EPISODES = 48
PANEL = (1, 4, 7, 10)

# the era every checkpoint is judged under in the controlled comparison
COMMON = "v6-full"


def cells(model, rules):
    """-> {level: per-episode stats}, one 48-episode eval a rung."""
    return {lv: X.duel_stats(X.MODELS / model, lv, EPISODES, rules)
            for lv in PANEL}


def row(model, rules):
    return {lv: X.wins(st) for lv, st in cells(model, rules).items()}


@pytest.mark.grade_cheap
def test_league_generalizes_where_the_ladder_does_not():
    """Same rules, same panel, same seeds: the ladder-trained checkpoint and
    the league champion are level against the easiest opponent, and they come
    apart as the panel moves away from it."""
    cur_st = cells("duel_ppo_v2_L3grad.json", COMMON)
    lg_st = cells("duel_ppo_v6_final.json", COMMON)
    cur = {lv: X.wins(st) for lv, st in cur_st.items()}
    lg = {lv: X.wins(st) for lv, st in lg_st.items()}
    print(f"\ncurriculum v2 @ {COMMON}: {cur}")
    print(f"league v6     @ {COMMON}: {lg}")

    # level at the bottom of the panel: neither trainer has an edge at L1
    assert abs(cur[1] - lg[1]) <= 6, f"L1 should be a wash: {cur[1]} vs {lg[1]}"

    # and apart in the middle, which is the whole result
    assert lg[4] - cur[4] >= 12, f"L4 gap too small: {lg[4]} vs {cur[4]}"
    assert lg[7] - cur[7] >= 15, f"L7 gap too small: {lg[7]} vs {cur[7]}"
    assert cur[7] <= 12, f"ladder row should fall off by L7, got {cur[7]}"

    # The gap has to be generalization rather than experiment 02's wall
    # exploit, so both rows must be flying clean. Bounds are per rung.
    for lv in PANEL:
        assert X.median_walls(cur_st[lv]) <= 1, \
            f"curriculum row is wall-riding at L{lv}"
        assert X.median_walls(lg_st[lv]) <= 1, \
            f"league row is wall-riding at L{lv}"

    # The limit of the claim: the ported top rung defeats both, and neither
    # trained against it. A champion retrained on the ported opponent should
    # break this assertion, and it will arrive as a NEW row rather than these.
    assert cur[10] <= 8 and lg[10] <= 8, \
        f"shipped checkpoints are not expected to hold L10: {cur[10]}, {lg[10]}"


@pytest.mark.grade_cheap
def test_league_holds_the_whole_panel():
    """In its own era the league model holds all four rungs at once. The full
    champion is strong at the low ladder and tapers toward the hardest."""

    # League training in its own era: the broadest row the repo ships.
    v3 = row("duel_ppo_v3_league.json", "v3-rude")
    print(f"\nduel_ppo_v3_league.json @ v3-rude: {v3}")
    assert v3[1] >= 44 and v3[4] >= 42, f"league should own the low ladder: {v3}"
    for lv in PANEL:
        assert v3[lv] >= 28, f"league collapsed vs L{lv}: {v3[lv]}/{EPISODES} < 28"

    # v6 full champion: strong at the bottom, a monotone taper toward L10.
    # There is deliberately no floor at L10: the ported L10 beats this
    # champion, and pretending otherwise is what the re-measure removed.
    #
    # This taper is asserted of THIS POLICY, not of the ladder. The rungs are
    # ordered by the scripted robot's parameters, and how steep that ordering
    # feels depends on who is climbing: two champion seeds measured the L7 to
    # L10 step 25 points apart, 5.64 se, so a monotone row is a property of the
    # (policy, ladder) pair. See the README's section on the panel mean.
    v6 = row("duel_ppo_v6_final.json", "v6-full")
    print(f"duel_ppo_v6_final.json @ v6-full: {v6}")
    assert v6[1] >= 44 and v6[4] >= 38, f"v6 should own the low ladder: {v6}"
    assert v6[1] >= v6[4] >= v6[7] >= v6[10], f"v6 taper not monotone: {v6}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "grade_cheap", "-q", "-s"]))
