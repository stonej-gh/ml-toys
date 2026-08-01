# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Golden evaluation: the shipped v6 champion, played through the pure-Python
reference forward (orbitduel/netpilot.py), must produce EXACTLY these outcomes
on these seeds. Framework-free double-precision math end to end, so any
platform that disagrees has a real defect, not a tolerance issue."""

from pathlib import Path

from agents.duel_eval import run

MODELS = Path(__file__).resolve().parent.parent / "agents" / "models"

# Locked 2026-07-10 on the reference platform (macOS / CPython 3.14);
# re-pinned 2026-07-11 when the physics adopted the game's exact algorithm
# (per-hull collision radii, curved-laser default, exact muzzle law).
#
# Re-pinned again 2026-07-31, when the scripted pilot was ported to the real
# game robot's behaviour through level 10. The champion is unchanged; its
# opponent is not. v6 was trained against a sparring partner that could not
# lead a shot, decelerate into a slew, or stop chasing long enough to fire,
# and against the corrected one it no longer wins this row at all.
#
# NOTE, and it is a real weakness: an all-loss row is a poor determinism
# anchor. v6 now loses about 94 percent of episodes at L10, so a platform
# that disagreed numerically would very likely still produce five losses and
# pass. The row pins what it always pinned, exact per-seed outcomes, but the
# discriminating power is mostly gone until this anchors on something
# continuous (episode lengths) or on a champion that is competitive here.
GOLDEN_V6_L10 = ["loss", "loss", "loss", "loss", "loss"]


def test_v6_vs_l10_golden(tmp_path):
    outcomes = run(model=str(MODELS / "duel_ppo_v6_final.json"), level=10,
                   episodes=5, rules="v6-full", out=tmp_path, record=0,
                   seed0=90_000)
    assert outcomes == GOLDEN_V6_L10


def test_v6_never_loses_to_l1(tmp_path):
    outcomes = run(model=str(MODELS / "duel_ppo_v6_final.json"), level=1,
                   episodes=5, rules="v6-full", out=tmp_path, record=0,
                   seed0=91_000)
    assert "loss" not in outcomes
