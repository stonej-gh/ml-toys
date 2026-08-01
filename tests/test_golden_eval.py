# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Golden evaluation: shipped champions, played through the pure-Python
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
# pass. The row pins what it always pinned, exact per-seed outcomes, but its
# discriminating power is mostly gone. The ported-champion row below is what
# carries that job now.
GOLDEN_V6_L10 = ["loss", "loss", "loss", "loss", "loss"]

# Locked 2026-08-01. This is the anchor with teeth. The ported champion is
# competitive against L10 rather than swept by it, so the row is a MIXED
# sequence and every entry is a decision the physics had to get right: a
# platform that diverged numerically would flip at least one of these long
# before it changed a win rate. Twelve episodes rather than five for the same
# reason, more edges to catch a divergence on.
#
# Every win here is a laser kill and every loss is the black hole, at a median
# of zero wall touches, so the row also pins that the champion is winning by
# flying rather than by the experiment 02 exploit.
GOLDEN_PORTED_L10 = ["loss", "win", "win", "loss", "win", "win",
                     "win", "loss", "loss", "win", "loss", "win"]


def test_v6_vs_l10_golden(tmp_path):
    outcomes = run(model=str(MODELS / "duel_ppo_v6_final.json"), level=10,
                   episodes=5, rules="v6-full", out=tmp_path, record=0,
                   seed0=90_000)
    assert outcomes == GOLDEN_V6_L10


def test_ported_champion_vs_l10_golden(tmp_path):
    outcomes = run(model=str(MODELS / "duel_ppo_ported_phone.json"), level=10,
                   episodes=12, rules="v6-full", out=tmp_path, record=0,
                   seed0=90_000)
    assert outcomes == GOLDEN_PORTED_L10


def test_the_quickstart_default_actually_wins(tmp_path):
    """`python agents/duel_eval.py` with no arguments must beat the top rung.

    The README promises a champion taking on the top scripted bot and then hands
    a first-time reader exactly this command. That promise was silently false
    for a while: the default pointed at the v6 era champion, and once the
    opponent was ported to the game's own robot, the repo's first command
    printed nine straight losses. Nothing caught it, because every other test
    names its model explicitly and so never exercises the DEFAULT.

    This asserts the wiring, not the policy. It reads DEFAULT_MODEL and the
    parser's own defaults rather than restating them, so pointing the default at
    a checkpoint that cannot clear the top rung fails here.
    """
    import agents.duel_eval as D

    outcomes = run(model=str(D.DEFAULT_MODEL), level=10, episodes=9,
                   rules="v6-full", out=tmp_path, record=0, seed0=90_000)
    wins, losses = outcomes.count("win"), outcomes.count("loss")
    assert wins > losses, (
        f"the quickstart's default loses to L10: {wins}W/{losses}L with "
        f"{Path(D.DEFAULT_MODEL).name}")


def test_v6_never_loses_to_l1(tmp_path):
    outcomes = run(model=str(MODELS / "duel_ppo_v6_final.json"), level=1,
                   episodes=5, rules="v6-full", out=tmp_path, record=0,
                   seed0=91_000)
    assert "loss" not in outcomes
