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
# (per-hull collision radii, curved-laser default, exact muzzle law) - the
# same v6 champion, new per-seed outcomes on the new world (still 4/5 vs L10).
GOLDEN_V6_L10 = ["win", "loss", "win", "win", "win"]


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
