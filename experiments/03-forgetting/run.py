#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Experiment 03: the era-matched win matrix, every shipped duel checkpoint
vs the fixed scripted panel. Pure-Python inference; the matrix is exact.

    python experiments/03-forgetting/run.py [--episodes N]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402

PANEL = (1, 4, 7, 10)

# every shipped checkpoint, in the rule era it trained under
CHECKPOINTS = [
    ("duel_ppo_v1.json", "v1-freewalls"),
    ("duel_ppo_v2_L3grad.json", "v3-rude"),
    ("duel_ppo_v3_league.json", "v3-rude"),
    ("duel_ppo_v4_final.json", "v4-honest"),
    ("duel_ppo_v5_curved.json", "v6-full"),
    ("duel_ppo_v6_final.json", "v6-full"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--model", help="add your own exported .json to the matrix")
    ap.add_argument("--rules", default="v6-full", help="era preset for --model")
    args = ap.parse_args()

    rows = [(X.MODELS / model, rules) for model, rules in CHECKPOINTS]
    if args.model:
        rows.append((Path(args.model), args.rules))

    header = "".join(f"  L{lv:<2}" for lv in PANEL)
    print(f"{'checkpoint':28s} {'era':12s}{header}   (wins/{args.episodes})")
    for path, rules in rows:
        cells = []
        for level in PANEL:
            st = X.duel_stats(path, level, args.episodes, rules)
            cells.append(f"{X.wins(st):3d} ")
        print(f"{path.name:28s} {rules:12s}{''.join(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
