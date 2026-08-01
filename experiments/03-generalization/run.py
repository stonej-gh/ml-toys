#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Experiment 03: every shipped duel checkpoint against the fixed scripted
panel, twice. Once under one common ruleset, which is the comparison the
experiment rests on, and once in each checkpoint's own era for reference.
Pure-Python inference, so both matrices are exact.

Each row prints wins and the median wall touches behind them, because a win
rate bought by wall-riding is experiment 02's exploit rather than a result.

    python experiments/03-generalization/run.py [--episodes N] [--era RULES]

The default 48 episodes a cell is what these claims need; the whole run takes
a few minutes. Smaller n is for a quick look, not for a number worth quoting.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402

PANEL = (1, 4, 7, 10)

# every shipped checkpoint, its trainer, and the rule era it trained under
CHECKPOINTS = [
    ("duel_ppo_v1.json", "curriculum", "v1-freewalls"),
    ("duel_ppo_v2_L3grad.json", "curriculum", "v3-rude"),
    ("duel_ppo_v3_league.json", "league", "v3-rude"),
    ("duel_ppo_v4_final.json", "league", "v4-honest"),
    ("duel_ppo_v5_curved.json", "league", "v6-full"),
    ("duel_ppo_v6_final.json", "league", "v6-full"),
]

# The ported champions are trained per field profile against the ported L1-L10
# ladder, so only the one matching ORBITDUEL_FIELD is meaningful here: judging
# the tablet champion in the phone world measures the wrong thing.
FIELD = os.environ.get("ORBITDUEL_FIELD", "phone")
CHECKPOINTS.append((f"duel_ppo_ported_{FIELD}.json", "league", "v6-full"))


def matrix(rows, episodes, title):
    head = "".join(f"   L{lv:<8}" for lv in PANEL)
    print(f"\n{title}")
    print(f"{'checkpoint':28s} {'trainer':11s} {'era':12s}{head}"
          f"   (wins/{episodes}, median wall touches)")
    for path, trainer, rules in rows:
        cells = []
        for level in PANEL:
            st = X.duel_stats(path, level, episodes, rules)
            cells.append(f"{X.wins(st):4d}/{X.median_walls(st):<5.1f}")
        print(f"{path.name:28s} {trainer:11s} {rules:12s}{''.join(cells)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--era", default="v6-full",
                    help="the common ruleset for the controlled matrix")
    ap.add_argument("--model", help="add your own exported .json to both matrices")
    ap.add_argument("--rules", default="v6-full", help="training era for --model")
    args = ap.parse_args()

    rows = [(X.MODELS / model, trainer, rules)
            for model, trainer, rules in CHECKPOINTS]
    if args.model:
        rows.append((Path(args.model), "yours", args.rules))

    matrix([(p, t, args.era) for p, t, _ in rows], args.episodes,
           f"Controlled: every checkpoint judged under {args.era}")
    matrix(rows, args.episodes,
           "Reference: each checkpoint in the era it trained under")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
