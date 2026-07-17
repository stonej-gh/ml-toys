#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Experiment 01 orchestrator: retrain the survive DQN, then print the
eval table (learned vs random vs coast) and the gate verdict.

    python experiments/01-survive/run.py [--steps N] [--run-dir runs/survive]

Training is agents/dqn_survive.py, unmodified; this script only sets the
budget and runs the same fixed-seed evaluation the grader uses.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None,
                    help="training steps (default: the trainer's 600k)")
    ap.add_argument("--run-dir", default="runs/survive")
    ap.add_argument("--skip-train", action="store_true",
                    help="only evaluate an existing runs/survive/final.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not args.skip_train:
        import agents.dqn_survive as dqn
        if args.steps:
            dqn.TOTAL_STEPS = args.steps
        dqn.train(run_dir)

    from orbitduel.netpilot import PolicyNet
    ckpt = run_dir / "final.json"
    if not ckpt.exists():
        ckpt = X.MODELS / "survive_dqn_v2.json"
        print(f"no retrain found; evaluating shipped {ckpt.name}")
    net = PolicyNet(ckpt)
    rng = random.Random(2)
    rows = [("learned (greedy)", net.act),
            ("random", lambda o: rng.randrange(6)),
            ("coast", lambda o: 2)]
    stats = {}
    for name, pol in rows:
        s = X.survive_stats(pol, episodes=50)
        stats[name] = s
        print(f"{name:>16}: median {s['median_s']:6.2f}s  "
              f"mean {s['mean_s']:6.2f}s  "
              f"survived-to-cap {s['survived_frac'] * 100:3.0f}%")
    ok = (stats["learned (greedy)"]["survived_frac"] >= 0.80 and
          stats["learned (greedy)"]["median_s"]
          >= 10 * stats["random"]["median_s"])
    print("GATE (>=80% survival, median >= 10x random):",
          "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
