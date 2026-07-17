#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Experiment 02, the live run: train a fresh PPO agent in the broken
free-walls world and watch it learn to cheat. Records browser-viewable
replays.

    python experiments/02-reward-hacking/run.py [--updates 120] [--seed 0]
        [--level 3] [--rules v1-freewalls] [--out runs/exp02-wallrider]

Defaults reproduce the reference emergence: by 120 updates the agent lives
on the wall (median ~13,800 touches per 120 s episode on the reference
platform). Try --rules v4-honest to watch the fixed spec shut down the exploit.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import exputil as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--rules", default="v1-freewalls")
    ap.add_argument("--out", default="runs/exp02-wallrider")
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--record", type=int, default=3)
    args = ap.parse_args()

    print(f"training fresh PPO: seed {args.seed}, {args.updates} updates, "
          f"rules {args.rules}, opponent L{args.level} ...")
    t0 = time.time()
    net = X.train_short_ppo(seed=args.seed, rules=args.rules,
                            level=args.level, updates=args.updates)
    print(f"trained in {time.time() - t0:.0f}s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = X.export_net(net, out / "fresh_agent.json")

    from agents.duel_eval import run
    run(model=str(model), level=args.level, episodes=args.episodes,
        rules=args.rules, out=str(out), record=args.record, seed0=70_000)
    print(f"\nreplays in {out}/replays; serve the repo root "
          f"(python -m http.server) and open "
          f"viz/watch.html?run={out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
