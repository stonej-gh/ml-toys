#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Experiment 06 orchestrator: the FULL stranger flow, end to end.

    python experiments/06-spotter-port/run.py [--train-n 2000]

Runs, in order: gen_dataset -> train_heatmap (M1) -> train_dense (M2) ->
export --mode dense -> quantize -> build_bundle -> deploy/verify.py. Stops at
the first failing stage. This rewrites deploy/models and deploy/golden with
your retrain's bundle; `git restore deploy` brings the shipped one back.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-n", type=int, default=2000)
    args = ap.parse_args()

    py = sys.executable
    stages = [
        ("dataset", [py, "tools/gen_dataset.py", "--train-n", str(args.train_n)]),
        ("M1 heatmap", [py, "-m", "spotter.train_heatmap"]),
        ("M2 dense", [py, "-m", "spotter.train_dense"]),
        ("export", [py, "-m", "spotter.export", "--mode", "dense"]),
        ("quantize", [py, "-m", "spotter.quantize"]),
        ("bundle", [py, "tools/build_bundle.py"]),
        ("verify", [py, "deploy/verify.py"]),
    ]
    for name, cmd in stages:
        print(f"\n=== {name}: {' '.join(cmd)}")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=REPO)
        print(f"=== {name} {'ok' if r.returncode == 0 else 'FAILED'} "
              f"({time.time() - t0:.0f}s)")
        if r.returncode != 0:
            return r.returncode
    print("\nstranger flow complete: your bundle verifies. "
          "`git restore deploy` brings the shipped bundle back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
