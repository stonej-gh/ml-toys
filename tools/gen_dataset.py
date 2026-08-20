#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Generate the training shards: rendered frames + exact class masks.

Each split is one compressed .npz (frames uint8 NHWC, masks uint8 NHW) plus a
manifest recording every seed, so any shard is bit-reproducible. Train mixes
seeded random scenes (coverage) with recorded replay frames (realistic duel
correlations); val/test are disjoint seed ranges; test additionally holds out
one entire replay episode the train split never sees.

Usage (from the repo root, with the virtual environment activated):
    python tools/gen_dataset.py                  # default sizes -> data/derived
    python tools/gen_dataset.py --train-n 4000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotter.render import Scene, load_replay_scenes, render  # noqa: E402
from spotter.sample import sample_scene                       # noqa: E402

# Disjoint seed blocks per split - the reproducibility contract.
SEED_BASE = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000}
REPLAYS = {
    "train": ["seed_000.json", "seed_002.json"],
    "test": ["seed_005.json"],                                # held-out episode, never trained on
}


def build_split(name, n_random, replay_frac, assets, rng):
    scenes, sources = [], []
    for k in range(n_random):
        seed = SEED_BASE[name] + k
        scenes.append(sample_scene(seed))
        sources.append(f"random:{seed}")
    pool = []
    for fname in REPLAYS.get(name, []):
        for j, sc in enumerate(load_replay_scenes(assets / fname)):
            pool.append((sc, f"replay:{fname}:{j}"))
    n_replay = int(n_random * replay_frac / max(1e-9, 1 - replay_frac))
    if pool and n_replay:
        idx = rng.choice(len(pool), size=min(n_replay, len(pool)), replace=False)
        for i in sorted(idx):
            sc, src = pool[i]

            # replays carry no star_seed; vary it so clutter isn't episode-constant
            sc.star_seed = int(rng.integers(1 << 31))
            scenes.append(sc)
            sources.append(src)
    frames, masks = [], []
    for sc in scenes:
        im, mk = render(sc)
        frames.append(np.asarray(im))
        masks.append(np.asarray(mk))
    return np.stack(frames), np.stack(masks), sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/derived")
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=400)
    ap.add_argument("--test-n", type=int, default=400)
    ap.add_argument("--replay-frac", type=float, default=0.3,
                    help="fraction of train/test frames drawn from replays")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    assets = root / "assets" / "replays"
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"canvas": [320, 192], "splits": {}}
    for name, n in [("train", args.train_n), ("val", args.val_n),
                    ("test", args.test_n)]:
        # replay pick order is itself seeded -> whole shard reproducible
        rng = np.random.default_rng(SEED_BASE[name] - 1)
        frac = args.replay_frac if name in REPLAYS else 0.0
        frames, masks, sources = build_split(name, n, frac, assets, rng)
        np.savez_compressed(out / f"{name}.npz", frames=frames, masks=masks)
        manifest["splits"][name] = {
            "file": f"{name}.npz", "count": len(sources),
            "seed_base": SEED_BASE[name], "sources": sources}
        print(f"{name}: {len(sources)} frames "
              f"({sum(s.startswith('replay') for s in sources)} replay)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", out / "manifest.json")


if __name__ == "__main__":
    main()
