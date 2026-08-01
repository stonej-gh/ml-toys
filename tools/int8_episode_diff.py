#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Measure float-vs-int8 argmax disagreement over a whole replay episode.

The golden bundle proves the int8 path on 8 frozen frames; this tool measures
the same disagreement across every frame of an episode, which is where the
notebook's "at most six pixels on the worst frame" class of claim comes from.
Reference numbers (held-out seed_005, measured 2026-08-01): 741 frames,
586 differing pixels of 45,527,040 total (0.79 per frame), worst frame 6,
333 frames identical.

Run from the repo root:  python tools/int8_episode_diff.py [replay.json]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotter.render import load_replay_scenes, render                   # noqa: E402
from deploy.spotter_forward import load_doc, forward_dense, forward_dense_int8  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    replay = sys.argv[1] if len(sys.argv) > 1 else ROOT / "assets/replays/seed_005.json"
    scenes = load_replay_scenes(str(replay))
    docf = load_doc(ROOT / "deploy/models/spotter_dense.json")
    doc8 = load_doc(ROOT / "deploy/models/spotter_dense_int8.json")
    diffs = []
    for i, sc in enumerate(scenes):
        im, _ = render(sc)
        x = (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)
        pf = forward_dense(x, docf).argmax(0).astype(np.uint8)
        _, pq = forward_dense_int8(x, doc8)
        diffs.append(int((pf != pq).sum()))
        if i % 60 == 0:
            print(f"frame {i:4d}: {diffs[-1]} differing px", flush=True)
    d = np.asarray(diffs)
    print(f"\n{replay}")
    print(f"frames {len(d)}  pixels {len(d) * pf.size:,}")
    print(f"differing {d.sum()}  per-frame mean {d.mean():.2f}  "
          f"worst frame {d.max()}  identical frames {(d == 0).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
