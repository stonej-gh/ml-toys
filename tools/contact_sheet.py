#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Render a contact sheet for eyeball QA: frame | frame+mask-overlay pairs.

Half the rows are seeded random scenes, half are replay frames, so one image
shows both training distributions. Output goes to data/derived/review/
(untracked); this replaces any formal label-review gate - the masks are exact
by construction, the sheet just lets a human confirm the renderer looks right.

Usage: python tools/contact_sheet.py [--rows 6] [--seed0 42]
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotter.render import load_replay_scenes, overlay, render  # noqa: E402
from spotter.sample import sample_scene                         # noqa: E402

PAD = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    scenes = [sample_scene(args.seed0 + k) for k in range(args.rows)]
    rep = load_replay_scenes(root / "assets/replays/seed_000.json")
    step = max(1, len(rep) // args.rows)
    scenes += rep[::step][:args.rows]

    tiles = []
    for sc in scenes:
        im, mk = render(sc)
        tiles.append((im, overlay(im, mk)))

    tw, th = tiles[0][0].size
    sheet = Image.new("RGB", (2 * tw + 3 * PAD,
                              len(tiles) * (th + PAD) + PAD), (30, 30, 36))
    for r, (im, ov) in enumerate(tiles):
        y = PAD + r * (th + PAD)
        sheet.paste(im, (PAD, y))
        sheet.paste(ov, (2 * PAD + tw, y))

    out = root / "data/derived/review"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contact_sheet.png"
    sheet.save(path)
    print("wrote", path, f"({len(tiles)} rows: "
          f"{len(tiles) - len(tiles) // 2} random + {len(tiles) // 2} replay)")


if __name__ == "__main__":
    main()
