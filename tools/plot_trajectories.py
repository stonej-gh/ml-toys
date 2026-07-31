#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Draw whole-episode trajectory panels from replay JSONs, side by side.

Each panel overlays the ship trails of every episode in one replay source
(a directory of seed_*.json or a single file), so an era's flying style is
visible in a single still: clean orbital arcs read as rings, wall riding as
a scribble hugging the boundary. Output is a deterministic function of the
input JSONs (stars are seeded), so a tracked image can be reproduced.

Usage, from the repo root:
    python tools/plot_trajectories.py docs/img/out.png \
        "fresh agent, free walls=runs/exp02-wallrider/replays" \
        "shipped champion=replays/v6-final/replays"
"""

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PW, PH = 640, 360               # panel size; same 16:9 canvas as the video renderer
MARGIN = 14                     # px inset so the boundary walls are visible
SCALE = (PH - 2 * MARGIN) / 768.0
AGENT, OPP = (57, 135, 229), (217, 89, 38)
BG = (13, 13, 20)
TEXT = (235, 235, 228)

# drawn geometry only: nominal phone-profile field values, as in viz/render_video.py;
# the physics never reads these
HOLE_R, PLAY_W = 68, 16 / 9 * 768

FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",              # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",           # Fedora
    "C:/Windows/Fonts/arial.ttf",                       # Windows
]


def font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_px(x, y):
    return (PW / 2 + x * SCALE, PH / 2 - y * SCALE)


def base_panel():
    im = Image.new("RGB", (PW, PH), BG)
    d = ImageDraw.Draw(im, "RGBA")
    r = random.Random(7)
    for _ in range(80):
        sx, sy, sr = r.random() * PW, r.random() * PH, r.random() * 1.4 + 0.4
        d.rectangle([sx, sy, sx + sr, sy + sr], fill=(255, 255, 255, 110))
    bw, bh = PLAY_W * SCALE, 768 * SCALE
    d.rectangle([PW / 2 - bw / 2, PH / 2 - bh / 2, PW / 2 + bw / 2, PH / 2 + bh / 2],
                outline=(46, 46, 58))
    hx, hy = to_px(0, 0)
    for rr, a in [(2.2, 40), (1.7, 70), (1.25, 110)]:
        d.ellipse([hx - HOLE_R * SCALE * rr, hy - HOLE_R * SCALE * rr,
                   hx + HOLE_R * SCALE * rr, hy + HOLE_R * SCALE * rr],
                  fill=(32, 180, 200, a))
    d.ellipse([hx - HOLE_R * SCALE, hy - HOLE_R * SCALE,
               hx + HOLE_R * SCALE, hy + HOLE_R * SCALE], fill=(0, 0, 0))
    return im


def episodes(source):
    """Replay frame arrays from a file or a directory of seed_*.json."""
    p = Path(source)
    files = (sorted(p.glob("seed_*.json")) + sorted(p.glob("ep*.json"))
             if p.is_dir() else [p])
    out = []
    for f in files:
        d = json.loads(f.read_text())
        if isinstance(d, dict) and "frames" in d:
            out.append(d["frames"])
    return out


def draw_trail(d, frames, offset, color, width):
    """Polyline through one ship's positions while it is alive."""
    pts = [to_px(f[offset], f[offset + 1]) for f in frames if f[offset + 3]]
    if len(pts) >= 2:
        d.line(pts, fill=color, width=width, joint="curve")
    if pts:
        x, y = pts[-1]
        d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)


def panel(label, source):
    im = base_panel()
    d = ImageDraw.Draw(im, "RGBA")
    for frames in episodes(source)[:8]:
        draw_trail(d, frames, 4, OPP + (90,), 1)     # opponent, faint
        draw_trail(d, frames, 0, AGENT + (170,), 2)  # the agent's trail carries the panel
    d.text((12, 10), label, font=font(20), fill=TEXT)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("panels", nargs="+", metavar="LABEL=SOURCE",
                    help="panel spec: label=replay dir or file")
    args = ap.parse_args()

    ims = [panel(*spec.split("=", 1)) for spec in args.panels]
    sheet = Image.new("RGB", (PW * len(ims) + 4 * (len(ims) - 1), PH), BG)
    for i, im in enumerate(ims):
        sheet.paste(im, (i * (PW + 4), 0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out} ({len(ims)} panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
