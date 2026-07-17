#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Render archived training replays to video: the experiment's permanent record.

Reads the same runs/duel/replays/ archive the live viewer uses, draws each
episode with PIL at 20 fps (replay frames are every 3 physics frames), adds a
title card per episode (update, opponent level, outcome), and hands the frame
stream to ffmpeg on stdin. Re-runnable any time; the archive is the source of
truth, videos are derived artifacts.

Usage (from ml/):
    .venv/bin/python viz/render_video.py --all -o duel-documentary.mp4
    .venv/bin/python viz/render_video.py --highlights -o duel-highlights.mp4
    .venv/bin/python viz/render_video.py --files upd_00120_ep0.json ... -o clip.mp4
    add --speed 2 to double playback speed

--highlights picks the story beats automatically: the first recorded episode,
the first win, the last eval before each level advance (the "graduation"
match, a win where possible), and the newest win.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
SCALE = H / 768.0
AGENT, OPP = (57, 135, 229), (217, 89, 38)      # blue learner / orange bot
BG = (13, 13, 20)
TEXT = (235, 235, 228)
MUTED = (150, 149, 140)
HOLE_R, SHIP_R, PLAY_W = 68, 27, 16 / 9 * 768
FPS = 20


FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",                       # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",           # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",                    # Fedora
    "C:/Windows/Fonts/arial.ttf",                                # Windows
]


def font(size):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_BIG, F_MED, F_SMALL = font(52), font(30), font(22)


def to_px(x, y):
    return (W / 2 + x * SCALE, H / 2 - y * SCALE)


def draw_ship(d, x, y, hd, color, thrusting):
    px, py = to_px(x, y)
    s = SCALE * 1.4
    pts = [(0, -20), (13, 16), (0, 9), (-13, 16)]
    cos, sin = math.cos(hd), math.sin(hd)   # canvas y-down: rotate by -hd
    rot = [(px + (X * cos + Y * sin) * s, py + (-X * sin + Y * cos) * s)
           for X, Y in pts]
    if thrusting:
        fx = [(-6, 14), (0, 34), (6, 14)]
        d.polygon([(px + (X * cos + Y * sin) * s, py + (-X * sin + Y * cos) * s)
                   for X, Y in fx], fill=(255, 210, 122))
    d.polygon(rot, fill=color, outline=(255, 255, 255, 80))


def stars():
    import random
    r = random.Random(7)
    return [(r.random() * W, r.random() * H, r.random() * 1.6 + 0.4)
            for _ in range(130)]


STARS = stars()


def base_frame():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    for sx, sy, sr in STARS:
        d.rectangle([sx, sy, sx + sr, sy + sr], fill=(255, 255, 255, 120))
    bw, bh = PLAY_W * SCALE, 768 * SCALE
    d.rectangle([W / 2 - bw / 2, H / 2 - bh / 2, W / 2 + bw / 2, H / 2 + bh / 2],
                outline=(46, 46, 58))
    hx, hy = to_px(0, 0)
    for rr, a in [(2.2, 40), (1.7, 70), (1.25, 110)]:
        d.ellipse([hx - HOLE_R * SCALE * rr, hy - HOLE_R * SCALE * rr,
                   hx + HOLE_R * SCALE * rr, hy + HOLE_R * SCALE * rr],
                  fill=(32, 180, 200, a))
    d.ellipse([hx - HOLE_R * SCALE, hy - HOLE_R * SCALE,
               hx + HOLE_R * SCALE, hy + HOLE_R * SCALE], fill=(0, 0, 0))
    return im


BASE = base_frame()


def title_card(meta, seconds=2.0):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    lines = [(f"Update {meta['update']}", F_BIG, TEXT, -60),
             (f"vs scripted pilot, Level {meta['level']}", F_MED, MUTED, 12),
             (meta["outcome"].upper(), F_MED,
              AGENT if meta["outcome"] == "win" else OPP, 64)]
    for txt, f, col, dy in lines:
        w = d.textlength(txt, font=f)
        d.text((W / 2 - w / 2, H / 2 + dy - 26), txt, font=f, fill=col)
    return [im] * int(seconds * FPS)


def render_episode(rep):
    meta, frames, events = rep["meta"], rep["frames"], rep["events"]
    ev_by_frame = {}
    for ev in events:
        ev_by_frame.setdefault(ev["f"], []).append(ev)
    out = []
    flash = []                                   # (frames_left, ship, kind)
    for i, f in enumerate(frames):
        im = BASE.copy()
        d = ImageDraw.Draw(im, "RGBA")
        # trails
        for i0, col in [(0, AGENT), (4, OPP)]:
            seg = [to_px(frames[k][i0], frames[k][i0 + 1])
                   for k in range(max(0, i - 90), i + 1, 2)]
            if len(seg) > 1:
                d.line(seg, fill=col + (70,), width=2)
        if f[3]:
            draw_ship(d, f[0], f[1], f[2], AGENT, f[8])
        if f[7]:
            draw_ship(d, f[4], f[5], f[6], OPP, f[9])
        for lx, ly in f[10]:
            px, py = to_px(lx, ly)
            d.rectangle([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255))
        for ev in ev_by_frame.get(i, []):
            flash.append([14, ev["ship"], ev["ev"]])
        for fl in flash:
            if fl[0] <= 0:
                continue
            sx, sy = (f[0], f[1]) if fl[1] == 0 else (f[4], f[5])
            px, py = to_px(sx, sy)
            rad = (12 + (14 - fl[0]) * 3.2 if fl[2] == "death"
                   else 8 + (14 - fl[0])) * SCALE * 1.4
            col = (255, 141, 92, int(255 * fl[0] / 14)) if fl[2] == "death" \
                else (255, 255, 255, int(255 * fl[0] / 14))
            d.ellipse([px - rad, py - rad, px + rad, py + rad], outline=col, width=3)
            fl[0] -= 1
        d.text((16, 12), f"update {meta['update']}   vs L{meta['level']}   "
               f"{meta['outcome'].upper()}", font=F_SMALL, fill=MUTED)
        d.text((W - 90, 12), f"{i * 3 / 60:5.1f} s", font=F_SMALL, fill=MUTED)
        out.append(im)
    out.extend([out[-1]] * FPS)                  # 1 s hold on the final frame
    return out


def pick_highlights(man):
    picks, seen = [], set()

    def add(entry, why):
        if entry and entry["file"] not in seen:
            seen.add(entry["file"])
            picks.append((entry, why))

    add(man[0], "first recorded episode")
    add(next((e for e in man if e["outcome"] == "win"), None), "first win")
    for i in range(1, len(man)):
        if man[i]["level"] != man[i - 1]["level"]:          # graduation eval
            grads = [e for e in man if e["update"] == man[i - 1]["update"]]
            best = next((e for e in grads if e["outcome"] == "win"), grads[0])
            add(best, f"graduates L{man[i - 1]['level']}")
    add(next((e for e in reversed(man) if e["outcome"] == "win"), man[-1]),
        "latest form")
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="duel.mp4")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--highlights", action="store_true")
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--run-dir", default="runs/duel")
    args = ap.parse_args()

    rep_dir = Path(args.run_dir) / "replays"
    man = json.loads((rep_dir / "manifest.json").read_text())
    if args.files:
        chosen = [(e, "") for e in man if e["file"] in set(args.files)]
    elif args.highlights:
        chosen = pick_highlights(man)
    else:
        chosen = [(e, "") for e in man]
    if not chosen:
        print("nothing to render")
        return 1
    print(f"rendering {len(chosen)} episode(s) -> {args.out}")

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(int(FPS * args.speed)),
         "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         args.out], stdin=subprocess.PIPE)
    for entry, why in chosen:
        rep = json.loads((rep_dir / entry["file"]).read_text())
        if why:
            rep["meta"] = {**rep["meta"], "outcome": rep["meta"]["outcome"]}
        cards = title_card({**rep["meta"]})
        if why:
            d = ImageDraw.Draw(cards[0])
            w = d.textlength(why, font=F_SMALL)
            for im in cards:
                ImageDraw.Draw(im).text((W / 2 - w / 2, H / 2 + 120), why,
                                        font=F_SMALL, fill=MUTED)
        for im in cards + render_episode(rep):
            ff.stdin.write(im.tobytes())
        print(f"  {entry['file']}  {why}")
    ff.stdin.close()
    ff.wait()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
