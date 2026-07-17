# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Seeded vector renderer for orbital-duel scenes: frame AND mask.

This is the whole labeling story: every entity is drawn twice from the same
geometry - once shaded into the RGB frame, once flat into the class-id mask -
so the "labels" are exact by construction. No hand labels, no label QA gate.

World model (constants match the game's headless physics twin at gameScale 1):
  - field H=768 pt tall, PLAY_W=16/9*768 wide, origin at the field center,
    y UP, heading in radians with 0 = nose along +y (zRotation convention)
  - central hole (event-horizon radius HOLE_R) that captures on contact
  - two ships with distinct hulls (interceptor: narrow dart; fighter: broad
    blunt wedge - the silhouette difference is what lets the net separate the
    ship classes without leaning on color alone)
  - lasers as small bright squares

Canvas is 320x192 (both /8 for the trunk stride; the 16:9 field maps to
320x180 with 6-px letterbox bars top and bottom, labeled background).
SCALE = 320/PLAY_W = 15/64 exactly, so px math is reproducible across
platforms.

Stylizations vs raw physics, all deliberate and documented:
  - lasers draw as 3x3-px squares (~2x their 4-pt physics radius) - at this
    canvas scale a physics-true laser would be sub-pixel and unlearnable
  - thrust flames draw in the frame but rate IGNORE in the mask (transient
    exhaust, not an object)
  - the hole's soft glow is frame-only decoration; the mask holes out only
    the hard event-horizon disk
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from . import IGNORE

# --- world constants (physics twin, gameScale 1) --------------------------------
FIELD_H = 768.0
PLAY_W = 16.0 / 9.0 * FIELD_H          # 1365.33
HOLE_R = 68.0                          # event-horizon radius (pt)
SHIP_R = 27.0                          # ship physics circle (pt)
LASER_R = 4.0                          # physics radius (pt); drawn stylized

# --- canvas ----------------------------------------------------------------------
W, H = 320, 192                        # model input size; both divisible by 8
SCALE = W / PLAY_W                     # = 15/64 = 0.234375 exactly
FIELD_H_PX = FIELD_H * SCALE           # 180; (H - 180)/2 = 6-px letterbox bars

# --- frame colors (game look) ------------------------------------------------
BG = (13, 13, 20)
SHIP_COLORS = [(57, 135, 229), (217, 89, 38)]   # ship0 blue, ship1 orange
FLAME = (255, 210, 122)
LASER_COLOR = (255, 255, 255)
BORDER = (46, 46, 58)

# Hull outlines in design points, nose at +y, one loop around the silhouette
# (ported from the game's ShipDesign). Width 32/44 design pt; HULL_SCALE maps
# design pt -> world pt so the interceptor spans its 54-pt physics circle.
HULL_SCALE = 1.68
HULLS = [
    # ship 0 - interceptor: sleek narrow dart with swept wingtips
    [(0, 26), (5, 6), (7, -7), (16, -16), (8, -11), (5, -19),
     (-5, -19), (-8, -11), (-16, -16), (-7, -7), (-5, 6)],
    # ship 1 - fighter: broader, blunter nose, wider swept wings
    [(0, 22), (7, 11), (11, 1), (22, -7), (13, -9), (9, -17),
     (3, -13), (-3, -13), (-9, -17), (-13, -9), (-22, -7), (-11, 1), (-7, 11)],
]
FLAME_PTS = [(-5, -18), (0, -34), (5, -18)]      # tail exhaust triangle

LASER_PX = 1.5                          # half-size of the drawn laser square


@dataclass
class ShipState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0      # rad, 0 = nose +y
    alive: bool = True
    thrusting: bool = False


@dataclass
class Scene:
    """Everything the renderer needs for one frame. Class ids: ship i -> i+1."""
    ships: list = field(default_factory=lambda: [ShipState(), ShipState()])
    lasers: list = field(default_factory=list)   # [(x, y), ...] world pt
    star_seed: int = 7                           # background clutter variant


def to_px(x: float, y: float) -> tuple[float, float]:
    """World (y up, origin center) -> canvas px (y down)."""
    return W / 2.0 + x * SCALE, H / 2.0 - y * SCALE


def _hull_px(pts, x, y, heading, scale=HULL_SCALE):
    """Design points (nose +y) -> rotated canvas polygon."""
    px, py = to_px(x, y)
    c, s = math.cos(heading), math.sin(heading)
    out = []
    for dx, dy in pts:
        wx = (dx * c - dy * s) * scale        # CCW rotation in world coords
        wy = (dx * s + dy * c) * scale
        out.append((px + wx * SCALE, py - wy * SCALE))
    return out


def _stars(seed: int):
    r = random.Random(seed)
    return [(r.random() * W, r.random() * H) for _ in range(60)]


def render(scene: Scene) -> tuple[Image.Image, Image.Image]:
    """One scene -> (RGB frame, 'L' class-id mask), same geometry for both."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    mk = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mk)

    # background clutter: stars + field border, all class 0
    for sx, sy in _stars(scene.star_seed):
        d.point((sx, sy), fill=(255, 255, 255, 120))
    bw = PLAY_W * SCALE
    d.rectangle([W / 2 - bw / 2, H / 2 - FIELD_H_PX / 2,
                 W / 2 + bw / 2 - 1, H / 2 + FIELD_H_PX / 2 - 1], outline=BORDER)

    # hole: soft glow (frame only) + hard core (frame black, mask class 4)
    hx, hy = to_px(0.0, 0.0)
    for rr, a in [(2.2, 40), (1.7, 70), (1.25, 110)]:
        g = HOLE_R * SCALE * rr
        d.ellipse([hx - g, hy - g, hx + g, hy + g], fill=(32, 180, 200, a))
    hr = HOLE_R * SCALE
    core = [hx - hr, hy - hr, hx + hr, hy + hr]
    d.ellipse(core, fill=(0, 0, 0))
    md.ellipse(core, fill=4)

    # ships: hull (frame color i, mask class i+1); flame frame-only + IGNORE
    for i, ship in enumerate(scene.ships):
        if not ship.alive:
            continue
        if ship.thrusting:
            fl = _hull_px(FLAME_PTS, ship.x, ship.y, ship.heading)
            d.polygon(fl, fill=FLAME)
            md.polygon(fl, fill=IGNORE)
        hull = _hull_px(HULLS[i], ship.x, ship.y, ship.heading)
        d.polygon(hull, fill=SHIP_COLORS[i], outline=(255, 255, 255, 80))
        md.polygon(hull, fill=i + 1)

    # lasers: stylized bright squares, mask class 3
    for lx, ly in scene.lasers:
        px, py = to_px(lx, ly)
        box = [px - LASER_PX, py - LASER_PX, px + LASER_PX, py + LASER_PX]
        d.rectangle(box, fill=LASER_COLOR)
        md.rectangle(box, fill=3)

    return im, mk


def overlay(frame: Image.Image, mask: Image.Image, alpha: int = 110) -> Image.Image:
    """Frame with the palette-colored mask blended on top (review + demo)."""
    from . import PALETTE
    m = np.asarray(mask)
    rgba = np.zeros((*m.shape, 4), dtype=np.uint8)
    for cid, col in PALETTE.items():
        if cid in (0, IGNORE):
            continue
        rgba[m == cid] = (*col, alpha)
    out = frame.convert("RGBA")
    out.alpha_composite(Image.fromarray(rgba))
    return out.convert("RGB")


# --- replay traces ----------------------------------------------------------------
# Recorded league self-play episodes (assets/replays/*.json). Frame schema is a
# flat list per (every-3rd-physics) frame:
#   [ax, ay, ahead, aalive, bx, by, bhead, balive, athrust, bthrust, [[lx,ly],..]]
def load_replay_scenes(path) -> list[Scene]:
    rep = json.loads(open(path).read())
    scenes = []
    for f in rep["frames"]:
        scenes.append(Scene(
            ships=[ShipState(f[0], f[1], f[2], bool(f[3]), bool(f[8])),
                   ShipState(f[4], f[5], f[6], bool(f[7]), bool(f[9]))],
            lasers=[tuple(l) for l in f[10]]))
    return scenes
