# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Renderer + sampler invariants: determinism, frame/mask alignment, coverage."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotter import CLASSES, IGNORE                                 # noqa: E402
from spotter.render import (H, HOLE_R, SCALE, W, Scene, ShipState,  # noqa: E402
                            load_replay_scenes, render, to_px)
from spotter.sample import sample_scene                             # noqa: E402

REPLAY = Path(__file__).resolve().parents[1] / "assets/replays/seed_000.json"


def test_render_deterministic():
    sc = sample_scene(123)
    f1, m1 = render(sc)
    f2, m2 = render(sc)
    assert f1.tobytes() == f2.tobytes()
    assert m1.tobytes() == m2.tobytes()


def test_mask_class_ids_and_presence():
    sc = Scene(ships=[ShipState(-300, 100, 1.0, True, True),
                      ShipState(300, -100, 4.0, True, False)],
               lasers=[(150.0, 50.0), (-150.0, -50.0)])
    _, mk = render(sc)
    m = np.asarray(mk)
    present = set(np.unique(m).tolist())

    # all five classes on screen, flame ignore pixels from the thrusting ship
    assert present == {0, 1, 2, 3, 4, IGNORE}
    assert len(CLASSES) == 5


def test_frame_mask_alignment():
    # a lone ship: its mask pixels must center on its world position
    x, y = 250.0, -120.0
    sc = Scene(ships=[ShipState(x, y, 2.2, True, False),
                      ShipState(0, 0, 0, False, False)], lasers=[])
    _, mk = render(sc)
    ys, xs = np.nonzero(np.asarray(mk) == 1)
    assert len(xs) > 40  # hull is well over 40 px
    px, py = to_px(x, y)
    assert abs(xs.mean() - px) < 3.0
    assert abs(ys.mean() - py) < 3.0


def test_hole_mask_geometry():
    sc = Scene(ships=[ShipState(alive=False), ShipState(alive=False)])
    _, mk = render(sc)
    ys, xs = np.nonzero(np.asarray(mk) == 4)
    cx, cy = to_px(0, 0)
    assert abs(xs.mean() - cx) < 1.0 and abs(ys.mean() - cy) < 1.0
    rad = np.hypot(xs - xs.mean(), ys - ys.mean()).max()
    assert abs(rad - HOLE_R * SCALE) < 2.0


def test_letterbox_rows_are_background():
    sc = sample_scene(7)
    _, mk = render(sc)
    m = np.asarray(mk)
    assert (m[:5] == 0).all() and (m[-5:] == 0).all()


def test_sampler_seeded_and_in_bounds():
    a, b = sample_scene(99), sample_scene(99)
    assert a == b
    for k in range(50):
        sc = sample_scene(1000 + k)
        for ship in sc.ships:
            assert abs(ship.x) < W / SCALE and abs(ship.y) < 768 / 2
            assert math.hypot(ship.x, ship.y) > HOLE_R  # outside the capture disk


def test_replay_loads_and_renders():
    scenes = load_replay_scenes(REPLAY)
    assert len(scenes) > 100
    frame, mk = render(scenes[len(scenes) // 2])
    assert frame.size == (W, H)
    m = np.asarray(mk)
    assert {1, 2, 4} <= set(np.unique(m).tolist())  # both ships + hole visible
