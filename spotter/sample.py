# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Seeded random-scene sampler - the iid half of the training distribution.

Random scenes give uniform coverage of poses/positions the replays under-visit
(replays orbit; random scenes also put ships in corners, nose-down dives, dead
ships, laser volleys). Training mixes both: sample_scene() for coverage,
render.load_replay_scenes() for the correlations a real duel produces.

Everything is driven by one integer seed -> numpy Generator, so any dataset
shard is reproducible from its manifest.
"""

from __future__ import annotations

import math

import numpy as np

from .render import (FIELD_H, HOLE_R, PLAY_W, SHIP_R, Scene, ShipState)

# keep spawns off the walls and outside the capture radius, like live play
_EDGE = SHIP_R + 4.0
_HOLE_KEEPOUT = HOLE_R + SHIP_R + 10.0
_MAX_LASERS = 8


def _position(rng: np.random.Generator) -> tuple[float, float]:
    """Uniform over the field, rejecting the hole keep-out disk."""
    half_w, half_h = PLAY_W / 2.0 - _EDGE, FIELD_H / 2.0 - _EDGE
    while True:
        x = rng.uniform(-half_w, half_w)
        y = rng.uniform(-half_h, half_h)
        if math.hypot(x, y) > _HOLE_KEEPOUT:
            return x, y


def sample_scene(seed: int) -> Scene:
    rng = np.random.default_rng(seed)
    ships = []
    for _ in range(2):
        x, y = _position(rng)
        ships.append(ShipState(
            x=x, y=y,
            heading=rng.uniform(0.0, 2.0 * math.pi),
            alive=rng.random() < 0.95,
            thrusting=rng.random() < 0.3))
    lasers = []
    for _ in range(min(rng.poisson(1.5), _MAX_LASERS)):
        lx, ly = _position(rng)
        lasers.append((lx, ly))
    # star_seed varies the background clutter so the net can't memorize it
    return Scene(ships=ships, lasers=lasers, star_seed=int(rng.integers(1 << 31)))
