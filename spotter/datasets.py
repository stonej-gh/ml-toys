# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Shard loading + class-balanced 32x32 patch extraction for M1 training.

Patches are labeled by their CENTER pixel's class. Sampling is per-frame,
per-class (background would otherwise drown the tiny laser squares ~10^4:1),
and fully seeded. Centers too close to the canvas edge are clamped inward and
then RELABELED from the actual post-clamp center, so labels stay honest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import IGNORE
from .render import H, W

PATCH = 32
HALF = PATCH // 2
STRIDE = 8            # trunk downsampling; heatmap cell pitch in input px


def load_split(path: str | Path):
    npz = np.load(path)
    return npz["frames"], npz["masks"]


def heatmap_labels(masks: np.ndarray) -> np.ndarray:
    """Cell-label grid for dense-coarse training. Cell (i,j) OWNS the 8x8
    tile at rows 8i+12..8i+20, cols 8j+12..8j+20 (the center block of its
    32x32 receptive window); its label is the rarest non-background class
    with any pixel in that tile (laser > interceptor > fighter > hole).

    Center-PIXEL labeling was tried first and failed for lasers: a 3-px
    laser square lands on the stride-8 grid of center pixels only ~1 time
    in 7, so most lasers produced no positive cell at all. Tile-presence
    labeling guarantees every entity lights at least one cell. Tiles that
    contain only flame pixels (IGNORE) stay excluded from the loss."""
    h = (H - PATCH) // STRIDE + 1
    w = (W - PATCH) // STRIDE + 1
    tiles = masks[:, HALF - 4:HALF - 4 + h * STRIDE,
                  HALF - 4:HALF - 4 + w * STRIDE]
    tiles = tiles.reshape(len(masks), h, STRIDE, w, STRIDE)
    labels = np.zeros((len(masks), h, w), dtype=np.int64)
    flame = (tiles == IGNORE).any(axis=(2, 4))
    labels[flame] = IGNORE
    for c in (4, 2, 1, 3):                    # rarest applied last -> wins
        labels[(tiles == c).any(axis=(2, 4))] = c
    return labels


def extract_patches(frames, masks, per_class: int = 4, seed: int = 0):
    """-> (X uint8 [N,32,32,3], y int64 [N]), balanced across present classes."""
    rng = np.random.default_rng(seed)
    xs_out, ys_out = [], []
    for i in range(len(frames)):
        m = masks[i]
        for c in np.unique(m):
            if c == IGNORE:
                continue
            ys, xs = np.nonzero(m == c)
            pick = rng.choice(len(xs), size=min(per_class, len(xs)),
                              replace=False)
            for k in pick:
                cx = int(np.clip(xs[k], HALF, W - HALF))
                cy = int(np.clip(ys[k], HALF, H - HALF))
                label = m[cy, cx]
                if label == IGNORE:
                    continue
                xs_out.append(frames[i, cy - HALF:cy + HALF,
                                     cx - HALF:cx + HALF])
                ys_out.append(label)
    return np.stack(xs_out), np.asarray(ys_out, dtype=np.int64)
