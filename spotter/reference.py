# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Pure-numpy reference forward pass over the exported JSON - the float gate.

This file is the seed of the deploy/ bundle's reference implementation: no
torch, no PIL, just numpy on the dense-JSON export. A downstream target
re-implements exactly these ops and checks against the golden vectors.

Three entry points:
  forward_patch(x, doc)   - one 32x32 patch  -> [num_classes] logits
  forward_heatmap(x, doc) - full HxW frame   -> [num_classes, H/8-3, W/8-3]
                            (sliding 4x4 window = the M1 coarse heatmap;
                            cell (i,j) sees the 32x32 input window whose
                            top-left is (8j, 8i))
  forward_dense(x, doc)   - full HxW frame   -> [num_classes, H, W]
                            (M2: trunk -> decoder ops in order -> seg head)
"""

from __future__ import annotations

import json

import numpy as np


def load_doc(path) -> dict:
    doc = json.loads(open(path).read())
    convs = doc["trunk"] + [l for l in doc.get("decoder", []) if "w" in l]
    for layer in convs:
        layer["w"] = np.asarray(layer["w"], dtype=np.float32)
        layer["b"] = np.asarray(layer["b"], dtype=np.float32)
    for key in ("head", "seg"):
        if key in doc:
            doc[key]["w"] = np.asarray(doc[key]["w"], dtype=np.float32)
            doc[key]["b"] = np.asarray(doc[key]["b"], dtype=np.float32)
    return doc


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray,
           stride: int = 1, pad: int = 1) -> np.ndarray:
    """x [C,H,W] float32 -> [O,H',W']. Direct sliding-window conv."""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    k = w.shape[-1]
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))
    win = win[:, ::stride, ::stride]                 # [C, H', W', k, k]
    out = np.einsum("chwij,ocij->ohw", win, w, optimize=True)
    return out + b[:, None, None]


def trunk_features(x: np.ndarray, doc: dict, all_stages: bool = False):
    """[3,H,W] in [0,1] -> ReLU'd trunk output [32, H/8, W/8]; with
    all_stages, the list of every stage's output (skip taps)."""
    feats = []
    for layer in doc["trunk"]:
        x = conv2d(x, layer["w"], layer["b"], stride=layer["stride"],
                   pad=layer["pad"])
        if layer["relu"]:
            np.maximum(x, 0.0, out=x)
        if all_stages:
            feats.append(x)
    return feats if all_stages else x


def forward_patch(x: np.ndarray, doc: dict) -> np.ndarray:
    """One [3,32,32] patch in [0,1] -> [num_classes] logits."""
    feat = trunk_features(x, doc)                    # [32, 4, 4]
    pooled = feat.mean(axis=(1, 2))
    return doc["head"]["w"] @ pooled + doc["head"]["b"]


def forward_heatmap(x: np.ndarray, doc: dict) -> np.ndarray:
    """Full [3,H,W] frame in [0,1] -> [num_classes, h, w] coarse heatmap."""
    feat = trunk_features(x, doc)
    p = doc["head"]["pool"]
    win = np.lib.stride_tricks.sliding_window_view(feat, (p, p), axis=(1, 2))
    pooled = win.mean(axis=(-1, -2))                 # [32, h, w]
    hm = np.einsum("chw,oc->ohw", pooled, doc["head"]["w"], optimize=True)
    return hm + doc["head"]["b"][:, None, None]


def forward_dense(x: np.ndarray, doc: dict) -> np.ndarray:
    """Full [3,H,W] frame in [0,1] -> [num_classes, H, W] per-pixel logits."""
    stages = trunk_features(x, doc, all_stages=True)
    feat = stages[-1]
    for step in doc["decoder"]:
        if "up" in step:
            feat = feat.repeat(step["up"], axis=1).repeat(step["up"], axis=2)
        elif "skip" in step:
            proj = np.einsum("chw,oc->ohw", stages[step["skip"]], step["w"],
                             optimize=True) + step["b"][:, None, None]
            feat = feat + proj
        else:
            feat = conv2d(feat, step["w"], step["b"], stride=step["stride"],
                          pad=step["pad"])
            if step["relu"]:
                np.maximum(feat, 0.0, out=feat)
    seg = np.einsum("chw,oc->ohw", feat, doc["seg"]["w"], optimize=True)
    return seg + doc["seg"]["b"][:, None, None]
