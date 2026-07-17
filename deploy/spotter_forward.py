#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Reference forward passes for the spotter golden bundle - numpy only.

This file is the executable specification. A downstream target re-implements
exactly these ops on its own hardware and checks against golden/ (see
verify.py). It is self-contained: no imports beyond numpy + stdlib, no
dependency on the training repo, relocatable anywhere with the bundle.

  forward_dense(x, doc)        float model:  x [3,H,W] in [0,1] -> logits
  forward_dense_int8(x, doc8)  int8 model:   same x -> (float logits, argmax)
                               integer path is deterministic bit-for-bit:
                               int8 weights, int32 accumulate, float
                               requantize multipliers applied per layer

Load either JSON with load_doc(path).
"""

from __future__ import annotations

import json

import numpy as np

QMAX = 127


def load_doc(path) -> dict:
    doc = json.loads(open(path).read())
    isq = doc["model"].endswith("_int8")
    for layer in doc["trunk"] + doc["decoder"] + [doc.get("seg") or {}]:
        for key, dt in (("w", np.float32), ("b", np.float32),
                        ("wq", np.int8), ("bq", np.int64), ("m", np.float64)):
            if key in layer:
                layer[key] = np.asarray(layer[key], dtype=dt)
    if "head" in doc:
        doc["head"]["w"] = np.asarray(doc["head"]["w"], dtype=np.float32)
        doc["head"]["b"] = np.asarray(doc["head"]["b"], dtype=np.float32)
    doc["_int8"] = isq
    return doc


def _windows(x: np.ndarray, k: int, stride: int, pad: int):
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))
    return win[:, ::stride, ::stride]


def conv2d(x, w, b, stride=1, pad=1):
    """Float conv: x [C,H,W] float32 -> [O,H',W']."""
    win = _windows(x, w.shape[-1], stride, pad)
    return np.einsum("chwij,ocij->ohw", win, w, optimize=True) \
        + b[:, None, None]


def conv2d_int(q, wq, stride=1, pad=1):
    """Integer conv: q [C,H,W] int8 -> int32 accumulator [O,H',W']."""
    win = _windows(q.astype(np.int32), wq.shape[-1], stride, pad)
    return np.einsum("chwij,ocij->ohw", win, wq.astype(np.int32),
                     optimize=True)


def _requant(acc, bq, m, relu):
    q = np.round((acc + bq[:, None, None]) * m[:, None, None])
    lo = 0 if relu else -QMAX
    return np.clip(q, lo, QMAX).astype(np.int8)


# --- float model -----------------------------------------------------------------
def _trunk(x, doc, all_stages=False):
    feats = []
    for layer in doc["trunk"]:
        x = conv2d(x, layer["w"], layer["b"], stride=layer["stride"],
                   pad=layer["pad"])
        np.maximum(x, 0.0, out=x)
        if all_stages:
            feats.append(x)
    return feats if all_stages else x


def forward_dense(x: np.ndarray, doc: dict) -> np.ndarray:
    """[3,H,W] float in [0,1] -> [num_classes,H,W] float logits."""
    stages = _trunk(x, doc, all_stages=True)
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
            np.maximum(feat, 0.0, out=feat)
    seg = np.einsum("chw,oc->ohw", feat, doc["seg"]["w"], optimize=True)
    return seg + doc["seg"]["b"][:, None, None]


# --- int8 model --------------------------------------------------------------------
def forward_dense_int8(x: np.ndarray, doc8: dict):
    """[3,H,W] float in [0,1] -> (float logits [num_classes,H,W], argmax).

    Everything between input quantization and the final dequantize is
    integer (int8 tensors, int32 accumulators) with per-layer float
    requantize multipliers - deterministic bit-for-bit on any platform.
    """
    q = np.clip(np.round(x * QMAX), 0, QMAX).astype(np.int8)
    stages = []
    for layer in doc8["trunk"]:
        acc = conv2d_int(q, layer["wq"], stride=layer["stride"],
                         pad=layer["pad"])
        q = _requant(acc, layer["bq"], layer["m"], relu=True)
        stages.append(q)
    q = stages[-1]
    for step in doc8["decoder"]:
        if "up" in step:
            q = q.repeat(step["up"], axis=1).repeat(step["up"], axis=2)
        elif "skip" in step:
            src = stages[step["skip"]].astype(np.int32)
            acc = np.einsum("chw,oc->ohw", src,
                            step["wq"].astype(np.int32), optimize=True)
            proj = np.round((acc + step["bq"][:, None, None])
                            * step["m"][:, None, None])
            carry = np.round(q.astype(np.float64) * step["m_carry"])
            q = np.clip(carry + proj, -QMAX, QMAX).astype(np.int8)
        else:
            acc = conv2d_int(q, step["wq"], stride=step["stride"],
                             pad=step["pad"])
            q = _requant(acc, step["bq"], step["m"], relu=True)
    seg = doc8["seg"]
    acc = np.einsum("chw,oc->ohw", q.astype(np.int32),
                    seg["wq"].astype(np.int32), optimize=True)
    logits = (acc + seg["bq"][:, None, None]) * seg["m"][:, None, None]
    return logits, logits.argmax(0).astype(np.uint8)
