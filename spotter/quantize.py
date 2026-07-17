# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""PTQ int8 quantization of the dense export - the build side of the bundle.

Scheme (kernel-friendly, no zero-points anywhere):
  - weights: per-OUT-CHANNEL symmetric int8, scale_w[o] = max|W[o]| / 127
  - activations: per-TENSOR symmetric int8, scale = percentile(|act|) / 127
    measured on seeded calibration frames (activations are post-ReLU, so the
    negative half of int8 goes unused - accepted for zero-point-free adds)
  - input: q = round(x * 127) for x in [0,1]  (scale_in = 1/127)
  - conv: int32 accumulate; bias pre-scaled to int32 (b / (s_in * s_w[o]));
    requantize with per-channel multiplier M[o] = s_in*s_w[o]/s_out, clamp
    to [0,127] after ReLU layers, [-127,127] otherwise
  - skip add: both addends requantized to the sum tensor's scale, added,
    clamped - plain int adds, no concat
  - seg head: int32 accumulate, then DEQUANTIZE to float logits
    (logits[o] = (acc + b_q[o]) * s_in * s_w[o])

The int8 JSON stores quantized weights, int32 biases, and the float
multipliers M - a hardware target replaces M with fixed-point multipliers;
the numpy reference (deploy/spotter_forward.py) applies them exactly as
written, so its integer outputs are bit-reproducible.

Run (after training M2):
    PYTHONPATH=src .venv/bin/python -m spotter.quantize \
        --float runs/m2/spotter_dense.json --out runs/m2/spotter_dense_int8.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .reference import conv2d, load_doc

PCT = 100.0        # activation calibration percentile (max: percentile clipping saturated exactly the rare bright laser activations)
QMAX = 127


def _record_dense(x: np.ndarray, doc: dict, rec: dict):
    """Float dense forward that appends |activation| samples per tensor."""
    def note(name, t):
        rec.setdefault(name, []).append(np.abs(t.ravel()[::7]))

    stages = []
    for i, layer in enumerate(doc["trunk"]):
        x = conv2d(x, layer["w"], layer["b"], stride=layer["stride"],
                   pad=layer["pad"])
        np.maximum(x, 0.0, out=x)
        note(f"t{i}", x)
        stages.append(x)
    feat = stages[-1]
    di = 0
    for step in doc["decoder"]:
        if "up" in step:
            feat = feat.repeat(2, axis=1).repeat(2, axis=2)
        elif "skip" in step:
            proj = np.einsum("chw,oc->ohw", stages[step["skip"]], step["w"],
                             optimize=True) + step["b"][:, None, None]
            feat = feat + proj
            di += 1
            note(f"s{di}", feat)
        else:
            feat = conv2d(feat, step["w"], step["b"], stride=step["stride"],
                          pad=step["pad"])
            np.maximum(feat, 0.0, out=feat)
            note(f"d{di}", feat)
    return feat


def calibrate(doc: dict, frames: np.ndarray, pct: float = PCT) -> dict:
    """frames uint8 NHWC -> per-tensor activation scales."""
    rec = {}
    for f in frames:
        x = (f.astype(np.float32) / 255.0).transpose(2, 0, 1)
        _record_dense(x, doc, rec)
    return {name: float(np.percentile(np.concatenate(v), pct)) / QMAX
            for name, v in rec.items()}


def _qw(w: np.ndarray):
    """Per-out-channel symmetric weight quantization -> (int8 w, scales)."""
    flat = np.abs(w.reshape(len(w), -1)).max(axis=1)
    sw = np.maximum(flat, 1e-12) / QMAX
    wq = np.clip(np.round(w / sw.reshape(-1, *([1] * (w.ndim - 1)))),
                 -QMAX, QMAX).astype(np.int8)
    return wq, sw


def _qconv(w, b, s_in: float, s_out: float | None):
    """-> dict fields for one quantized conv (s_out None = dequant head)."""
    wq, sw = _qw(np.asarray(w))
    bq = np.round(np.asarray(b) / (s_in * sw)).astype(np.int64)
    m = s_in * sw if s_out is None else s_in * sw / s_out
    return {"wq": wq.tolist(), "bq": bq.tolist(), "m": m.tolist()}


def quantize(doc: dict, scales: dict) -> dict:
    s_in = 1.0 / QMAX
    trunk = []
    s_prev = s_in
    for i, layer in enumerate(doc["trunk"]):
        s_out = scales[f"t{i}"]
        q = _qconv(layer["w"], layer["b"], s_prev, s_out)
        q.update(stride=layer["stride"], pad=layer["pad"], relu=True,
                 scale_out=s_out)
        trunk.append(q)
        s_prev = s_out

    dec = []
    s_carry = scales["t5"]
    di = 0
    for step in doc["decoder"]:
        if "up" in step:
            dec.append({"up": 2})
        elif "skip" in step:
            di += 1
            s_sum = scales[f"s{di}"]
            q = _qconv(step["w"], step["b"], scales[f"t{step['skip']}"], s_sum)
            q.update(skip=step["skip"], m_carry=s_carry / s_sum,
                     scale_out=s_sum)
            dec.append(q)
            s_carry = s_sum
        else:
            s_out = scales[f"d{di}"]
            q = _qconv(step["w"], step["b"], s_carry, s_out)
            q.update(stride=step["stride"], pad=step["pad"], relu=True,
                     scale_out=s_out)
            dec.append(q)
            s_carry = s_out

    seg = _qconv(doc["seg"]["w"], doc["seg"]["b"], s_carry, None)
    return {"model": doc["model"] + "_int8", "classes": doc["classes"],
            "input": "q = round(rgb/255 * 127) CHW int8; scales per tensor",
            "calibration_pct": PCT,
            "trunk": trunk, "decoder": dec, "seg": seg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float", dest="fdoc",
                    default="runs/m2/spotter_dense.json")
    ap.add_argument("--out", default="runs/m2/spotter_dense_int8.json")
    ap.add_argument("--calib-frames", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()

    doc = load_doc(args.fdoc)
    data = np.load(Path("data/derived/train.npz"))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(data["frames"]), size=args.calib_frames,
                     replace=False)
    scales = calibrate(doc, data["frames"][idx])
    print("activation scales:", {k: round(v, 5) for k, v in scales.items()})
    doc8 = quantize(doc, scales)
    Path(args.out).write_text(json.dumps(doc8))
    kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
