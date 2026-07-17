# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""BN-fold + dense-JSON export - the bridge from PyTorch to the golden bundle.

The exported JSON has NO BatchNorm: each ConvBNReLU folds its BN into the conv
(per out-channel scale g/sqrt(var+eps): W' = W*s, b' = beta - mean*s). The
schema is deliberately dumb - nested lists a simple custom kernel (or the
numpy reference in reference.py) can consume:

  {"model": "spotter_patch", "classes": [...], "input": "rgb/255 CHW",
   "trunk":  [{"k":3, "stride":S, "pad":1, "relu":true,
               "w": [out][in][3][3], "b": [out]}, ...],
   "head":   {"pool": 4, "w": [classes][32], "b": [classes]}}

Run:  PYTHONPATH=src .venv/bin/python -m spotter.export \
          --ckpt runs/m1/best.pt --out runs/m1/spotter_patch.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import CLASSES
from .model import SpotterNet


def fold_conv_bn(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d):
    """-> (W', b') with the BN's eval-time affine folded into the conv."""
    s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * s[:, None, None, None]
    b = bn.bias - bn.running_mean * s
    if conv.bias is not None:
        b = b + conv.bias * s
    return w, b


def _folded(layer) -> dict:
    w, b = fold_conv_bn(layer.conv, layer.bn)
    return {"k": 3, "stride": layer.conv.stride[0], "pad": 1,
            "relu": True, "w": w.tolist(), "b": b.tolist()}


def export_patch(model: SpotterNet, path: str | Path) -> dict:
    model.eval()
    hw = model.head.classifier.weight[:, :, 0, 0]      # 1x1 conv -> matrix
    doc = {"model": "spotter_patch", "classes": CLASSES,
           "input": "rgb/255 CHW",
           "trunk": [_folded(l) for l in model.trunk.layers],
           "head": {"pool": 4, "w": hw.tolist(),
                    "b": model.head.classifier.bias.tolist()}}
    Path(path).write_text(json.dumps(doc))
    return doc


def export_dense(model: SpotterNet, path: str | Path) -> dict:
    """Trunk + decoder + 1x1 seg head. Decoder steps are explicit ops a
    kernel replays in order: {"up": 2} = x2 nearest-neighbor; {"skip": i,
    ...} = add the 1x1-projected trunk stage i output; else a conv."""
    model.eval()
    dec = []
    for skip_i, skip, layer in ((4, model.decoder.s1, model.decoder.d1),
                                (2, model.decoder.s2, model.decoder.d2),
                                (0, model.decoder.s3, model.decoder.d3)):
        dec.append({"up": 2})
        dec.append({"skip": skip_i,
                    "w": skip.weight[:, :, 0, 0].tolist(),
                    "b": skip.bias.tolist()})
        dec.append(_folded(layer))
    sw = model.decoder.seg.weight[:, :, 0, 0]
    doc = {"model": "spotter_dense", "classes": CLASSES,
           "input": "rgb/255 CHW",
           "trunk": [_folded(l) for l in model.trunk.layers],
           "decoder": dec,
           "seg": {"w": sw.tolist(), "b": model.decoder.seg.bias.tolist()}}
    Path(path).write_text(json.dumps(doc))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("patch", "dense"), default="patch")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # defaults follow the mode, so the README flow needs no explicit paths
    if args.ckpt is None:
        args.ckpt = "runs/m2/best.pt" if args.mode == "dense" else "runs/m1/best.pt"
    if args.out is None:
        args.out = ("runs/m2/spotter_dense.json" if args.mode == "dense"
                    else "runs/m1/spotter_patch.json")
    model = SpotterNet(mode=args.mode)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    (export_patch if args.mode == "patch" else export_dense)(model, args.out)
    kb = Path(args.out).stat().st_size / 1024
    print(f"wrote {args.out} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
