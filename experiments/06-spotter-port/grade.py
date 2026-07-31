# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Grader for experiment 06 (spotter retrain + port). Run: pytest -m grade_cheap.

A micro edition of the stranger flow: seeded dataset (hash-locked, platform-
exact), 12-epoch from-scratch CPU retrain (~30 s, deterministic per machine),
BN-folded export vs the numpy reference, and int8 quantization through the
shipped bundle's integer forward. Reference numbers, 2026-07-10: val IoU
interceptor 0.518 / fighter 0.673 / laser 0.447 / hole 0.933; float argmax
agreement 1.0 (max delta 2.6e-5); int8 agreement min 0.9994."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "deploy"))

SEED = 20260710
TRAIN_N, VAL_N = 64, 16  # 64 random + replay mix -> 85 frames total
EPOCHS = 12
DATASET_SHA = "7b8260189ec9e86f1616820ec59a66dbf835f2b619d99b3b11fe223f1edf114e"
IOU_FLOOR = 0.15         # every foreground class; reference worst 0.447
FLOAT_AGREE = 0.9999
INT8_AGREE = 0.98


def micro_dataset():
    from tools.gen_dataset import build_split
    rng = np.random.default_rng(SEED)
    assets = REPO / "assets/replays"
    Xtr, Ytr, _ = build_split("train", TRAIN_N, 0.25, assets, rng)
    Xv, Yv, _ = build_split("val", VAL_N, 0.0, assets, rng)
    return Xtr, Ytr, Xv, Yv


@pytest.fixture(scope="module")
def flow():
    """Generate the micro dataset and retrain once for all port tests."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from spotter import CLASSES, IGNORE
    from spotter.model import SpotterNet

    Xtr, Ytr, Xv, Yv = micro_dataset()
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    Y = Ytr.astype(np.int64)
    counts = np.bincount(Y[Y != IGNORE].ravel(), minlength=len(CLASSES))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w = w / w.sum() * len(CLASSES)
    w_t = torch.tensor(w, dtype=torch.float32)

    model = SpotterNet(mode="dense")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    perm_rng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        model.train()
        perm = perm_rng.permutation(len(Xtr))
        for i in range(0, len(perm), 8):
            idx = perm[i:i + 8]
            x = (torch.from_numpy(Xtr[idx]).float().div_(255.0)
                 .permute(0, 3, 1, 2).contiguous())
            loss = F.cross_entropy(model(x), torch.from_numpy(Y[idx]),
                                   weight=w_t, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model, Xv, Yv.astype(np.int64)


@pytest.mark.grade_cheap
def test_dataset_is_bit_reproducible():
    """Rendering is integer-exact: the micro shards hash identically on
    every platform. No torch required."""
    h = hashlib.sha256()
    for a in micro_dataset():
        h.update(a.tobytes())
    assert h.hexdigest() == DATASET_SHA, \
        f"dataset drifted: {h.hexdigest()} != {DATASET_SHA}"


@pytest.mark.grade_cheap
def test_micro_retrain_learns_every_class(flow):
    import torch
    from spotter import CLASSES, IGNORE
    model, Xv, Yv = flow
    ncls = len(CLASSES)
    inter, union = np.zeros(ncls), np.zeros(ncls)
    with torch.no_grad():
        for i in range(0, len(Xv), 8):
            x = (torch.from_numpy(Xv[i:i + 8]).float().div_(255.0)
                 .permute(0, 3, 1, 2).contiguous())
            pred = model(x).argmax(1).numpy()
            y = Yv[i:i + 8]
            valid = y != IGNORE
            for c in range(ncls):
                p, t = (pred == c) & valid, (y == c) & valid
                inter[c] += (p & t).sum()
                union[c] += (p | t).sum()
    iou = inter / np.maximum(union, 1)
    print("\nval IoU:", {c: round(float(v), 3) for c, v in zip(CLASSES, iou)})
    for c, v in list(zip(CLASSES, iou))[1:]:  # foreground classes
        assert v >= IOU_FLOOR, f"{c} IoU {v:.3f} < {IOU_FLOOR}"


@pytest.mark.grade_cheap
def test_float_port_matches_torch(flow, tmp_path):
    import torch
    from spotter.export import export_dense
    from spotter.reference import forward_dense, load_doc
    model, Xv, _ = flow
    export_dense(model, tmp_path / "dense.json")
    doc = load_doc(tmp_path / "dense.json")
    agree_min = 1.0
    with torch.no_grad():
        for i in range(len(Xv)):
            x = (Xv[i].astype(np.float32) / 255.0).transpose(2, 0, 1)
            want = model(torch.from_numpy(x)[None]).numpy()[0]
            got = forward_dense(x, doc)
            agree_min = min(agree_min,
                            float((got.argmax(0) == want.argmax(0)).mean()))
    print(f"\nfloat port: min argmax agreement {agree_min:.6f}")
    assert agree_min >= FLOAT_AGREE


@pytest.mark.grade_cheap
def test_int8_port_holds(flow, tmp_path):
    import json
    from spotter.export import export_dense
    from spotter.quantize import calibrate, quantize
    from spotter_forward import forward_dense, forward_dense_int8, load_doc
    model, Xv, _ = flow
    export_dense(model, tmp_path / "dense.json")
    doc = load_doc(tmp_path / "dense.json")
    doc8_raw = quantize(doc, calibrate(doc, Xv))
    (tmp_path / "int8.json").write_text(json.dumps(doc8_raw))
    doc8 = load_doc(tmp_path / "int8.json")
    agree_min = 1.0
    for i in range(len(Xv)):
        x = (Xv[i].astype(np.float32) / 255.0).transpose(2, 0, 1)
        pf = forward_dense(x, doc).argmax(0)
        _, pq = forward_dense_int8(x, doc8)
        agree_min = min(agree_min, float((pq == pf).mean()))
    print(f"\nint8 port: min argmax agreement vs float {agree_min:.4f}")
    assert agree_min >= INT8_AGREE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-m", "grade_cheap", "-q", "-s"]))
