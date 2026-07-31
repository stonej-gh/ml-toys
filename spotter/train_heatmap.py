# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""M1 training: trunk + patch head trained DENSELY on full frames.

Why dense and not on cropped patches: an isolated 32x32 patch zero-pads every
conv at its borders, but the same window inside a full frame sees real
neighbors (and the trunk's receptive field is wider than 32 px anyway) - a
patch-trained net collapses to all-background when run fully-convolutionally.
Training directly on the full-frame heatmap against the cell-label grid
(datasets.heatmap_labels) removes the mismatch; the architecture and export
are unchanged.

Run from the repo root:
    python -m spotter.train_heatmap

Seeded and reproducible from metrics.json. Gates (docs/SPOTTER-DESIGN.md): val cell
accuracy >= 0.90 overall and per-class recall >= 0.80. Cross-entropy is
class-weighted (inverse-sqrt frequency) - background cells outnumber laser
cells ~10^3:1. Best checkpoint + metrics land in runs/m1/ (untracked).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import CLASSES, IGNORE
from .datasets import heatmap_labels, load_split
from .model import SpotterNet

GATE_OVERALL, GATE_PER_CLASS = 0.90, 0.80


def to_frames(X, idx, device):
    x = torch.from_numpy(X[idx]).to(device).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def evaluate(model, X, Y, device, batch=32):
    model.eval()
    ncls = len(CLASSES)
    correct = np.zeros(ncls)
    count = np.zeros(ncls)
    for i in range(0, len(X), batch):
        idx = np.arange(i, min(i + batch, len(X)))
        pred = model.heatmap(to_frames(X, idx, device)).argmax(1).cpu().numpy()
        y = Y[idx]
        for c in range(ncls):
            sel = y == c
            count[c] += sel.sum()
            correct[c] += (pred[sel] == c).sum()
    per_class = correct / np.maximum(count, 1)
    return correct.sum() / count.sum(), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived")
    ap.add_argument("--out", default="runs/m1")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Path(args.data)
    X, masks = load_split(data / "train.npz")
    Y = heatmap_labels(masks)
    Xv, masks_v = load_split(data / "val.npz")
    Yv = heatmap_labels(masks_v)

    counts = np.bincount(Y[Y != IGNORE].ravel(), minlength=len(CLASSES))
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.sum() * len(CLASSES)
    print(f"frames: train {len(X)}, val {len(Xv)}; device {device}")
    print("cell class counts:", counts.tolist())
    print("loss weights:", [f"{w:.2f}" for w in weights])
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)

    model = SpotterNet(mode="patch").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = {"val_acc": 0.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(X))
        losses = []
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            hm = model.heatmap(to_frames(X, idx, device))
            y = torch.from_numpy(Y[idx]).to(device)
            loss = F.cross_entropy(hm, y, weight=w_t, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        acc, per_class = evaluate(model, Xv, Yv, device)
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  val {acc:.4f}  "
              f"per-class {[f'{p:.3f}' for p in per_class]}")
        if acc > best["val_acc"]:
            best = {"val_acc": float(acc),
                    "per_class": {c: float(p)
                                  for c, p in zip(CLASSES, per_class)},
                    "epoch": epoch}
            torch.save(model.state_dict(), out / "best.pt")

    best.update(seed=args.seed, epochs=args.epochs, lr=args.lr,
                batch=args.batch, train_frames=len(X), val_frames=len(Xv),
                loss_weights=[round(float(w), 3) for w in weights],
                train_s=round(time.time() - t0, 1))
    gates = {"overall": best["val_acc"] >= GATE_OVERALL,
             "per_class": all(v >= GATE_PER_CLASS
                              for v in best["per_class"].values())}
    best["gates"] = gates
    (out / "metrics.json").write_text(json.dumps(best, indent=1))
    print("best:", json.dumps(best, indent=1))
    print("GATES:", "PASS" if all(gates.values()) else "FAIL")


if __name__ == "__main__":
    main()
