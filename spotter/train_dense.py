# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""M2 training: dense per-pixel segmentation, fine-tuned from the M1 trunk.

Adds domain-randomization augmentation (photometric only - masks unchanged):
per-sample brightness gain/bias, per-channel gain, gaussian noise. A fixed
seeded STRESS variant of the test split (stronger noise/jitter) is evaluated
alongside clean splits so robustness is measured, not assumed.

Run from the repo root:
    PYTHONPATH=src .venv/bin/python -m spotter.train_dense

Gates (docs/DESIGN.md): per-class IoU >= 0.70 on val, >= 0.60 on the fully
held-out replay episode, >= 0.50 on the stress set. Artifacts in runs/m2/.
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
from .datasets import load_split
from .model import SpotterNet
from .render import load_replay_scenes, render

GATE_VAL, GATE_HELDOUT, GATE_STRESS = 0.70, 0.60, 0.50


def to_frames(X, idx, device):
    x = torch.from_numpy(X[idx]).to(device).float().div_(255.0)
    return x.permute(0, 3, 1, 2).contiguous()


def augment(x: torch.Tensor) -> torch.Tensor:
    """Photometric domain randomization; label-preserving by construction."""
    n = x.shape[0]
    gain = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.3
    bias = (torch.rand(n, 1, 1, 1, device=x.device) - 0.5) * 0.16
    ch = 1.0 + (torch.rand(n, 3, 1, 1, device=x.device) - 0.5) * 0.2
    sigma = torch.rand(n, 1, 1, 1, device=x.device) * (8.0 / 255.0)
    x = x * gain * ch + bias + torch.randn_like(x) * sigma
    return x.clamp_(0.0, 1.0)


def stress(frames: np.ndarray, seed: int = 77) -> np.ndarray:
    """Fixed, stronger perturbation of a uint8 frame stack (eval only)."""
    rng = np.random.default_rng(seed)
    x = frames.astype(np.float32) / 255.0
    n = len(x)
    gain = 1.0 + (rng.random((n, 1, 1, 1), dtype=np.float32) - 0.5) * 0.4
    bias = (rng.random((n, 1, 1, 1), dtype=np.float32) - 0.5) * 0.24
    noise = rng.normal(0.0, 12.0 / 255.0, x.shape).astype(np.float32)
    x = np.clip(x * gain + bias + noise, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


@torch.no_grad()
def iou(model, X, Y, device, batch=16):
    """Per-class IoU over a frame stack; IGNORE pixels excluded."""
    model.eval()
    ncls = len(CLASSES)
    inter = np.zeros(ncls)
    union = np.zeros(ncls)
    for i in range(0, len(X), batch):
        idx = np.arange(i, min(i + batch, len(X)))
        pred = model(to_frames(X, idx, device)).argmax(1).cpu().numpy()
        y = Y[idx]
        valid = y != IGNORE
        for c in range(ncls):
            p, t = (pred == c) & valid, (y == c) & valid
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
    return inter / np.maximum(union, 1)


def heldout_stack():
    """Render the fully held-out episode (never in any training shard)."""
    scenes = load_replay_scenes(
        Path(__file__).resolve().parents[1] / "assets/replays/seed_005.json")
    frames, masks = [], []
    for sc in scenes[::3]:
        im, mk = render(sc)
        frames.append(np.asarray(im))
        masks.append(np.asarray(mk))
    return np.stack(frames), np.stack(masks).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/derived")
    ap.add_argument("--out", default="runs/m2")
    ap.add_argument("--init", default="runs/m1/best.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260708)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Path(args.data)
    X, Y = load_split(data / "train.npz")
    Y = Y.astype(np.int64)
    Xv, Yv = load_split(data / "val.npz")
    Yv = Yv.astype(np.int64)

    counts = np.bincount(Y[Y != IGNORE].ravel(), minlength=len(CLASSES))
    weights = 1.0 / np.sqrt(np.maximum(counts, 1))
    weights = weights / weights.sum() * len(CLASSES)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"frames: train {len(X)}, val {len(Xv)}; device {device}")
    print("pixel counts:", counts.tolist())

    model = SpotterNet(mode="dense")
    missing = model.load_state_dict(
        torch.load(args.init, map_location="cpu"), strict=False)
    print(f"init from {args.init}; new params: "
          f"{[k.split('.')[0] for k in missing.missing_keys[:1]]}...")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = {"val_min_iou": -1.0}
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(X))
        losses = []
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            logits = model(augment(to_frames(X, idx, device)))
            y = torch.from_numpy(Y[idx]).to(device)
            loss = F.cross_entropy(logits, y, weight=w_t, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val = iou(model, Xv, Yv, device)
        print(f"epoch {epoch:2d}  loss {np.mean(losses):.4f}  "
              f"val IoU {[f'{v:.3f}' for v in val]}  min {val.min():.3f}")
        if val.min() > best["val_min_iou"]:
            best = {"val_min_iou": float(val.min()),
                    "val_iou": {c: float(v) for c, v in zip(CLASSES, val)},
                    "epoch": epoch}
            torch.save(model.state_dict(), out / "best.pt")

    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    Xt, Yt = load_split(data / "test.npz")
    Yt = Yt.astype(np.int64)
    Xh, Yh = heldout_stack()
    evals = {"test": iou(model, Xt, Yt, device),
             "heldout_episode": iou(model, Xh, Yh, device),
             "stress": iou(model, stress(Xt), Yt, device)}
    for k, v in evals.items():
        best[f"{k}_iou"] = {c: float(x) for c, x in zip(CLASSES, v)}
    best.update(seed=args.seed, epochs=args.epochs, lr=args.lr,
                batch=args.batch, init=args.init,
                train_s=round(time.time() - t0, 1))
    gates = {"val": best["val_min_iou"] >= GATE_VAL,
             "heldout": min(best["heldout_episode_iou"].values()) >= GATE_HELDOUT,
             "stress": min(best["stress_iou"].values()) >= GATE_STRESS}
    best["gates"] = gates
    (out / "metrics.json").write_text(json.dumps(best, indent=1))
    print("best:", json.dumps(best, indent=1))
    print("GATES:", "PASS" if all(gates.values()) else "FAIL")


if __name__ == "__main__":
    main()
