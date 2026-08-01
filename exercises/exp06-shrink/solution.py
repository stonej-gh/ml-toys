#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, experiment 06 "Shrink it": how small can the spotter go?

Scales the trunk's channel budget (WIDTHS in spotter/model.py) by a factor,
retrains the micro edition of the stranger flow at each size, and reports
per-class validation IoU against the micro grader's floor. The question the
sweep answers: which gate fails first, and on which class?

    python exercises/exp06-shrink/solution.py                 # 1.0x, 0.5x, 0.25x
    python exercises/exp06-shrink/solution.py --scales 1.0 0.5

Each size costs one ~30 s CPU retrain. Walkthrough: README.md here.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import spotter.model as spotter_model
from spotter import CLASSES, IGNORE
from tools.gen_dataset import build_split

SEED = 20260710
TRAIN_N, VAL_N = 64, 16
EPOCHS = 12
IOU_FLOOR = 0.15     # the micro grader's bar (experiments/06-spotter-port)
BASE = [8, 16, 16, 24, 24, 32]


def micro_dataset():
    rng = np.random.default_rng(SEED)
    assets = REPO / "assets/replays"
    Xtr, Ytr, _ = build_split("train", TRAIN_N, 0.25, assets, rng)
    Xv, Yv, _ = build_split("val", VAL_N, 0.0, assets, rng)
    return Xtr, Ytr.astype(np.int64), Xv, Yv.astype(np.int64)


def train_at(widths, Xtr, Y, Xv, Yv):
    """The micro retrain from the experiment's grader, at a chosen budget."""
    spotter_model.WIDTHS[:] = widths  # the model reads this list at build time
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    counts = np.bincount(Y[Y != IGNORE].ravel(), minlength=len(CLASSES))
    w = 1.0 / np.sqrt(np.maximum(counts, 1))
    w_t = torch.tensor(w / w.sum() * len(CLASSES), dtype=torch.float32)
    model = spotter_model.SpotterNet(mode="dense")
    n_par = sum(p.numel() for p in model.parameters())
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
    return n_par, inter / np.maximum(union, 1)


def main():
    argv = sys.argv
    scales = [float(s) for s in argv[argv.index("--scales") + 1:]] \
        if "--scales" in argv else [1.0, 0.5, 0.25]
    Xtr, Y, Xv, Yv = micro_dataset()
    print(f"micro flow: {len(Xtr)} train / {len(Xv)} val frames, "
          f"{EPOCHS} epochs, IoU floor {IOU_FLOOR} per foreground class\n")
    for s in scales:
        widths = [max(2, round(w * s)) for w in BASE]
        n_par, iou = train_at(widths, Xtr, Y, Xv, Yv)
        cells = "  ".join(f"{c} {v:.3f}" for c, v in zip(CLASSES, iou))
        fails = [c for c, v in list(zip(CLASSES, iou))[1:] if v < IOU_FLOOR]
        verdict = "PASS" if not fails else f"FAIL: {', '.join(fails)}"
        print(f"scale {s:4.2f}  widths {widths}  params {n_par:6d}\n"
              f"           {cells}\n           {verdict}\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
