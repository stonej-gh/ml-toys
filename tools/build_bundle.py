#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Assemble (or reproduce) the deploy/ golden bundle from the M2 artifacts.

    .venv/bin/python tools/build_bundle.py          # then: python deploy/verify.py

Deterministic end to end: the frozen float JSON is copied verbatim, int8
calibration frames and golden frames are chosen by SEED, the renderer is
seeded, and the int8 path is integer - re-running reproduces the committed
bundle bit-for-bit (the --emit-golden reproducibility gate).
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))
from spotter.quantize import calibrate, quantize                 # noqa: E402
from spotter.render import load_replay_scenes, render            # noqa: E402
from spotter_forward import (forward_dense, forward_dense_int8,  # noqa: E402
                             load_doc)

SEED = 20260708
N_GOLDEN = 8
FLOAT_SRC = ROOT / "runs/m2/spotter_dense.json"
REPLAY = ROOT / "assets/replays/seed_005.json"                   # fully held-out episode


def main():
    deploy = ROOT / "deploy"
    (deploy / "models").mkdir(exist_ok=True)
    (deploy / "golden").mkdir(exist_ok=True)

    # 1. freeze float model
    dst = deploy / "models/spotter_dense.json"
    shutil.copyfile(FLOAT_SRC, dst)
    doc = load_doc(dst)

    # 2. quantize with seeded calibration
    data = np.load(ROOT / "data/derived/train.npz")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(data["frames"]), size=64, replace=False)
    scales = calibrate(doc, data["frames"][idx])
    doc_f = json.loads(dst.read_text())
    for layer in doc_f["trunk"] + doc_f["decoder"] + [doc_f["seg"]]:
        for k in ("w", "b"):
            if k in layer:
                layer[k] = np.asarray(layer[k], dtype=np.float32)
    doc8_raw = quantize(doc_f, scales)
    q_path = deploy / "models/spotter_dense_int8.json"
    q_path.write_text(json.dumps(doc8_raw))
    doc8 = load_doc(q_path)

    # 3. golden vectors from seeded held-out frames
    scenes = load_replay_scenes(REPLAY)
    frames_i = sorted(np.random.default_rng(SEED)
                      .choice(len(scenes), size=N_GOLDEN, replace=False)
                      .tolist())
    inputs, f_sub, f_arg, q_sub, q_arg = [], [], [], [], []
    for i in frames_i:
        im, _ = render(scenes[i])
        arr = np.asarray(im)
        x = (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)
        lg = forward_dense(x, doc)
        lq, pq = forward_dense_int8(x, doc8)
        inputs.append(arr)
        f_sub.append(lg[:, 4::8, 4::8].astype(np.float32))
        f_arg.append(lg.argmax(0).astype(np.uint8))
        q_sub.append(lq[:, 4::8, 4::8].astype(np.float32))
        q_arg.append(pq)
    np.savez_compressed(
        deploy / "golden/spotter_dense.golden.npz",
        inputs=np.stack(inputs), float_logits_sub=np.stack(f_sub),
        float_argmax=np.stack(f_arg), int8_logits_sub=np.stack(q_sub),
        int8_argmax=np.stack(q_arg))
    (deploy / "golden/spotter_dense.golden.json").write_text(json.dumps(
        {"seed": SEED, "replay": REPLAY.name, "frame_indices": frames_i,
         "classes": doc["classes"], "subgrid": "logits[:, 4::8, 4::8]",
         "float_tol": 1e-4, "int8_agree_gate": 0.99, "int8_iou_gate": 0.98},
        indent=1))
    for p in sorted(deploy.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size/1024:.0f} KB")
    print("bundle built; run: python deploy/verify.py")


if __name__ == "__main__":
    main()
