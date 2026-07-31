#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Golden-bundle verifier - numpy only, relocatable, no training repo needed.

    python verify.py            # from anywhere; paths resolve to this file

Checks, in ladder order (docs/SPOTTER-DESIGN.md in the source repo):
  1. FLOAT : recomputed logits match golden subgrid within TOL (1e-4, frozen
             after measurement); per-pixel argmax matches stored 100%
  2. INT8  : recomputed integer path matches stored argmax BIT-EXACTLY;
             int8-vs-float argmax agreement >= 99%; per-class IoU vs the
             float masks >= 0.98

A hardware port passes the same bundle by replacing spotter_forward with its
own implementation and running this script unchanged.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spotter_forward import forward_dense, forward_dense_int8, load_doc  # noqa: E402

TOL = 1e-4
GATE_AGREE = 0.99
GATE_IOU = 0.98


def main() -> int:
    doc = load_doc(HERE / "models/spotter_dense.json")
    doc8 = load_doc(HERE / "models/spotter_dense_int8.json")
    g = np.load(HERE / "golden/spotter_dense.golden.npz")
    meta = json.loads((HERE / "golden/spotter_dense.golden.json").read_text())
    n = len(g["inputs"])
    print(f"golden set: {n} frames, seed {meta['seed']}")

    ok = True
    ncls = len(meta["classes"])
    worst_d, agree_min = 0.0, 1.0
    inter, union = np.zeros(ncls), np.zeros(ncls)
    for i in range(n):
        x = (g["inputs"][i].astype(np.float32) / 255.0).transpose(2, 0, 1)
        lg = forward_dense(x, doc)
        worst_d = max(worst_d,
                      float(np.abs(lg[:, 4::8, 4::8]
                                   - g["float_logits_sub"][i]).max()))
        pf = lg.argmax(0).astype(np.uint8)
        if not (pf == g["float_argmax"][i]).all():
            print(f"FAIL float argmax mismatch frame {i}")
            ok = False
        _, pq = forward_dense_int8(x, doc8)
        if not (pq == g["int8_argmax"][i]).all():
            print(f"FAIL int8 argmax not bit-exact frame {i}")
            ok = False
        agree_min = min(agree_min, float((pq == pf).mean()))
        for c in range(ncls):
            inter[c] += ((pf == c) & (pq == c)).sum()
            union[c] += ((pf == c) | (pq == c)).sum()

    print(f"float : max|delta| {worst_d:.2e} (tol {TOL})  "
          f"{'OK' if worst_d < TOL else 'FAIL'}")
    ok &= worst_d < TOL
    print(f"int8  : argmax agreement vs float {agree_min:.5f} "
          f"(gate {GATE_AGREE})  {'OK' if agree_min >= GATE_AGREE else 'FAIL'}")
    ok &= agree_min >= GATE_AGREE

    # standard dataset-aggregated IoU: sum(inter)/sum(union) over the set
    for c, name in enumerate(meta["classes"]):
        v = inter[c] / union[c] if union[c] else 1.0
        good = v >= GATE_IOU
        print(f"int8  : IoU vs float {name:12s} {v:.4f} (gate {GATE_IOU})  "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
    print("VERIFY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
