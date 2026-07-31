# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The float gate: BN-folded JSON export vs the pure-numpy reference.

Uses a randomly-initialized net with exercised BN running stats - the gate
must hold for ANY weights, not just a lucky trained set. Trained-model gates
re-run the same checks via the export CLI at milestone time.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotter.export import export_dense, export_patch           # noqa: E402
from spotter.model import SpotterNet                            # noqa: E402
from spotter.reference import (forward_dense, forward_heatmap,  # noqa: E402
                               forward_patch, load_doc)
from spotter.render import render                               # noqa: E402
from spotter.sample import sample_scene                         # noqa: E402

# fp32 numpy and torch sum in different orders; observed max |delta| is ~1e-6
# on logits of order 1. Threshold frozen two decades above that observation.
ATOL = 1e-4


def _fresh_model():
    torch.manual_seed(1234)
    m = SpotterNet(mode="patch")
    m.train()
    for _ in range(3):  # move BN stats off their defaults
        m(torch.randn(8, 3, 32, 32))
    return m.eval()


def _export(tmp_path):
    m = _fresh_model()
    doc_path = tmp_path / "model.json"
    export_patch(m, doc_path)
    return m, load_doc(doc_path)


def test_patch_logits_match(tmp_path):
    m, doc = _export(tmp_path)
    rng = np.random.default_rng(0)
    x = rng.random((64, 3, 32, 32), dtype=np.float32)
    with torch.no_grad():
        want = m(torch.from_numpy(x)).numpy()
    got = np.stack([forward_patch(p, doc) for p in x])
    assert np.abs(got - want).max() < ATOL
    assert (got.argmax(1) == want.argmax(1)).all()  # 100% argmax agreement


def test_dense_matches_on_real_frame(tmp_path):
    torch.manual_seed(4321)
    m = SpotterNet(mode="dense")
    m.train()
    for _ in range(3):
        m(torch.randn(2, 3, 64, 64))
    m.eval()
    doc_path = tmp_path / "dense.json"
    export_dense(m, doc_path)
    doc = load_doc(doc_path)
    frame, _ = render(sample_scene(6))
    x = (np.asarray(frame).astype(np.float32) / 255.0).transpose(2, 0, 1)
    with torch.no_grad():
        want = m(torch.from_numpy(x)[None]).numpy()[0]
    got = forward_dense(x, doc)
    assert got.shape == want.shape == (5, 192, 320)
    assert np.abs(got - want).max() < ATOL
    assert (got.argmax(0) == want.argmax(0)).mean() == 1.0


def test_heatmap_matches_on_real_frame(tmp_path):
    m, doc = _export(tmp_path)
    frame, _ = render(sample_scene(5))
    x = (np.asarray(frame).astype(np.float32) / 255.0).transpose(2, 0, 1)
    with torch.no_grad():
        want = m.heatmap(torch.from_numpy(x)[None]).numpy()[0]
    got = forward_heatmap(x, doc)
    assert got.shape == want.shape == (5, 21, 37)
    assert np.abs(got - want).max() < ATOL
    agree = (got.argmax(0) == want.argmax(0)).mean()
    assert agree == 1.0
