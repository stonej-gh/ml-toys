# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Architecture invariants: param budget and output shapes (docs/SPOTTER-DESIGN.md)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spotter.model import SpotterNet, count_params  # noqa: E402

# Raw training-time counts (BN affine included). At export each BN folds into
# its conv (-2c +c bias per layer): trunk 19,464 -> 19,344, total -> 28,090.
TRUNK_PARAMS = 19_464
HEAD_PARAMS = 165                                   # (32+1) * 5 classes
DECODER_PARAMS = 8_497                              # 3 up-conv stages + additive 1x1 skips (1,180)
TOTAL_PARAMS = TRUNK_PARAMS + HEAD_PARAMS + DECODER_PARAMS


def test_param_budget():
    m = SpotterNet(mode="dense")
    assert count_params(m.trunk) == TRUNK_PARAMS
    assert count_params(m.head) == HEAD_PARAMS
    assert count_params(m.decoder) == DECODER_PARAMS
    assert count_params(m) == TOTAL_PARAMS


def test_output_shapes():
    m = SpotterNet(mode="dense")
    x = torch.zeros(1, 3, 192, 320)
    assert tuple(m(x).shape) == (1, 5, 192, 320)
    hm = m.heatmap(x)  # /8 trunk map minus 4x4 pool margin
    assert tuple(hm.shape) == (1, 5, 21, 37)
    patch = SpotterNet(mode="patch")(torch.zeros(4, 3, 32, 32))
    assert tuple(patch.shape) == (4, 5)
