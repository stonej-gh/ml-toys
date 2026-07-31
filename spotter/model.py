# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""spotter architecture - the single source of truth for the network.

Design constraints (see docs/SPOTTER-DESIGN.md):
  - tiny (~28k params) and int8-quantizable
  - ops a simple custom kernel can do: 3x3 conv, 1x1 conv, stride-2 conv,
    4x4 avg-pool, x2 nearest-neighbor upsample, elementwise add, ReLU
  - BatchNorm in training, FOLDED into conv weights at export (the deployed
    bundle has no BN layers)
  - one shared Trunk; a light M1 patch head OR an M2 dense decoder on top

Channel widths are single constants (WIDTHS) so the budget can be retuned in
one place if validation quality demands it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import CLASSES

NUM_CLASSES = len(CLASSES)

# Trunk channel schedule. Strides applied at conv2/conv4/conv6 (=> /8 total).
WIDTHS = [8, 16, 16, 24, 24, 32]
STRIDES = [1, 2, 1, 2, 1, 2]


class ConvBNReLU(nn.Module):
    """3x3 conv + BN + ReLU. BN is folded into the conv at export time."""

    def __init__(self, cin: int, cout: int, stride: int = 1, k: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Trunk(nn.Module):
    """Shared feature extractor: 3-channel input -> 32-channel /8 feature map."""

    def __init__(self):
        super().__init__()
        cin = 3
        layers = []
        for cout, stride in zip(WIDTHS, STRIDES):
            layers.append(ConvBNReLU(cin, cout, stride=stride))
            cin = cout
        self.layers = nn.ModuleList(layers)
        self.out_ch = WIDTHS[-1]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_all(self, x):
        """All six stage outputs (the decoder taps /1, /2, /4 as skips)."""
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class PatchHead(nn.Module):
    """Milestone 1 patch-classifier head.

    Trained on 32x32 patches: trunk -> 4x4 feature map -> 4x4 avg-pool -> 1x1
    conv -> NUM_CLASSES logits. At inference the SAME weights run
    fully-convolutionally over a full frame's trunk output (avg-pool as a
    sliding window) to yield a coarse class heatmap.
    """

    def __init__(self, cin: int):
        super().__init__()
        self.classifier = nn.Conv2d(cin, NUM_CLASSES, 1)

    def forward(self, feat, *, dense: bool = False):
        if dense:
            # sliding 4x4 average (stride 1) preserves the heatmap resolution;
            # equivalent to classifying every 32x32 window of the input
            pooled = F.avg_pool2d(feat, 4, stride=1, padding=0)
            return self.classifier(pooled)
        pooled = F.adaptive_avg_pool2d(feat, 1)  # patch -> 1x1
        return self.classifier(pooled).flatten(1)


class Decoder(nn.Module):
    """Milestone 2 dense decoder: upsample the /8 trunk feature back to full
    resolution and emit a per-pixel class map. Nearest-neighbor upsampling +
    3x3 conv (no transposed conv), with ADDITIVE 1x1-projected skips from the
    trunk's /4, /2, /1 stages - elementwise adds stay int8-simple (requantize;
    no concat buffers). Skips were added after the no-skip decoder plateaued
    at IoU ~0.59 on the small sprites (see docs/SPOTTER-DESIGN.md lessons)."""

    def __init__(self, cin: int):
        super().__init__()
        self.s1 = nn.Conv2d(WIDTHS[4], cin, 1)   # /4 skip (deeper 24ch stage)
        self.d1 = ConvBNReLU(cin, 16)            # after x2 -> 80x48
        self.s2 = nn.Conv2d(WIDTHS[2], 16, 1)    # /2 skip (deeper 16ch stage)
        self.d2 = ConvBNReLU(16, 12)             # after x2 -> 160x96
        self.s3 = nn.Conv2d(WIDTHS[0], 12, 1)    # /1 skip
        self.d3 = ConvBNReLU(12, 8)              # after x2 -> 320x192
        self.seg = nn.Conv2d(8, NUM_CLASSES, 1)

    def forward(self, feats):
        x = F.interpolate(feats[5], scale_factor=2, mode="nearest")
        x = self.d1(x + self.s1(feats[4]))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d2(x + self.s2(feats[2]))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.d3(x + self.s3(feats[0]))
        return self.seg(x)


class SpotterNet(nn.Module):
    """Full model. mode='patch' (M1) or mode='dense' (M2). The dense-mask path
    and the M1 heatmap path share the trunk, so M2 fine-tunes from M1 weights."""

    def __init__(self, mode: str = "dense"):
        super().__init__()
        assert mode in ("patch", "dense")
        self.mode = mode
        self.trunk = Trunk()
        self.head = PatchHead(self.trunk.out_ch)
        self.decoder = Decoder(self.trunk.out_ch) if mode == "dense" else None

    def forward(self, x):
        if self.mode == "patch":
            return self.head(self.trunk(x))
        return self.decoder(self.trunk.forward_all(x))

    def heatmap(self, x):
        """M1 coarse class heatmap over a full frame (fully-convolutional)."""
        return self.head(self.trunk(x), dense=True)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    # Param-budget self-check - the numbers docs/SPOTTER-DESIGN.md commits to.
    m = SpotterNet(mode="dense")
    trunk = count_params(m.trunk)
    head = count_params(m.head)
    dec = count_params(m.decoder)
    print(f"trunk   : {trunk:,}")
    print(f"M1 head : {head:,}  (trunk+head = {trunk + head:,})")
    print(f"decoder : {dec:,}")
    print(f"TOTAL   : {trunk + head + dec:,}")
    x = torch.zeros(1, 3, 192, 320)
    print("dense out:", tuple(m(x).shape))
    print("heatmap  :", tuple(m.heatmap(x).shape))
    p = torch.zeros(4, 3, 32, 32)
    print("patch out:", tuple(SpotterNet(mode="patch")(p).shape))
