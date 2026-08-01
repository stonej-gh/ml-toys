#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, stage 5: port the pilot by hand, then shrink it to int8.

Three parts, in the order a real hardware port would do them:

1. A from-scratch forward pass, written against nothing but the .json weight
   file. No imports from orbitduel/netpilot.py: the point is to prove the
   file plus the spec is enough to rebuild the brain.
2. An equality check against the repo's reference forward on 100 random
   observations. All 100 action picks must match before anything else counts.
3. Weight-only int8 quantization (symmetric, per layer), then a measured
   win-rate comparison of float vs int8 flying the real duel against the
   ported L10. The accuracy-vs-precision comparison is the whole edge-AI
   methodology in miniature.

    python exercises/stage5-port/solution.py                 # all three parts
    python exercises/stage5-port/solution.py --episodes 20   # quicker part 3

Walkthrough with the measured results: README.md in this directory.
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orbitduel.env import OrbitDuelEnv
from orbitduel.netpilot import PolicyNet
from orbitduel.pilot import ScriptedPilot
from orbitduel import physics as P

MODEL = Path(__file__).resolve().parents[2] / "agents/models" \
    / f"duel_ppo_ported_{P.FIELD_NAME}.json"
LEVEL = 10
EPISODES = 60


def load(path):
    spec = json.loads(Path(path).read_text())
    return spec["layers"], spec.get("activation", "tanh"), spec["obs_dim"]


def forward(layers, activation, x):
    """The whole brain: matrix multiply, add bias, bend, repeat; last layer straight."""
    for k, layer in enumerate(layers):
        out = []
        for row, bias in zip(layer["w"], layer["b"]):
            s = bias
            for wi, xi in zip(row, x):
                s += wi * xi
            if k < len(layers) - 1:
                s = math.tanh(s) if activation == "tanh" else max(s, 0.0)
            out.append(s)
        x = out
    return x


def argmax(v):
    return max(range(len(v)), key=v.__getitem__)


def quantize_int8(layers, per_row=False):
    """Weight-only symmetric int8: weights map onto -127..127 by a scale.

    scale = max|w| / 127, taken over the whole layer, or over each output row
    when per_row is set (the same idea as the spotter's per-channel scales: a
    quiet row no longer wastes its 254 levels on the loudest row's range).
    Every weight becomes round(w / scale) and the forward pass multiplies the
    scale back in. The biases stay float: there are only a few of them and
    they set the working point, so shrinking them buys nothing and risks a lot."""
    q = []
    for layer in layers:
        rows, scales = [], []
        layer_top = max(abs(w) for row in layer["w"] for w in row)
        for row in layer["w"]:
            top = max(abs(w) for w in row) if per_row else layer_top
            s = (top or 1e-12) / 127.0
            scales.append(s)
            rows.append([round(w / s) for w in row])
        q.append({"w": rows, "b": layer["b"], "scales": scales})
    return q


def forward_int8(qlayers, activation, x):
    """Same shape as forward(); weights are ints, rescaled as they are used."""
    for k, layer in enumerate(qlayers):
        out = []
        for row, bias, s_w in zip(layer["w"], layer["b"], layer["scales"]):
            s = bias
            for wi, xi in zip(row, x):
                s += wi * s_w * xi
            if k < len(qlayers) - 1:
                s = math.tanh(s) if activation == "tanh" else max(s, 0.0)
            out.append(s)
        x = out
    return x


def duel(act, episodes, seed0=40_000):
    wins = 0
    for ep in range(episodes):
        env = OrbitDuelEnv(opponent=ScriptedPilot(LEVEL, seed=seed0 + ep),
                           rules="v6-full", seed=seed0 + ep)
        obs, info = env.reset(seed=seed0 + ep)
        while True:
            obs, r, term, trunc, info = env.step(act(obs))
            if term or trunc:
                break
        wins += info.get("outcome") == "win"
    return wins


def main():
    argv = sys.argv
    episodes = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv \
        else EPISODES
    layers, activation, obs_dim = load(MODEL)
    ref = PolicyNet(MODEL)
    print(f"model: {MODEL.name}  ({obs_dim} inputs, {len(layers)} layers)")

    # part 2: the port is correct only if it NEVER disagrees with the reference
    rng = random.Random(0)
    agree = 0
    for _ in range(100):
        obs = [rng.uniform(-1, 1) for _ in range(obs_dim)]
        agree += argmax(forward(layers, activation, obs)) == ref.act(obs)
    print(f"hand-rolled vs reference forward: {agree}/100 identical actions")
    if agree != 100:
        print("STOP: fix the port before measuring anything else")
        return 1

    # part 3: what does 4x smaller cost, in wins, against the real top rung?
    # Two schemes: one scale per layer (the crude first try) and one scale per
    # output row (the spotter's per-channel idea, applied to the pilot).
    for per_row, label in ((False, "per-layer"), (True, "per-row ")):
        qlayers = quantize_int8(layers, per_row=per_row)
        flips = 0
        for _ in range(1_000):
            obs = [rng.uniform(-1, 1) for _ in range(obs_dim)]
            flips += argmax(forward(layers, activation, obs)) \
                != argmax(forward_int8(qlayers, activation, obs))
        wq = duel(lambda o: argmax(forward_int8(qlayers, activation, o)), episodes)
        print(f"int8 {label} scales: {flips}/1,000 random decisions differ "
              f"from float; {wq} wins vs L10 over {episodes} episodes")

    wf = duel(lambda o: argmax(forward(layers, activation, o)), episodes)
    print(f"float reference:        {wf} wins on the same {episodes} seeds")
    print("weights at fp32: 24,880 bytes; at int8: 6,220 bytes plus the scales")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
