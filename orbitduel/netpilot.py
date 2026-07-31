# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Dependency-free forward pass for exported policy nets (the .json bundles).

The trainers export policies as plain weight arrays (export_json in
agents/ppo_duel.py and agents/dqn_survive.py): row-major W and b per Linear
layer, tanh or relu trunk, argmax head, plus the flattened action table. This module re-implements inference in
pure Python floats so that:

  * golden evaluations are bit-identical on every platform (no BLAS, no
    framework, IEEE double all the way), and
  * anyone porting a policy to new hardware has a reference implementation
    with zero dependencies to diff against, mirroring the spotter bundle's
    numpy reference.

A 2x64 net is ~6k multiply-accumulates per decision; pure Python does that
in well under a millisecond, which is plenty for evaluation and replays.
"""

import json
import math

from . import physics as P
from .env import ACTIONS, FuelTank, gate_fire, obs_from


def _forward(layers, activation, x):
    """x -> logits through [{'w': [[...]], 'b': [...]}, ...]; last layer linear."""
    act = math.tanh if activation == "tanh" else (lambda v: v if v > 0 else 0.0)
    for k, layer in enumerate(layers):
        w, b = layer["w"], layer["b"]
        out = []
        for row, bias in zip(w, b):
            s = bias
            for wi, xi in zip(row, x):
                s += wi * xi
            out.append(s if k == len(layers) - 1 else act(s))
        x = out
    return x


class PolicyNet:
    """A frozen exported policy: load once, call act(obs) -> action index."""

    def __init__(self, path):
        spec = json.loads(open(path).read())
        self.obs_dim = spec["obs_dim"]
        self.actions = [tuple(a) for a in spec["actions"]]
        self.activation = spec.get("activation", "tanh")
        self.layers = spec["layers"]

    def logits(self, obs):
        x = [float(v) for v in obs]
        if len(x) < self.obs_dim:  # older 18-input exports: pad fuel=1
            x = x + [1.0] * (self.obs_dim - len(x))
        return _forward(self.layers, self.activation, x[:self.obs_dim])

    def act(self, obs):
        lg = self.logits(obs)
        return max(range(len(lg)), key=lg.__getitem__)


class NetPilot:
    """A PolicyNet flying a seat in the arena: mirrored obs, greedy, the same
    fire cone and fuel budget as the learning seat (symmetric rules).

    Usable as the opponent callable of OrbitDuelEnv, or driven manually for
    evaluation from either seat.
    """

    def __init__(self, path, fire_cone_deg=15.0, fuel=True, decision_frames=4):
        self.net = PolicyNet(path)
        self.fire_cone = (math.radians(fire_cone_deg)
                          if fire_cone_deg else None)
        self.fuel = fuel
        self.tank = FuelTank()
        self.decision_frames = decision_frames

    def reset(self):
        self.tank = FuelTank()

    def __call__(self, arena, idx):
        obs = obs_from(arena, idx, fuel=self.tank.level)
        t, th, f = ACTIONS[self.net.act(obs)]
        f = gate_fire(arena, idx, f, self.fire_cone)
        if self.fuel:
            th = self.tank.gate(th, frames=self.decision_frames)
        return (t * P.TURN_RATE, th, f)
