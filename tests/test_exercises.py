# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The back of the book stays runnable. The full sample solutions take
minutes of training, so this suite runs the cheap parts for real (stage 0
end to end, stage 5's port-equality gate) and proves the rest still import
against the current tree. A walkthrough whose solution no longer runs is
worse than no walkthrough at all."""

import importlib.util
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "exercises"


def load(rel):
    spec = importlib.util.spec_from_file_location(rel.replace("/", "_"),
                                                  EX / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_stage0_gravity_runs_and_measures():
    """The stdlib solution runs end to end and still finds Kepler's 3/2."""
    out = subprocess.run([sys.executable, str(EX / "stage0-gravity/solution.py")],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "log-log slope: 1.5000" in out.stdout
    assert "net drift +0.0" in out.stdout  # the stable integrator stays put


def test_stage5_port_agrees_with_reference():
    """The hand-rolled forward matches netpilot on 100 random observations,
    and both int8 schemes still quantize; no duels, so it stays quick."""
    from orbitduel.netpilot import PolicyNet
    m = load("stage5-port/solution.py")
    layers, activation, obs_dim = m.load(m.MODEL)
    ref = PolicyNet(m.MODEL)
    rng = random.Random(0)
    for _ in range(100):
        obs = [rng.uniform(-1, 1) for _ in range(obs_dim)]
        assert m.argmax(m.forward(layers, activation, obs)) == ref.act(obs)
    for per_row in (False, True):
        q = m.quantize_int8(layers, per_row=per_row)
        obs = [rng.uniform(-1, 1) for _ in range(obs_dim)]
        assert len(m.forward_int8(q, activation, obs)) == 12


def test_training_solutions_still_import():
    """Torch solutions: importing proves their env and agent imports resolve
    against the current tree (their mains are argv-guarded and do not run)."""
    import pytest
    pytest.importorskip("torch")
    for rel in ("stage1-dqn/solution.py", "stage2-reinforce/solution.py",
                "stage4-league/solution.py", "stage6-selection/solution.py",
                "exp06-shrink/solution.py"):
        load(rel)
