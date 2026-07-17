#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""M2: hand-rolled DQN on the T0 SURVIVE task (one readable file, CleanRL-style).

The whole algorithm is here on purpose: no framework, every moving part
visible: a tiny 2x64 MLP Q-network (small enough for a microcontroller),
a uniform replay buffer, epsilon-greedy exploration, a periodically
synced target network, and Huber TD loss.

Run from the repo root:
    python agents/dqn_survive.py                 # train (~600k steps)
    python agents/dqn_survive.py --eval PATH.pt  # evaluate a checkpoint

Artifacts land in runs/survive/ (gitignored): metrics.csv, best.pt, final.pt,
and final.json (plain weight arrays, the portable deployment format).
Evaluation compares the greedy policy against the random and coast baselines
on fixed seeds; the M2 gate is median lifetime >= 10x random's median AND
survival-to-cap >= 80%.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbitduel.survive import SurviveEnv, ACTIONS, OBS_DIM

# --- hyperparameters -----------------------------------------------------------
TOTAL_STEPS = 600_000        # env steps (each = 4 physics frames)
BUFFER_SIZE = 200_000
WARMUP_STEPS = 5_000
BATCH = 128
GAMMA = 0.99
LR = 5e-4                    # 1e-3 learned fast then collapsed (catastrophic forgetting)
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 150_000
TARGET_SYNC = 4_000
EVAL_EVERY = 25_000
EVAL_EPISODES = 50
HIDDEN = 64
DEVICE = "cpu"               # tiny net: CPU beats MPS's per-op overhead


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, len(ACTIONS)))

    def forward(self, x):
        return self.net(x)


class Replay:
    """Uniform ring buffer on preallocated tensors."""

    def __init__(self, size):
        self.size = size
        self.obs = torch.zeros((size, OBS_DIM))
        self.act = torch.zeros(size, dtype=torch.long)
        self.rew = torch.zeros(size)
        self.nxt = torch.zeros((size, OBS_DIM))
        self.done = torch.zeros(size)
        self.n = 0
        self.i = 0

    def push(self, o, a, r, o2, d):
        self.obs[self.i] = torch.as_tensor(o)
        self.act[self.i] = a
        self.rew[self.i] = r
        self.nxt[self.i] = torch.as_tensor(o2)
        self.done[self.i] = float(d)
        self.i = (self.i + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = torch.randint(0, self.n, (batch,))
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.nxt[idx], self.done[idx])


def greedy_action(q, obs):
    with torch.no_grad():
        return int(q(torch.as_tensor(obs).float().unsqueeze(0)).argmax())


def evaluate(policy, episodes=EVAL_EPISODES, seed=10_000):
    """-> (median, mean, survived_frac) lifetime in wall seconds on fixed seeds."""
    lifetimes, survived = [], 0
    for ep in range(episodes):
        env = SurviveEnv(seed=seed + ep)
        obs, _ = env.reset()
        while True:
            obs, _, term, trunc, _ = env.step(policy(obs))
            if term or trunc:
                lifetimes.append(env.wall_seconds())
                survived += trunc
                break
    lifetimes.sort()
    med = lifetimes[len(lifetimes) // 2]
    return med, sum(lifetimes) / len(lifetimes), survived / episodes


def export_json(q, path):
    """Plain weight arrays (row-major), the format orbitduel/netpilot.py reads."""
    layers = []
    for m in q.net:
        if isinstance(m, nn.Linear):
            layers.append({"w": m.weight.tolist(), "b": m.bias.tolist()})
    meta = {"obs_dim": OBS_DIM, "actions": [list(a) for a in ACTIONS],
            "hidden": HIDDEN, "activation": "relu", "layers": layers}
    Path(path).write_text(json.dumps(meta))


def train(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    q, tgt = QNet(), QNet()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR)
    buf = Replay(BUFFER_SIZE)

    env = SurviveEnv(seed=1)
    obs, _ = env.reset()
    episode_lifetimes = []
    best = (-1.0, -1.0)          # (eval median, survival rate)
    t0 = time.time()

    metrics = csv.writer(open(run_dir / "metrics.csv", "w", newline=""))
    metrics.writerow(["step", "eps", "recent_mean_life",
                      "eval_median", "eval_mean", "eval_survived"])

    for step in range(1, TOTAL_STEPS + 1):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END)
                  * step / EPS_DECAY_STEPS)
        a = random.randrange(len(ACTIONS)) if random.random() < eps \
            else greedy_action(q, obs)
        obs2, r, term, trunc, _ = env.step(a)
        # bootstrapping cut only on real deaths - truncation is not a terminal state
        buf.push(obs, a, r, obs2, term)
        obs = obs2
        if term or trunc:
            episode_lifetimes.append(env.wall_seconds())
            obs, _ = env.reset()

        if step > WARMUP_STEPS:
            o, act, rew, nxt, done = buf.sample(BATCH)
            with torch.no_grad():
                # Double DQN: online net picks the action, target net prices it -
                # curbs the max-operator overestimation that drove the v1 collapse
                a_star = q(nxt).argmax(1, keepdim=True)
                target = rew + GAMMA * (1 - done) \
                    * tgt(nxt).gather(1, a_star).squeeze(1)
            pred = q(o).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % TARGET_SYNC == 0:
                tgt.load_state_dict(q.state_dict())

        if step % EVAL_EVERY == 0:
            med, mean, surv = evaluate(lambda o: greedy_action(q, o))
            recent = sum(episode_lifetimes[-50:]) / max(1, len(episode_lifetimes[-50:]))
            metrics.writerow([step, f"{eps:.3f}", f"{recent:.2f}",
                              f"{med:.2f}", f"{mean:.2f}", f"{surv:.2f}"])
            print(f"step {step:>7}  eps {eps:.2f}  recent-life {recent:6.2f}s  "
                  f"eval median {med:6.2f}s mean {mean:6.2f}s "
                  f"survived {surv * 100:3.0f}%  ({time.time() - t0:.0f}s)",
                  flush=True)
            if (med, surv) > best:
                best = (med, surv)
                torch.save(q.state_dict(), run_dir / "best.pt")

    torch.save(q.state_dict(), run_dir / "final.pt")
    export_json(q, run_dir / "final.json")
    print(f"done in {time.time() - t0:.0f}s; best eval median {best[0]:.2f}s survival {best[1] * 100:.0f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", metavar="CKPT", help="evaluate a checkpoint vs baselines")
    ap.add_argument("--run-dir", default="runs/survive")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    if args.eval:
        q = QNet()
        q.load_state_dict(torch.load(args.eval))
        q.eval()
        rng = random.Random(2)
        rows = [("learned (greedy)", lambda o: greedy_action(q, o)),
                ("random", lambda o: rng.randrange(len(ACTIONS))),
                ("coast", lambda o: 2)]
        results = {}
        for name, pol in rows:
            med, mean, surv = evaluate(pol, episodes=200)
            results[name] = (med, mean, surv)
            print(f"{name:>16}: median {med:6.2f}s  mean {mean:6.2f}s  "
                  f"survived-to-cap {surv * 100:3.0f}%")
        lm, _, ls = results["learned (greedy)"]
        rm = results["random"][0]
        ok = lm >= 10 * rm and ls >= 0.80
        print(f"\nM2 GATE ({lm:.2f}s vs 10x random median {rm:.2f}s, "
              f"survival {ls * 100:.0f}% vs 80%):", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    train(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
