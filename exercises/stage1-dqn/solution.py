#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, stage 1: DQN from scratch on the survive task.

A deliberately compact re-implementation of everything the survive trainer
needs: a two-layer Q-network, a ring-buffer replay memory, epsilon-greedy
exploration, a target network, and the two-line Double DQN fix. The point of
the exercise is the failure mode, so the broken variant is a flag away:

    python exercises/stage1-dqn/solution.py                    # Double DQN, lr 5e-4
    python exercises/stage1-dqn/solution.py --vanilla --lr 1e-3   # watch it collapse

Walkthrough with measured runs of both: README.md in this directory.
"""

import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orbitduel.survive import SurviveEnv, ACTIONS, OBS_DIM

STEPS = 400_000     # env decisions (each is 4 physics frames)
WARMUP = 5_000      # pure-random steps before learning starts
BATCH = 128
GAMMA = 0.99
LR = 5e-4           # the stable choice; 1e-3 is the exercise's cliff
SYNC = 4_000        # steps between target-network syncs
EVAL_EVERY = 25_000
EPS_HI, EPS_LO, EPS_DECAY_STEPS = 1.0, 0.05, 150_000


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.ReLU(),
                               nn.Linear(64, 64), nn.ReLU(),
                               nn.Linear(64, len(ACTIONS)))

    def forward(self, x):
        return self.f(x)


class Replay:
    """A ring buffer of (obs, action, reward, next_obs, ended) steps."""

    def __init__(self, cap=200_000):
        self.buf, self.cap, self.i = [], cap, 0

    def push(self, *item):
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.i] = item
        self.i = (self.i + 1) % self.cap

    def sample(self, n):
        rows = random.sample(self.buf, n)
        o, a, r, nxt, d = zip(*rows)
        return (torch.tensor(o), torch.tensor(a), torch.tensor(r),
                torch.tensor(nxt), torch.tensor(d, dtype=torch.float32))


def evaluate(q, episodes=50, seed=10_000):
    """Greedy policy on fixed seeds -> (median lifetime s, survival fraction)."""
    lifetimes, survived = [], 0
    for ep in range(episodes):
        env = SurviveEnv(seed=seed + ep)
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                a = int(q(torch.tensor(obs).unsqueeze(0)).argmax())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                survived += trunc
                lifetimes.append(env._frames / 60.0)
                break
    return statistics.median(lifetimes), survived / episodes


def main():
    argv = sys.argv
    vanilla = "--vanilla" in argv
    lr = float(argv[argv.index("--lr") + 1]) if "--lr" in argv else LR
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else STEPS
    torch.manual_seed(0)
    random.seed(0)
    q, tgt = QNet(), QNet()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf = Replay()
    print(f"{'vanilla' if vanilla else 'double'} DQN, lr {lr}, {steps} steps")

    env = SurviveEnv(seed=1)
    obs, _ = env.reset()
    history = []
    for step in range(1, steps + 1):
        eps = max(EPS_LO, EPS_HI - (EPS_HI - EPS_LO) * step / EPS_DECAY_STEPS)
        if random.random() < eps:
            a = random.randrange(len(ACTIONS))
        else:
            with torch.no_grad():
                a = int(q(torch.tensor(obs).unsqueeze(0)).argmax())
        obs2, r, term, trunc, _ = env.step(a)

        # ended=terminated only: after a timeout there IS future reward to estimate
        buf.push(obs, a, r, obs2, term)
        obs = obs2
        if term or trunc:
            obs, _ = env.reset()

        if step > WARMUP:
            o, act, rew, nxt, done = buf.sample(BATCH)
            with torch.no_grad():
                if vanilla:
                    # plain DQN target: the max runs over the target net's own
                    # noisy estimates, and a max over noise is optimistic
                    nxt_q = tgt(nxt).max(1).values
                else:
                    # Double DQN: online net PICKS the action, target net PRICES it
                    a_star = q(nxt).argmax(1, keepdim=True)
                    nxt_q = tgt(nxt).gather(1, a_star).squeeze(1)
                target = rew + GAMMA * (1 - done) * nxt_q
            pred = q(o).gather(1, act.unsqueeze(1)).squeeze(1)
            loss = nn.functional.smooth_l1_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if step % SYNC == 0:
            tgt.load_state_dict(q.state_dict())

        if step % EVAL_EVERY == 0:
            med, surv = evaluate(q)
            history.append((step, med, surv))
            print(f"step {step:7d}  eps {eps:.2f}  eval median {med:6.2f}s  "
                  f"survival {surv * 100:3.0f}%", flush=True)

    best = max(history, key=lambda h: (h[1], h[2]))
    print(f"\nbest eval: median {best[1]:.2f}s, survival {best[2] * 100:.0f}% "
          f"at step {best[0]}")
    print(f"final eval: median {history[-1][1]:.2f}s, "
          f"survival {history[-1][2] * 100:.0f}%")
    print("if best >> final, you just watched the collapse the walkthrough "
          "describes; always save the best checkpoint, not the last")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
