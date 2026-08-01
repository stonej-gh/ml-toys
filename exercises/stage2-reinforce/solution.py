#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, stage 2: REINFORCE, the policy gradient with nothing on.

The simplest policy-gradient method that exists: play one episode, then nudge
every action's probability by the discounted return that followed it. No
critic, no advantage, no clipping, no batching. HOW it behaves is the lesson:
the learning signal is so noisy that progress lurches, stalls, and falls
over, which is exactly the variance problem the value head and PPO's
machinery exist to tame.

    python exercises/stage2-reinforce/solution.py                # ~3,000 episodes
    python exercises/stage2-reinforce/solution.py --episodes 500 # a quick look

Walkthrough (including the pencil-and-paper bandit derivation): README.md.
"""

import random
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orbitduel.survive import SurviveEnv, ACTIONS, OBS_DIM

EPISODES = 3_000
GAMMA = 0.99
LR = 2e-3
EVAL_EVERY = 250


def evaluate(net, episodes=50, seed=10_000):
    """Greedy policy on fixed seeds -> (median lifetime s, survival fraction)."""
    lifetimes, survived = [], 0
    for ep in range(episodes):
        env = SurviveEnv(seed=seed + ep)
        obs, _ = env.reset()
        while True:
            with torch.no_grad():
                a = int(net(torch.tensor(obs).unsqueeze(0)).argmax())
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                survived += trunc
                lifetimes.append(env._frames / 60.0)
                break
    return statistics.median(lifetimes), survived / episodes


def main():
    argv = sys.argv
    episodes = int(argv[argv.index("--episodes") + 1]) if "--episodes" in argv \
        else EPISODES
    torch.manual_seed(0)
    random.seed(0)
    net = nn.Sequential(nn.Linear(OBS_DIM, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(),
                        nn.Linear(64, len(ACTIONS)))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    evals = []

    for ep in range(1, episodes + 1):
        env = SurviveEnv(seed=ep)
        obs, _ = env.reset()
        logps, rewards = [], []
        while True:
            dist = torch.distributions.Categorical(
                logits=net(torch.tensor(obs).unsqueeze(0)))
            a = dist.sample()
            logps.append(dist.log_prob(a))
            obs, r, term, trunc, _ = env.step(int(a))
            rewards.append(r)
            if term or trunc:
                break

        # returns-to-go: each action is credited with everything that followed it
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.append(G)
        returns.reverse()

        # the whole algorithm: push up log-probability in proportion to return.
        # dividing by episode length keeps step size independent of lifetime
        loss = -sum(lp * g for lp, g in zip(logps, returns)) / len(returns)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % EVAL_EVERY == 0:
            med, surv = evaluate(net)
            evals.append((ep, med, surv))
            print(f"episode {ep:5d}  eval median {med:6.2f}s  "
                  f"survival {surv * 100:3.0f}%", flush=True)

    meds = [m for _, m, _ in evals]
    print(f"\neval medians over training: min {min(meds):.1f}s  "
          f"max {max(meds):.1f}s  last {meds[-1]:.1f}s")
    print("that spread IS the variance problem; compare the DQN's curve from "
          "stage 1, or ppo_duel's, which climb far more steadily")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
