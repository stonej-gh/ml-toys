#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, stage 4: the simplest league that is still a league.

Two trainers, identical in every way except who they practice against:

  fixed   every episode vs the scripted L3, full stop (a one-rung curriculum)
  league  half the episodes vs L3, half vs a random frozen snapshot of the
          agent's own recent past (a pool of the last 5, one added every 40
          updates)

Both get the same budget, then both are graded across the whole scripted
ladder L1..L10, which neither trained on. The question the numbers answer:
what does variety in sparring buy, and where on the ladder does it show up?

    python exercises/stage4-league/solution.py
    python exercises/stage4-league/solution.py --updates 120   # quicker

Walkthrough with a measured run: README.md in this directory.
"""

import copy
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orbitduel.env import OrbitDuelEnv
from orbitduel.pilot import ScriptedPilot
import agents.ppo_duel as M3
from agents.ppo_duel import ActorCritic
from agents.league_duel import NetOpponent

RULES = "v4-honest"
TEACHER = 3          # the one scripted rung both trainers may practice on
UPDATES = 240
N_ENV = 8
T_ROLL = 256
SNAP_EVERY = 40
POOL_MAX = 5
EVAL_EPISODES = 12   # per rung; exercise-scale, so read spreads, not points


def make_env(opponent, seed):
    return OrbitDuelEnv(opponent=opponent, rules=RULES, seed=seed,
                        hit_reward=M3.HIT_R, spawn_jitter=M3.SPAWN_JITTER,
                        pot_coef=M3.POT_COEF, time_cost=M3.TIME_COST,
                        gamma=M3.GAMMA, max_wall_seconds=M3.EPISODE_WALL_S)


def train(league, updates, seed=7):
    """The stage-2 loop plus, when league is on, an opponent pool."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=M3.LR)
    pool = []
    fire_cone = make_env(ScriptedPilot(TEACHER, 0), 0).fire_cone

    def fresh_env():
        if league and pool and random.random() < 0.5:
            opp = NetOpponent(random.choice(pool), fire_cone)
        else:
            opp = ScriptedPilot(TEACHER, seed=random.randrange(1 << 30))
        return make_env(opp, seed=random.randrange(1 << 30))

    envs = [fresh_env() for _ in range(N_ENV)]
    obs = [e.reset()[0] for e in envs]
    for update in range(1, updates + 1):
        O = torch.zeros((T_ROLL, N_ENV, 19))
        A = torch.zeros((T_ROLL, N_ENV), dtype=torch.long)
        LOGP = torch.zeros((T_ROLL, N_ENV))
        R = torch.zeros((T_ROLL, N_ENV))
        D = torch.zeros((T_ROLL, N_ENV))
        V = torch.zeros((T_ROLL, N_ENV))
        for t in range(T_ROLL):
            ot = torch.as_tensor([list(map(float, o)) for o in obs]).float()
            with torch.no_grad():
                logits, v = net(ot)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            O[t], A[t], LOGP[t], V[t] = ot, a, lp, v
            for i, env in enumerate(envs):
                o2, r, term, trunc, _ = env.step(int(a[i]))
                R[t, i], D[t, i] = r, float(term)
                if term or trunc:
                    envs[i] = fresh_env()
                    o2, _ = envs[i].reset()
                obs[i] = o2
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(
                [list(map(float, o)) for o in obs]).float())
        ADV = torch.zeros_like(R)
        gae = torch.zeros(N_ENV)
        for t in reversed(range(T_ROLL)):
            nxt_v = last_v if t == T_ROLL - 1 else V[t + 1]
            delta = R[t] + M3.GAMMA * nxt_v * (1 - D[t]) - V[t]
            gae = delta + M3.GAMMA * M3.LAMBDA * (1 - D[t]) * gae
            ADV[t] = gae
        RET = ADV + V
        b_o, b_a = O.reshape(-1, 19), A.reshape(-1)
        b_lp, b_ret, b_adv = LOGP.reshape(-1), RET.reshape(-1), ADV.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        for _ in range(M3.EPOCHS):
            for idx in torch.randperm(b_o.shape[0]).split(M3.MINIBATCH):
                logits, v = net(b_o[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_a[idx]) - b_lp[idx]).exp()
                s1 = ratio * b_adv[idx]
                s2 = ratio.clamp(1 - M3.CLIP, 1 + M3.CLIP) * b_adv[idx]
                loss = (-torch.min(s1, s2).mean()
                        + M3.VF_COEF * (v - b_ret[idx]).pow(2).mean()
                        - 0.01 * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), M3.GRAD_CLIP)
                opt.step()
        if league and update % SNAP_EVERY == 0:
            pool.append(copy.deepcopy(net.state_dict()))
            del pool[:-POOL_MAX]
    return net


def grade(net):
    """Win count per rung across the whole ladder, fixed seeds, greedy."""
    per = []
    for lv in range(1, 11):
        wins = 0
        for ep in range(EVAL_EPISODES):
            env = make_env(ScriptedPilot(lv, seed=70_000 + ep),
                           seed=70_000 + lv * 100 + ep)
            obs, info = env.reset(seed=70_000 + lv * 100 + ep)
            while True:
                with torch.no_grad():
                    logits, _ = net(torch.as_tensor(obs).float().unsqueeze(0))
                obs, r, term, trunc, info = env.step(int(logits.argmax()))
                if term or trunc:
                    break
            wins += info.get("outcome") == "win"
        per.append(wins)
    return per


def main():
    argv = sys.argv
    updates = int(argv[argv.index("--updates") + 1]) if "--updates" in argv \
        else UPDATES
    print(f"budget {updates} updates each, teacher L{TEACHER}, rules {RULES}")
    for league in (False, True):
        name = "league(L3+selves)" if league else "fixed(L3 only)   "
        net = train(league, updates)
        per = grade(net)
        print(f"{name}  " + " ".join(f"L{i + 1}:{w:2d}" for i, w in enumerate(per))
              + f"   total {sum(per)}/{10 * EVAL_EPISODES}", flush=True)
    print(f"\n({EVAL_EPISODES} episodes a rung: read the SHAPE of each row and "
          "the totals, not any single cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
