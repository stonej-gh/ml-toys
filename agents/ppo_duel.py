#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""M3: hand-rolled PPO on the duel task, climbing the scripted-pilot ladder.

One readable file, same spirit as dqn_survive.py: a tiny shared-trunk
actor-critic (2x64, small enough for a microcontroller), GAE(lambda), the clipped surrogate,
entropy bonus: every moving part visible.

Curriculum: train vs ScriptedPilot(level) starting at L1; when the greedy
policy's eval win rate crosses ADVANCE_WIN, move up a level (fresh optimizer
momentum, same weights). Rewards: +/-1 on win/loss plus +/-HIT_R per hit
dealt/taken (the duel itself scores per hit, so this is score-shaped, not
hand-authored strategy).

The experiment is DOCUMENTED as it runs (see viz/):
  runs/duel/metrics.csv                 one row per eval
  runs/duel/replays/upd_*.json          every eval episode's full replay
  runs/duel/replays/manifest.json       index the viewer + video renderer read
  runs/duel/level_L*.pt|.json, final.*  checkpoints (json = the portable
                                        format orbitduel/netpilot.py replays)

Run from the repo root:   python agents/ppo_duel.py
"""

import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbitduel.env import OrbitDuelEnv, ACTIONS
from orbitduel.pilot import ScriptedPilot

OBS_DIM, N_ACT = 19, len(ACTIONS)

# hyperparameters
N_ENV = 8
T_ROLL = 256  # steps per env per update -> 2048-step batches
TOTAL_UPDATES = 5000
LR = 3e-4
GAMMA = 0.995
LAMBDA = 0.95
CLIP = 0.2
EPOCHS = 4
MINIBATCH = 512
ENT_COEF = 0.02
VF_COEF = 0.5
GRAD_CLIP = 0.5
HIT_R = 0.3          # shaping per hit dealt/taken (game-scored anyway)
EPISODE_WALL_S = 90.0
SPAWN_JITTER = 0.08
EVAL_EVERY = 10      # updates between evals
EVAL_EPISODES = 40   # The curriculum advances on the MAXIMUM of many noisy
                     # evals, and the max of noisy estimates overshoots the
                     # true rate, so a small sample promotes on luck: at 24 a
                     # genuinely 45% policy threw 15/24 often enough to climb.
                     # More episodes shrink the noise the maximum feeds on
RECORD_EPISODES = 3  # replays persisted per eval
ADVANCE_WIN = 0.60   # win rate to climb a curriculum level
START_LEVEL, MAX_LEVEL = 1, 10
HIDDEN = 64

# Run-time overrides, same lightweight argv style as league_duel.py. The
# defaults reproduce the modern-era run. An era preset plus its prices is how
# experiment 03's curriculum checkpoint is reproduced, which was not previously
# expressible from this file at all:
#   python agents/ppo_duel.py --rules v1-freewalls --wall-pen 0 --thrust-cost 0
RULES = None         # None = modern rules; else an env.RULE_PRESETS name
SEED = 0
RUN_DIR = "runs/duel"


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
                                   nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.pi = nn.Linear(HIDDEN, N_ACT)
        self.v = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        z = self.trunk(x)
        return self.pi(z), self.v(z).squeeze(-1)


POT_COEF = 0.5      # potential shaping: approach is worth up to ~0.5
TIME_COST = 0.004   # per wall-second: a 90 s draw costs ~0.36
WALL_PEN = 0.8      # per wall strike, ~0.8x a death: out-orbit, don't bank
THRUST_COST = 0.05  # per wall-second lit: ~6 s of deliberate burns costs a
                    # quarter-win; a 20 s power-hover costs a death. Calibrated
                    # from the scripted bots' fuel governor (burn:regen 19:1, ~5% duty)


def make_env(level, seed, record=False):
    return OrbitDuelEnv(opponent=ScriptedPilot(level, seed=seed), rules=RULES,
                           hit_reward=HIT_R, spawn_jitter=SPAWN_JITTER,
                           max_wall_seconds=EPISODE_WALL_S, seed=seed,
                           record=record, pot_coef=POT_COEF,
                           time_cost=TIME_COST, gamma=GAMMA,
                           wall_penalty=WALL_PEN, thrust_cost=THRUST_COST)


def evaluate(net, level, run_dir, update, episodes=EVAL_EPISODES,
             seed_base=50_000, keep_replays=True):
    """Greedy eval; persists the first RECORD_EPISODES replays. -> (win, loss, draw)

    seed_base is an argument so the promotion check can draw a second,
    independent sample instead of re-reading the one that proposed the climb."""
    wins = losses = 0
    for ep in range(episodes):
        record = keep_replays and ep < RECORD_EPISODES
        env = make_env(level, seed=seed_base + update * 100 + ep, record=record)
        obs, _ = env.reset()
        total_r, outcome = 0.0, "draw"
        while True:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(obs).float().unsqueeze(0))
            obs, r, term, trunc, _ = env.step(int(logits.argmax()))
            total_r += r
            if term or trunc:
                if term:
                    died = [e for e in (env.replay or {"events": []})["events"]
                            if e["ev"] == "death"]

                    # without a replay, infer from reward sign
                    won = (died and died[-1]["ship"] == 1) or (not died and r > 0)
                    outcome = "win" if won else "loss"
                    wins += won
                    losses += not won
                break
        if record and env.replay is not None:
            rep_dir = run_dir / "replays"
            rep_dir.mkdir(parents=True, exist_ok=True)
            meta = {"update": update, "level": level, "episode": ep,
                    "outcome": outcome, "reward": round(total_r, 2),
                    "wall_s": round(env._frames / 60.0, 1)}
            (rep_dir / f"upd_{update:05d}_ep{ep}.json").write_text(
                json.dumps({"meta": meta, **env.replay}))
            man_path = rep_dir / "manifest.json"
            man = json.loads(man_path.read_text()) if man_path.exists() else []
            man.append({"file": f"upd_{update:05d}_ep{ep}.json", **meta})
            man_path.write_text(json.dumps(man))
    return wins / episodes, losses / episodes, 1 - (wins + losses) / episodes


def export_json(net, path):
    layers = []
    for m in list(net.trunk) + [net.pi]:
        if isinstance(m, nn.Linear):
            layers.append({"w": m.weight.tolist(), "b": m.bias.tolist()})
    Path(path).write_text(json.dumps(
        {"obs_dim": OBS_DIM, "actions": [list(a) for a in ACTIONS],
         "hidden": HIDDEN, "activation": "tanh", "head": "argmax",
         "layers": layers}))


def train():
    run_dir = Path(RUN_DIR)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    level = START_LEVEL

    envs = [make_env(level, seed=i) for i in range(N_ENV)]
    obs = [e.reset()[0] for e in envs]

    best_win = -1.0
    mfile = open(run_dir / "metrics.csv", "w", newline="")
    metrics = csv.writer(mfile)
    metrics.writerow(["update", "level", "win", "loss", "draw", "steps_s"])
    mfile.flush()
    t0 = time.time()

    for update in range(1, TOTAL_UPDATES + 1):
        # rollout
        O = torch.zeros((T_ROLL, N_ENV, OBS_DIM))
        A = torch.zeros((T_ROLL, N_ENV), dtype=torch.long)
        LOGP = torch.zeros((T_ROLL, N_ENV))
        R = torch.zeros((T_ROLL, N_ENV))
        D = torch.zeros((T_ROLL, N_ENV))
        V = torch.zeros((T_ROLL, N_ENV))
        for t in range(T_ROLL):
            ot = torch.as_tensor(
                [list(map(float, o)) for o in obs]).float()
            with torch.no_grad():
                logits, v = net(ot)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            O[t], A[t], LOGP[t], V[t] = ot, a, logp, v
            for i, env in enumerate(envs):
                o2, r, term, trunc, _ = env.step(int(a[i]))
                R[t, i] = r
                D[t, i] = float(term)  # truncation is not terminal
                if term or trunc:
                    o2, _ = env.reset()
                obs[i] = o2
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(
                [list(map(float, o)) for o in obs]).float())

        # GAE
        ADV = torch.zeros_like(R)
        gae = torch.zeros(N_ENV)
        for t in reversed(range(T_ROLL)):
            nxt_v = last_v if t == T_ROLL - 1 else V[t + 1]
            delta = R[t] + GAMMA * nxt_v * (1 - D[t]) - V[t]
            gae = delta + GAMMA * LAMBDA * (1 - D[t]) * gae
            ADV[t] = gae
        RET = ADV + V
        b_o = O.reshape(-1, OBS_DIM)
        b_a = A.reshape(-1)
        b_lp = LOGP.reshape(-1)
        b_adv = ADV.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = RET.reshape(-1)

        # clipped-surrogate epochs (lr annealed linearly to 0)
        for g in opt.param_groups:
            g["lr"] = LR * (1 - (update - 1) / TOTAL_UPDATES)
        n = b_o.shape[0]
        for _ in range(EPOCHS):
            for idx in torch.randperm(n).split(MINIBATCH):
                logits, v = net(b_o[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_a[idx]) - b_lp[idx]).exp()
                s1 = ratio * b_adv[idx]
                s2 = ratio.clamp(1 - CLIP, 1 + CLIP) * b_adv[idx]
                loss = (-torch.min(s1, s2).mean()
                        + VF_COEF * (v - b_ret[idx]).pow(2).mean()
                        - ENT_COEF * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                opt.step()

        # eval / curriculum
        if update % EVAL_EVERY == 0:
            win, lose, draw = evaluate(net, level, run_dir, update)
            sps = (update * T_ROLL * N_ENV) / (time.time() - t0)
            metrics.writerow([update, level, f"{win:.3f}", f"{lose:.3f}",
                              f"{draw:.3f}", f"{sps:.0f}"])
            mfile.flush()
            print(f"upd {update:4d}  L{level}  win {win * 100:3.0f}%  "
                  f"loss {lose * 100:3.0f}%  draw {draw * 100:3.0f}%  "
                  f"({sps:.0f} steps/s, {time.time() - t0:.0f}s)", flush=True)
            if win > best_win:  # M2's lesson: always keep the best
                best_win = win
                torch.save(net.state_dict(), run_dir / f"best_L{level}.pt")

            # A crossing has to survive a second look. The ladder evaluates
            # every EVAL_EVERY updates, so a rung gets ~100 chances to throw a
            # lucky sample: at 40 episodes a genuinely 45% policy clears 60% on
            # about 3.5% of evals, which is near certain to happen at least
            # once per rung. Demanding that an INDEPENDENT sample clear the bar
            # too takes that from ~97% to ~11%. The confirmation keeps no
            # replays and does not touch best_win, so it costs one eval and
            # changes nothing else.
            promote = False
            if win >= ADVANCE_WIN:
                conf, _, _ = evaluate(net, level, run_dir, update,
                                      seed_base=600_000, keep_replays=False)
                promote = conf >= ADVANCE_WIN
                if not promote:
                    print(f"  L{level} crossing not confirmed "
                          f"({win * 100:.0f}% then {conf * 100:.0f}%), staying",
                          flush=True)
            if promote:
                torch.save(net.state_dict(), run_dir / f"level_L{level}.pt")
                export_json(net, run_dir / f"level_L{level}.json")
                if level >= MAX_LEVEL:
                    print("curriculum complete at L10")
                    break
                level += 1
                best_win = -1.0
                print(f"=== ADVANCE to L{level} ===", flush=True)
                envs = [make_env(level, seed=1000 * level + i) for i in range(N_ENV)]
                obs = [e.reset()[0] for e in envs]
                opt = torch.optim.Adam(net.parameters(), lr=LR)

    torch.save(net.state_dict(), run_dir / "final.pt")
    export_json(net, run_dir / "final.json")
    print(f"done in {(time.time() - t0) / 60:.1f} min; reached L{level}")


if __name__ == "__main__":
    _argv = sys.argv

    def _opt(flag, cast, default):
        return cast(_argv[_argv.index(flag) + 1]) if flag in _argv else default

    RULES = _opt("--rules", str, RULES)
    SEED = _opt("--seed", int, SEED)
    RUN_DIR = _opt("--run-dir", str, RUN_DIR)
    WALL_PEN = _opt("--wall-pen", float, WALL_PEN)
    THRUST_COST = _opt("--thrust-cost", float, THRUST_COST)
    TOTAL_UPDATES = _opt("--updates", int, TOTAL_UPDATES)
    print(f"rules={RULES} seed={SEED} wall_pen={WALL_PEN} "
          f"thrust_cost={THRUST_COST} -> {RUN_DIR}", flush=True)
    train()
