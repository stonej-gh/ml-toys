#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""M4: league self-play: one generalist instead of a forgetful ladder-climber.

Extends the M3 PPO (imported from ppo_duel) with an OPPONENT LEAGUE, the fix
for curriculum forgetting (same reason OpenAI Five / AlphaStar used leagues):

- Pool = all ten scripted pilot levels + frozen snapshots of the agent itself
  (a snapshot joins every SNAP_EVERY updates; the pool keeps a spread).
- Each env samples a fresh opponent per episode: SCRIPTED_MIX of the time a
  scripted level (prioritized toward the ones the agent is losing to, with a
  floor so nothing is forgotten), else a uniform pool snapshot.
- Net opponents fly ship 1 through the same 19-float observation (mirrored
  seat) and the same fire cone (symmetric rules,, no seat advantage.
- Eval is a fixed scripted panel (L1/L4/L7/L10, greedy, fixed seeds); the
  dashboard metric "win" is the panel mean, so climbing is measurable against
  a yardstick that never moves. league.csv holds the per-level detail.

Run from the repo root:   python agents/league_duel.py
"""

import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbitduel.env import OrbitDuelEnv, ACTIONS, obs_from, gate_fire, FuelTank
from orbitduel.pilot import ScriptedPilot
from orbitduel import physics as P
import agents.ppo_duel as M3
from agents.ppo_duel import ActorCritic, export_json

# --- league hyperparameters (PPO core inherits M3's) ----------------------------
TOTAL_UPDATES = 20_000       # ~4x the M3 budget
N_ENV = 8
T_ROLL = 256
SNAP_EVERY = 500             # updates between self-snapshots joining the pool
POOL_MAX = 12                # snapshots kept (evenly thinned when over)
SCRIPTED_MIX = 0.6           # P(scripted opponent); else a pool snapshot
PRIORITY_FLOOR = 0.1         # min sampling weight per scripted level
EVAL_EVERY = 100
EVAL_PANEL = (1, 4, 7, 10)
EVAL_EPISODES = 24           # per panel level
RECORD_EPISODES = 1          # replays persisted per panel level per eval
ENT_START, ENT_END = 0.02, 0.005


class NetOpponent:
    """A frozen policy flying ship 1: mirrored obs, greedy, same fire cone."""

    def __init__(self, state_dict, fire_cone):
        self.net = ActorCritic()
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.fire_cone = fire_cone
        self.tank = FuelTank()               # symmetric rules: same burst budget

    def __call__(self, arena, idx):
        with torch.no_grad():
            logits, _ = self.net(torch.as_tensor(
                obs_from(arena, idx, fuel=self.tank.level)).float().unsqueeze(0))
        t, th, f = ACTIONS[int(logits.argmax())]
        f = gate_fire(arena, idx, f, self.fire_cone)
        th = self.tank.gate(th, frames=4)    # opponent decides per 4-frame block
        return (t * P.TURN_RATE, th, f)


def make_env(opponent, seed, record=False):
    # gravity_on_lasers: the original game's shots curve under the field -
    # straight-laser training taught v4 an aim model that visibly missed there
    env = OrbitDuelEnv(opponent=opponent, hit_reward=M3.HIT_R, rules=None,
                          gravity_on_lasers=True,
                          spawn_jitter=M3.SPAWN_JITTER,
                          max_wall_seconds=M3.EPISODE_WALL_S, seed=seed,
                          record=record, pot_coef=M3.POT_COEF,
                          time_cost=M3.TIME_COST, gamma=M3.GAMMA,
                          wall_penalty=M3.WALL_PEN,
                          thrust_cost=M3.THRUST_COST)
    return env


class League:
    """Opponent pool + prioritized scripted sampling + EMA win tracking."""

    def __init__(self, fire_cone, seed=0):
        self.fire_cone = fire_cone
        self.rng = random.Random(seed)
        self.snapshots = []                      # state_dicts (frozen selves)
        self.win_ema = {lv: 0.0 for lv in range(1, 11)}   # vs scripted levels

    def add_snapshot(self, net):
        self.snapshots.append(copy.deepcopy(net.state_dict()))
        if len(self.snapshots) > POOL_MAX:
            # keep the very first (maximum diversity) + the most recent rest
            self.snapshots = (self.snapshots[:1]
                              + self.snapshots[1:][-(POOL_MAX - 1):])

    def note_result(self, level, won):
        self.win_ema[level] = 0.95 * self.win_ema[level] + 0.05 * float(won)

    def sample_opponent(self, seed):
        if not self.snapshots or self.rng.random() < SCRIPTED_MIX:
            # prioritize the levels we're losing to; keep a floor on all
            weights = [max(PRIORITY_FLOOR, 1.0 - self.win_ema[lv])
                       for lv in range(1, 11)]
            lv = self.rng.choices(range(1, 11), weights=weights)[0]
            return ScriptedPilot(lv, seed=seed), lv
        return NetOpponent(self.rng.choice(self.snapshots), self.fire_cone), 0


def evaluate_panel(net, run_dir, update):
    """Greedy vs the fixed panel; persists replays; -> (mean, {lv: win})."""
    per = {}
    rep_dir = run_dir / "replays"
    rep_dir.mkdir(parents=True, exist_ok=True)
    for lv in EVAL_PANEL:
        wins = 0
        for ep in range(EVAL_EPISODES):
            record = ep < RECORD_EPISODES
            env = make_env(ScriptedPilot(lv, seed=70_000 + ep),
                           seed=80_000 + update * 100 + lv * 10 + ep,
                           record=record)
            obs, _ = env.reset()
            total_r, outcome = 0.0, "draw"
            while True:
                with torch.no_grad():
                    logits, _ = net(torch.as_tensor(obs).float().unsqueeze(0))
                obs, r, term, trunc, _ = env.step(int(logits.argmax()))
                total_r += r
                if term or trunc:
                    if term:
                        deaths = [e for e in env.replay["events"]
                                  if e["ev"] == "death"] if env.replay else []
                        won = (deaths and deaths[-1]["ship"] == 1) \
                            or (not deaths and r > 0)
                        outcome = "win" if won else "loss"
                        wins += won
                    break
            if record and env.replay is not None:
                meta = {"update": update, "level": lv, "episode": ep,
                        "outcome": outcome, "reward": round(total_r, 2),
                        "wall_s": round(env._frames / 60.0, 1)}
                (rep_dir / f"upd_{update:05d}_L{lv}.json").write_text(
                    json.dumps({"meta": meta, **env.replay}))
                man_path = rep_dir / "manifest.json"
                man = json.loads(man_path.read_text()) if man_path.exists() else []
                man.append({"file": f"upd_{update:05d}_L{lv}.json", **meta})
                man_path.write_text(json.dumps(man))
        per[lv] = wins / EVAL_EPISODES
    return sum(per.values()) / len(per), per


def train():
    # lightweight overrides: --run-dir X --wall-pen Y (refinement runs)
    argv = sys.argv
    run_name = argv[argv.index("--run-dir") + 1] if "--run-dir" in argv else "runs/league"
    if "--wall-pen" in argv:
        M3.WALL_PEN = float(argv[argv.index("--wall-pen") + 1])
        print(f"wall penalty override: {M3.WALL_PEN}")
    run_dir = Path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    net = ActorCritic()
    # fresh start: the pre-thrust-cost checkpoints embody the power-hover
    # habit this run exists to price out - no warm start
    opt = torch.optim.Adam(net.parameters(), lr=M3.LR)
    league = League(fire_cone=None, seed=0)      # cone radians set per-env below
    league.fire_cone = make_env(ScriptedPilot(1, 0), 0).fire_cone
    league.add_snapshot(net)

    def fresh_env(i):
        opp, lv = league.sample_opponent(seed=random.randrange(1 << 30))
        env = make_env(opp, seed=random.randrange(1 << 30))
        env._league_level = lv                   # 0 = self-play snapshot
        return env

    envs = [fresh_env(i) for i in range(N_ENV)]
    obs = [e.reset()[0] for e in envs]

    mfile = open(run_dir / "metrics.csv", "w", newline="")
    metrics = csv.writer(mfile)
    metrics.writerow(["update", "level", "win", "loss", "draw", "steps_s"])
    lfile = open(run_dir / "league.csv", "w", newline="")
    lcsv = csv.writer(lfile)
    lcsv.writerow(["update"] + [f"win_L{lv}" for lv in EVAL_PANEL]
                  + ["panel_mean", "pool_size"])
    best = -1.0
    t0 = time.time()

    for update in range(1, TOTAL_UPDATES + 1):
        frac = (update - 1) / TOTAL_UPDATES
        ent_coef = ENT_START + (ENT_END - ENT_START) * frac
        for g in opt.param_groups:
            g["lr"] = M3.LR * (1 - frac)

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
                logp = dist.log_prob(a)
            O[t], A[t], LOGP[t], V[t] = ot, a, logp, v
            for i, env in enumerate(envs):
                o2, r, term, trunc, _ = env.step(int(a[i]))
                R[t, i] = r
                D[t, i] = float(term)
                if term or trunc:
                    if env._league_level > 0 and term:
                        league.note_result(env._league_level, r > 0)
                    envs[i] = fresh_env(i)       # new opponent every episode
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
        b_lp, b_ret = LOGP.reshape(-1), RET.reshape(-1)
        b_adv = ADV.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_o.shape[0]
        for _ in range(M3.EPOCHS):
            for idx in torch.randperm(n).split(M3.MINIBATCH):
                logits, v = net(b_o[idx])
                dist = torch.distributions.Categorical(logits=logits)
                ratio = (dist.log_prob(b_a[idx]) - b_lp[idx]).exp()
                s1 = ratio * b_adv[idx]
                s2 = ratio.clamp(1 - M3.CLIP, 1 + M3.CLIP) * b_adv[idx]
                loss = (-torch.min(s1, s2).mean()
                        + M3.VF_COEF * (v - b_ret[idx]).pow(2).mean()
                        - ent_coef * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), M3.GRAD_CLIP)
                opt.step()

        if update % SNAP_EVERY == 0:
            league.add_snapshot(net)

        if update % EVAL_EVERY == 0:
            mean, per = evaluate_panel(net, run_dir, update)
            sps = (update * T_ROLL * N_ENV) / (time.time() - t0)
            metrics.writerow([update, 0, f"{mean:.3f}", f"{1 - mean:.3f}", "0",
                              f"{sps:.0f}"])
            mfile.flush()
            lcsv.writerow([update] + [f"{per[lv]:.3f}" for lv in EVAL_PANEL]
                          + [f"{mean:.3f}", len(league.snapshots)])
            lfile.flush()
            print(f"upd {update:5d}  panel {mean * 100:3.0f}%  "
                  + "  ".join(f"L{lv} {per[lv] * 100:3.0f}%" for lv in EVAL_PANEL)
                  + f"  pool {len(league.snapshots)}  ema "
                  + "/".join(f"{league.win_ema[lv]:.2f}" for lv in (1, 5, 10))
                  + f"  ({sps:.0f} steps/s, {(time.time() - t0) / 60:.1f}m)",
                  flush=True)
            if mean > best:
                best = mean
                torch.save(net.state_dict(), run_dir / "best.pt")

    torch.save(net.state_dict(), run_dir / "final.pt")
    export_json(net, run_dir / "final.json")
    print(f"done in {(time.time() - t0) / 60:.1f} min; best panel {best * 100:.0f}%")


if __name__ == "__main__":
    train()
