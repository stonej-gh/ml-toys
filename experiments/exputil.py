# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Shared helpers for the graded experiments (experiments/NN-*/).

Every run.py / grade.py bootstraps the repo root onto sys.path and imports
this module by name; experiment-specific thresholds stay in each grade.py so
an experiment reads as one self-contained story. Two kinds of evaluation live
here:

  * pure-Python policy evals (survive_stats, duel_stats) that go through
    orbitduel/netpilot.py's reference forward: bit-identical on every
    platform, so graders can assert EXACT seeded outcomes, golden-eval style;
  * a compact seeded PPO trainer (train_short_ppo) for the live
    reward-hacking demonstration: torch, so deterministic per-platform
    (same machine, two runs -> same result) but only threshold-comparable
    across platforms. Graders that use it must leave margin.
"""

import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from orbitduel.env import OrbitDuelEnv     # noqa: E402
from orbitduel.netpilot import PolicyNet   # noqa: E402
from orbitduel.pilot import ScriptedPilot  # noqa: E402
from orbitduel.survive import SurviveEnv   # noqa: E402

MODELS = REPO / "agents" / "models"


# survive task

def survive_stats(policy, episodes, seed0=10_000, max_wall_seconds=60.0):
    """Run `policy(obs) -> action index` on fixed-seed SurviveEnv episodes.
    -> dict(median_s, mean_s, survived_frac). Pure Python end to end when the
    policy is a PolicyNet, so the numbers are platform-exact."""
    lifetimes, survived = [], 0
    for ep in range(episodes):
        env = SurviveEnv(seed=seed0 + ep, max_wall_seconds=max_wall_seconds)
        obs, _ = env.reset()
        while True:
            obs, _, term, trunc, _ = env.step(policy(obs))
            if term or trunc:
                lifetimes.append(env.wall_seconds())
                survived += trunc
                break
    return {"median_s": statistics.median(lifetimes),
            "mean_s": sum(lifetimes) / len(lifetimes),
            "survived_frac": survived / episodes}


# duel evals with grader counters

def duel_stats(model_json, level, episodes, rules, seed0=90_000,
               max_wall_seconds=120.0, **env_kwargs):
    """Evaluate an exported .json policy vs ScriptedPilot(level) under an era
    preset, through the pure-Python reference forward. -> per-episode dicts:
    outcome, wall_touches, duty (thrust fraction), wall_s. Extra kwargs go to
    OrbitDuelEnv and override the preset (ablation hook)."""
    net = PolicyNet(model_json)
    out = []
    for ep in range(episodes):
        env = OrbitDuelEnv(opponent=ScriptedPilot(level, seed=seed0 + ep),
                           rules=rules, seed=seed0 + ep,
                           max_wall_seconds=max_wall_seconds, **env_kwargs)
        obs, info = env.reset(seed=seed0 + ep)
        while True:
            obs, _, term, trunc, info = env.step(net.act(obs))
            if term or trunc:
                break
        out.append({"outcome": info.get("outcome", "draw"),
                    "wall_touches": info["wall_touches"],
                    "duty": info["thrust_frames"] / max(1, info["frames"]),
                    "longest_burn_s": info["longest_burn_s"],
                    "wall_s": info["frames"] / 60.0})
    return out


def wins(stats):
    return sum(1 for s in stats if s["outcome"] == "win")


def median_walls(stats):
    return statistics.median(s["wall_touches"] for s in stats)


# compact seeded PPO (the live reward-hacking run)

def train_short_ppo(seed, rules, level, updates, n_env=8, t_roll=256,
                    episode_wall_s=90.0, thrust_cost=0.0, log=print,
                    **env_kwargs):
    """A budgeted replica of agents/ppo_duel.py's loop: same net, same
    hyperparameters, no curriculum, no file I/O. Returns the trained
    ActorCritic. Seeded torch on one thread: two runs on one machine are
    identical; across platforms only threshold-comparable. Extra kwargs go
    to OrbitDuelEnv and beat the M3 defaults (e.g. pot_coef=0,
    fire_cone_deg=None reproduces the pre-shaping, pre-cone eras)."""
    import random
    import torch
    import agents.ppo_duel as M3
    from agents.ppo_duel import ActorCritic

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    random.seed(seed)

    def make_env(env_seed):
        kw = dict(hit_reward=M3.HIT_R, spawn_jitter=M3.SPAWN_JITTER,
                  pot_coef=M3.POT_COEF, time_cost=M3.TIME_COST,
                  gamma=M3.GAMMA, thrust_cost=thrust_cost)
        kw.update(env_kwargs)
        return OrbitDuelEnv(opponent=ScriptedPilot(level, seed=env_seed),
                            rules=rules, seed=env_seed,
                            max_wall_seconds=episode_wall_s, **kw)

    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=M3.LR)
    envs = [make_env(seed * 1_000 + i) for i in range(n_env)]
    obs = [e.reset()[0] for e in envs]

    for update in range(1, updates + 1):
        O = torch.zeros((t_roll, n_env, M3.OBS_DIM))
        A = torch.zeros((t_roll, n_env), dtype=torch.long)
        LOGP = torch.zeros((t_roll, n_env))
        R = torch.zeros((t_roll, n_env))
        D = torch.zeros((t_roll, n_env))
        V = torch.zeros((t_roll, n_env))
        for t in range(t_roll):
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
                    o2, _ = env.reset()
                obs[i] = o2
        with torch.no_grad():
            _, last_v = net(torch.as_tensor(
                [list(map(float, o)) for o in obs]).float())

        ADV = torch.zeros_like(R)
        gae = torch.zeros(n_env)
        for t in reversed(range(t_roll)):
            nxt_v = last_v if t == t_roll - 1 else V[t + 1]
            delta = R[t] + M3.GAMMA * nxt_v * (1 - D[t]) - V[t]
            gae = delta + M3.GAMMA * M3.LAMBDA * (1 - D[t]) * gae
            ADV[t] = gae
        RET = ADV + V
        b_o = O.reshape(-1, M3.OBS_DIM)
        b_a = A.reshape(-1)
        b_lp = LOGP.reshape(-1)
        b_adv = ADV.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        b_ret = RET.reshape(-1)

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
                        - M3.ENT_COEF * dist.entropy().mean())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), M3.GRAD_CLIP)
                opt.step()
        if log and update % 10 == 0:
            log(f"  update {update}/{updates}")
    return net


def export_net(net, path):
    """Export an ActorCritic to the portable .json (agents/ppo_duel format)."""
    from agents.ppo_duel import export_json
    export_json(net, path)
    return Path(path)
