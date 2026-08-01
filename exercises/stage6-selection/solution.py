#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Sample solution, stage 6: measure your own selection bias.

Train one short PPO run against a fixed opponent, saving a checkpoint every
30 updates. Score every checkpoint twice: once on a SMALL noisy seed set
(the kind an in-training eval uses, and the kind "save the best model" picks
by), then again on a LARGER fresh seed set the pick never saw. The gap
between the winner's two scores is your selection bias, measured on your own
machine in a few minutes.

    python exercises/stage6-selection/solution.py
    python exercises/stage6-selection/solution.py --updates 120   # quicker

Walkthrough with a measured run: README.md in this directory.
"""

import copy
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from orbitduel.env import OrbitDuelEnv
from orbitduel.pilot import ScriptedPilot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
import exputil as X

LEVEL = 1            # a rung a short run can sometimes beat: p near 0.5 is
                     # where win-rate estimates are at their noisiest
RULES = "v4-honest"
UPDATES = 240
SNAP_EVERY = 30
PICK_EPISODES = 12   # the small, noisy sample the argmax picks by
CONFIRM_EPISODES = 48
PICK_SEED = 50_000
CONFIRM_SEED = 60_000


def win_rate(net, episodes, seed0):
    """Greedy eval vs the fixed scripted rung on fixed seeds -> fraction won."""
    wins = 0
    for ep in range(episodes):
        env = OrbitDuelEnv(opponent=ScriptedPilot(LEVEL, seed=seed0 + ep),
                          rules=RULES, seed=seed0 + ep, max_wall_seconds=45.0)
        obs, info = env.reset(seed=seed0 + ep)
        while True:
            with torch.no_grad():
                logits, _ = net(torch.as_tensor(obs).float().unsqueeze(0))
            obs, r, term, trunc, info = env.step(int(logits.argmax()))
            if term or trunc:
                break
        wins += info.get("outcome") == "win"
    return wins / episodes


def main():
    argv = sys.argv
    updates = int(argv[argv.index("--updates") + 1]) if "--updates" in argv \
        else UPDATES

    # exputil.train_short_ppo runs a fixed budget and returns the net, with no
    # mid-run snapshot hook. Training is seeded and deterministic per machine,
    # though, so re-running the same seed to longer horizons yields snapshots
    # of ONE trajectory: the 30-update run is the exact prefix of the 60
    snaps = []
    for upd in range(SNAP_EVERY, updates + 1, SNAP_EVERY):
        net = X.train_short_ppo(seed=7, rules=RULES, level=LEVEL, updates=upd,
                                log=lambda *_: None)
        snaps.append((f"upd{upd}", copy.deepcopy(net.state_dict())))
        print(f"trained to {upd} updates", flush=True)

    from agents.ppo_duel import ActorCritic
    rows = []
    for tag, sd in snaps:
        net = ActorCritic()
        net.load_state_dict(sd)
        net.eval()
        picked = win_rate(net, PICK_EPISODES, PICK_SEED)
        confirmed = win_rate(net, CONFIRM_EPISODES, CONFIRM_SEED)
        rows.append((tag, picked, confirmed))
        print(f"  {tag:>7s}  small noisy sample {picked * 100:5.1f}%   "
              f"fresh larger sample {confirmed * 100:5.1f}%", flush=True)

    by_pick = max(rows, key=lambda r: r[1])
    by_confirm = max(rows, key=lambda r: r[2])
    print(f"\nargmax of the noisy sample picks {by_pick[0]}: "
          f"scored {by_pick[1] * 100:.1f}%, really {by_pick[2] * 100:.1f}% "
          f"(gap {100 * (by_pick[1] - by_pick[2]):+.1f} points)")
    print(f"the fresh sample's own best is {by_confirm[0]} "
          f"at {by_confirm[2] * 100:.1f}%")
    if by_pick[0] != by_confirm[0]:
        print("the two picks DISAGREE: the noisy argmax shipped the wrong model")
    print("\nrule: select on one sample, report on another; the trainer that "
          "shipped this repo's champions does exactly that (league_duel.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
