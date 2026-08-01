#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Regenerate the curated era galleries in replays/ from shipped checkpoints.

Every episode is pure-Python inference (orbitduel/netpilot.py) on a baked
seed, so a gallery is reproducible from tracked inputs alone. The wall-rider
seeds were hunted for wall expression (the exploit era's signature move);
the other galleries use sequential seeds and show the era's ordinary play.

There is deliberately no drawcamper gallery: the camping runs predate this
repo, their replay archives were not lifted, and the attractor does not
reproduce under the revised physics (the same broken spec now finds the
wall exploit instead; see experiments/02-reward-hacking).

One caveat before regenerating the HISTORICAL galleries (v3/v4/v6): the
scripted opponent is defined by today's pilot.py, which flies the ported
robot, while those galleries were recorded against their own era's opponent.
Re-running them replays the same checkpoints and seeds in TODAY's arena, so
the episodes (and the v6 win-loss record) will differ from the shipped set.
That is the current-opponent truth, not a bug, but it is a different story
than the historical label tells; regenerate them deliberately or not at all.
The ported-champion gallery has no such gap: its opponent is current by
definition.

Usage, from the repo root:
    python tools/curate_era_replays.py              # regenerate all galleries
    python tools/curate_era_replays.py v3-wallrider # just one
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from orbitduel.env import OrbitDuelEnv                # noqa: E402
from orbitduel.netpilot import PolicyNet              # noqa: E402
from orbitduel.pilot import ScriptedPilot             # noqa: E402

# gallery -> (checkpoint, rules, opponent level, episode seeds)
# wall-rider seeds ranked by wall touches (177/80/79/37/31/27/25/24 at curation)
ERAS = {
    "v3-wallrider": ("duel_ppo_v1.json", "v1-freewalls", 10,
                     [110_004, 70_020, 70_018, 70_004, 70_022, 80_009, 90_012, 90_007]),
    "v4-honest":    ("duel_ppo_v4_final.json", "v4-honest", 10,
                     [60_000, 60_001, 60_002, 60_003, 60_004, 60_005, 60_006, 60_007]),
    "v6-final":     ("duel_ppo_v6_final.json", "v6-full", 10,
                     [61_000, 61_001, 61_002, 61_003, 61_004, 61_005, 61_006, 61_007]),
    "ported-champion": ("duel_ppo_ported_phone.json", "v6-full", 10,
                     [62_000, 62_001, 62_002, 62_003, 62_004, 62_005, 62_006, 62_007]),
}


def record(era, model, rules, level, seeds):
    net = PolicyNet(Path(__file__).resolve().parents[1] / "agents/models" / model)
    rep_dir = Path("replays") / era / "replays"
    rep_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for ep, seed in enumerate(seeds):
        env = OrbitDuelEnv(opponent=ScriptedPilot(level, seed=seed),
                           rules=rules, seed=seed, record=True)
        obs, info = env.reset(seed=seed)
        total_r = 0.0
        while True:
            obs, r, term, trunc, info = env.step(net.act(obs))
            total_r += r
            if term or trunc:
                break
        meta = {"update": 0, "level": level, "episode": ep,
                "outcome": info.get("outcome", "draw"),
                "reward": round(total_r, 2),
                "wall_s": round(info["frames"] / 60.0, 1),
                "walls": info["wall_touches"]}
        fn = f"seed_{seed}.json"
        (rep_dir / fn).write_text(json.dumps({"meta": meta, **env.replay}))
        manifest.append({"file": fn, **meta})
        print(f"{era} {fn}: {meta['outcome']:5s} walls {meta['walls']:3d} "
              f"{meta['wall_s']:6.1f} s")
    (rep_dir / "manifest.json").write_text(json.dumps(manifest))


def main():
    picks = sys.argv[1:] or list(ERAS)
    for era in picks:
        record(era, *ERAS[era])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
