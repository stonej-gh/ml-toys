#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Evaluate an exported policy (.json) against the scripted ladder, and write
browser-viewable replays. Framework-free: inference runs through the pure-
Python reference forward (orbitduel/netpilot.py), so results are bit-identical
on every platform - this script is also the golden-eval anchor in tests/.

Examples, from the repo root:
    python agents/duel_eval.py                     # current champion vs L10
    python agents/duel_eval.py --level 4 --episodes 24
    python agents/duel_eval.py --model agents/models/duel_ppo_v6_final.json
    python agents/duel_eval.py --model agents/models/duel_ppo_v3_league.json \
        --rules v3-rude --out runs/rude

Replays land in <out>/replays/ with a manifest.json; serve the repo root
(python -m http.server) and open viz/watch.html?run=<out name>.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orbitduel.env import OrbitDuelEnv
from orbitduel.netpilot import PolicyNet
from orbitduel.pilot import ScriptedPilot
from orbitduel import physics as P

# The default is the CURRENT champion, matched to the field profile, because
# the README tells a first-time reader to run this with no arguments right after
# promising they will watch the champion take on the top scripted bot. It used
# to default to v6, which was the champion of the v6 ERA and is not the champion
# of this ladder: against the ported L10 it wins about 1 game in 10, so the
# quickstart's first command printed nine straight losses. Era checkpoints are
# still here and still the story LEARNING-NOTES tells; they are just not the
# thing to hand someone as "the champion" any more.
DEFAULT_MODEL = (Path(__file__).parent / "models" /
                 f"duel_ppo_ported_{P.FIELD_NAME}.json")


def run(model, level, episodes, rules, out, record, seed0):
    net = PolicyNet(model)
    out_dir = Path(out)
    rep_dir = out_dir / "replays"
    rep_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    outcomes = []
    for ep in range(episodes):
        env = OrbitDuelEnv(opponent=ScriptedPilot(level, seed=seed0 + ep),
                           rules=rules, seed=seed0 + ep, record=ep < record)
        obs, info = env.reset(seed=seed0 + ep)
        total_r = 0.0
        while True:
            obs, r, term, trunc, info = env.step(net.act(obs))
            total_r += r
            if term or trunc:
                break
        outcome = info.get("outcome", "draw")
        outcomes.append(outcome)
        print(f"ep {ep:2d} seed {seed0 + ep}  {outcome:5s}"
              f"  cause={info.get('cause', '-'):9s}"
              f"  walls={info['wall_touches']:2d}"
              f"  duty={info['thrust_frames'] / max(1, info['frames']):.2f}"
              f"  {info['frames'] / 60.0:6.1f} wall-s")
        if env.replay is not None:
            meta = {"update": 0, "level": level, "episode": ep,
                    "outcome": outcome, "reward": round(total_r, 2),
                    "wall_s": round(info["frames"] / 60.0, 1),
                    "walls": info["wall_touches"]}
            fn = f"ep{ep:03d}.json"
            (rep_dir / fn).write_text(json.dumps({"meta": meta, **env.replay}))
            manifest.append({"file": fn, **meta})
    if manifest:
        (rep_dir / "manifest.json").write_text(json.dumps(manifest))
    w = outcomes.count("win")
    l = outcomes.count("loss")
    d = outcomes.count("draw")
    print(f"\n{Path(model).name} vs L{level} ({rules}):  "
          f"{w} wins / {l} losses / {d} draws over {episodes} episodes")
    return outcomes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--episodes", type=int, default=9)
    ap.add_argument("--rules", default="v6-full")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--record", type=int, default=3,
                    help="episodes to persist as replays")
    ap.add_argument("--seed0", type=int, default=90_000)
    args = ap.parse_args()
    run(args.model, args.level, args.episodes, args.rules, args.out,
        args.record, args.seed0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
