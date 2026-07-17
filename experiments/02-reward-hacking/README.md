# Experiment 02, reward hacking: watch a fresh agent learn to cheat

**Question.** The arena's first ruleset (`rules="v1-freewalls"`) made wall
bounces free. The original project's agent noticed before its authors did and
learned to carom off the walls for propulsion and evasion (see
[docs/REWARD-SPEC.md](../../docs/REWARD-SPEC.md), "The wall rider"). Is that a
one-off, or does specification gaming reliably re-emerge if you hand a fresh
agent the same broken spec?

**Method.** Two measurements, one fossil and one live.

1. **The fossil record.** Replay the shipped free-walls-era champion
   ([duel_ppo_v1](../../agents/models/duel_ppo_v1.json)) and the modern
   champion ([duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json))
   in the same free-walls world, fixed seeds, and read the env's
   `wall_touches` counter. Pure-Python inference: the counts are exact on
   every platform.
2. **The live run.** Train a brand-new PPO agent from scratch in
   `v1-freewalls` against a scripted L3 opponent it cannot yet beat, for 90
   seeded updates (well under a minute of CPU). Then count its wall touches.

**Expected result.** The fossil: v1 touches walls in all 12 fixed-seed
episodes (137 touches total, one 91-touch pinball episode); v6 in the same
world touches a wall **zero** times in 12 episodes. The habit was learned
where it paid, and only where it paid.

The live run is the point of the experiment: with walls free and the opponent
too strong to fight, the fresh agent discovers within ~25 seconds of training
that the optimal policy is to *ride the wall* and run out the clock. Measured
medians on the reference platform: 540 wall touches per episode after 60
updates, ~4,800 after 90, ~13,800 after 120 (the ship essentially lives on
the wall). It wins nothing, loses nothing, and racks up survival reward: by
the rules as written, it is playing well. The grader's bar (median >= 100) is
deliberately far below the observed values because torch training is only
reproducible per-platform; the emergence itself showed up at every seed and
level we tried, so the margin is the experiment's robustness, not luck.

**Watch your cheater fly.** The live run records replays:

```
python experiments/02-reward-hacking/run.py       # train + eval + replays
python -m http.server                             # from the repo root
# then open http://localhost:8000/viz/watch.html?run=exp02-wallrider
```

Compare with the curated historical episodes at
`viz/watch.html?dir=replays/v3-wallrider`, and read the fix (physics first,
then reward) in [docs/REWARD-SPEC.md](../../docs/REWARD-SPEC.md). Then re-run
with `--rules v4-honest`: same seed, same budget, walls that spin you and
cost 0.8 of a win per strike. The agent stops cheating (zero wall touches)
and now just *loses*, falling into the hole in ~3 s, because at this budget
the exploit was the only thing it had learned. That is the shape of real
reward hacking: remove the exploit and the agent looks worse while the
experiment gets more true.

## Grade it

```
python -m pytest experiments/02-reward-hacking/grade.py -m grade_cheap -q
```
