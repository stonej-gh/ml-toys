# ml-toys

Small, complete machine-learning systems built for learning: an orbital-duel
reinforcement-learning environment with reference agents and every era of
their training history, and a tiny computer-vision net that learns to see those
duels from raw pixels, with a portable verification bundle. The environment grew
out of an iOS arcade game of mine, Spacewar: Orbital Duel; its physics is a twin
of the game's, and the pilots trained here fly that same game's ships. Everything here is meant to
be run, retrained, and taken apart. The stories behind these systems are written
up at [jeffrey-stone.com/research](https://jeffrey-stone.com/research/).

## Quickstart: the pilot

Requires Python 3.10+. Clone, then:

```
python -m venv .venv
.venv/bin/pip install -e ".[train,viz,dev]"
```

**1. Watch the champion beat the top scripted bot.** No compute needed; the
curated replays ship in the repo:

```
python -m http.server 8000
# open http://localhost:8000/viz/watch.html
# (defaults to the shipped champion episodes; ?dir=replays/v3-wallrider etc.
#  select other eras, ?run=demo follows a live local run)
```

**2. Re-run that match yourself.** Inference goes through a pure-Python
reference forward pass (no torch needed), so the outcome is bit-identical on
any platform:

```
python agents/duel_eval.py
```

**3. Watch an agent that learned to cheat.** The wall-riding champion from an
early era, back in the broken world it exploited, with the counters that
caught it:

```
python agents/duel_eval.py --model agents/models/duel_ppo_v1.json \
    --rules v1-freewalls --level 10
# then compare: viz/watch.html?dir=replays/v3-wallrider
```

The reward specification, and the four exploits that shaped it, are in
[docs/REWARD-SPEC.md](docs/REWARD-SPEC.md). Read that first if you read one
thing here.

**4. Train a survivor from scratch.** A 5.6k-parameter Double DQN learns to
rescue a doomed orbit in a few minutes of CPU:

```
python agents/dqn_survive.py
```

## What's in the box

| path | what |
|---|---|
| `orbitduel/` | Layer 0: stdlib physics core, `OrbitDuel-v0` Gymnasium env, era rule presets, scripted opponent ladder L1-L10, survive task, dependency-free policy forward |
| `agents/` | single-file reference trainers (DQN, PPO, league self-play) + `duel_eval.py` + era checkpoints v1-v6 in `agents/models/` |
| `viz/` | Layer 1: replay schema readers; browser viewer (`watch.html`), video renderer |
| `replays/` | curated episodes from every era, including the exploits |
| `experiments/` | graded questions: survive, reward hacking live, forgetting matrix, fuel/laser ablations, spotter port ([index](experiments/README.md)) |
| `spotter/` | the eye: seeded renderer-labeler, tiny FCN, training, export, int8 quantization |
| `deploy/` | the spotter's frozen golden bundle: numpy reference forward + seeded vectors + `verify.py` |
| `demo/` | local web demo running off the frozen bundle |
| `assets/` | recorded duel traces the renderer-labeler consumes |
| `docs/` | reward spec, learning notes, spotter design notes |
| `tests/` | analytic physics checks, bit-exact determinism, golden evals, train smoke, spotter gates |

Layering is a tested invariant: `orbitduel/` imports no viz, no trainers, no
LLM anything, and the suite proves the core trains with those directories
deleted.

## Quickstart: the spotter

The perception half: a 28k-parameter segmentation net whose training data
comes from its own renderer, so every label is free and pixel-perfect. Its
story: [jeffrey-stone.com/research/spotter](https://jeffrey-stone.com/research/spotter/).

**1. Verify the frozen bundle.** Numpy-only, no training, no torch; the
golden vectors prove the reference forward bit for bit:

```
python deploy/verify.py
```

**2. See it watch a duel.** Live overlay with a float/int8 A/B toggle,
running off the frozen bundle:

```
python demo/app.py    # http://127.0.0.1:5051
```

**3. Retrain it from scratch.** All data is generated in-repo from seeds
(about 5 minutes end to end on a laptop):

```
python tools/gen_dataset.py
python -m spotter.train_heatmap && python -m spotter.export
python -m spotter.train_dense   && python -m spotter.export --mode dense
python tools/build_bundle.py && python deploy/verify.py
```

Design notes and the five recorded lessons:
[docs/SPOTTER-DESIGN.md](docs/SPOTTER-DESIGN.md). The replay traces the
renderer consumes come from this repo's RL half; the two systems are the two
halves of one autonomy stack (a policy that decides, perception that sees).

## Experiments

Six packaged questions in [experiments/](experiments/README.md), each a
README + `run.py`, most with a seeded, thresholded grader. The flagship is
[02-reward-hacking](experiments/02-reward-hacking/README.md): train a fresh
agent in the arena's original broken ruleset and watch it discover, within a
minute of CPU, that riding the free walls beats playing the game.

```
python -m pytest -m grade_cheap -q     # run every grader
```

## Tests

```
.venv/bin/python -m pytest tests/ -q
```

Golden evaluations are exact-match, not tolerance-based: the shipped v6
champion must produce the recorded outcomes on the recorded seeds through
the reference forward pass on every platform.

## License and support

MIT ([LICENSE](LICENSE)); independent work, see [PROVENANCE.md](PROVENANCE.md).
Solo-maintained in spare time: [SUPPORT.md](SUPPORT.md).
