# Experiment 01, survive: can a from-scratch DQN learn to rescue a doomed orbit?

**Question.** The [survive task](../../orbitduel/survive.py) spawns one ship on
an orbit whose periapsis is inside the black hole's capture radius. Doing nothing
dies in seconds. Can a tiny DQN (2x64 MLP, ~5,600 parameters) learn the
point-prograde-and-burn rescue from scratch, on a CPU, in minutes?

**Method.** Train [agents/dqn_survive.py](../../agents/dqn_survive.py) (Double
DQN, uniform replay, epsilon-greedy; every moving part in one file). Evaluate
the greedy policy on a fixed seed panel against two baselines: uniform-random
actions and pure coasting. Lifetimes are reported as medians because the
distribution is bimodal (rescued-to-cap vs died-early); means lie here (see
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md), Stage 1).

**Expected result.** The shipped checkpoint
([survive_dqn_v2](../../agents/models/survive_dqn_v2.json)), replayed through
the dependency-free reference forward, survives to the 60 s cap in 84% of the
50 fixed-seed episodes with a median lifetime of 60.0 s. Random's median is
2.0 s; coasting dies almost immediately by construction. A healthy retrain
lands in the same band; the grader's bar is >= 80% survival and a median at
least 10x random's.

## Run it

```
# full training run, ~600k steps, minutes on CPU (from the repo root)
python agents/dqn_survive.py

# or via the orchestrator here (train + eval table)
python experiments/01-survive/run.py
```

## Grade it

```
python -m pytest experiments/01-survive/grade.py -m grade_cheap -q
```

The grader prefers your own retrain (`runs/survive/final.json`) when one
exists, else it grades the shipped checkpoint. Evaluation goes through
[orbitduel/netpilot.py](../../orbitduel/netpilot.py) (pure Python floats), so
the shipped checkpoint's numbers are bit-identical on every platform.
