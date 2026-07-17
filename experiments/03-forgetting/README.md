# Experiment 03: curricula climb, leagues retain

**Question.** A curriculum trainer advances through opponents L1 -> L10,
always practicing against the current teacher. What happens to its play
against the teachers it left behind, and does league training (a pool of all
past opponents plus frozen self-snapshots) actually fix it?

**Method.** Evaluate shipped era checkpoints against the fixed scripted panel
(L1 / L4 / L7 / L10), each in the rule era it trained under, 12 episodes per
cell on fixed seeds. Inference goes through the pure-Python reference forward
([orbitduel/netpilot.py](../../orbitduel/netpilot.py)), so every cell of the
matrix is an exact, platform-independent integer. The checkpoints:

| checkpoint | trainer | era |
|---|---|---|
| [duel_ppo_v1](../../agents/models/duel_ppo_v1.json) | curriculum ladder ([agents/ppo_duel.py](../../agents/ppo_duel.py)) | `v1-freewalls` |
| [duel_ppo_v3_league](../../agents/models/duel_ppo_v3_league.json) | league ([agents/league_duel.py](../../agents/league_duel.py)) | `v3-rude` |
| [duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json) | league, full rules | `v6-full` |

**Expected result** (measured 2026-07-10, exact on every platform):

| | L1 | L4 | L7 | L10 |
|---|---|---|---|---|
| v1 curriculum | **6**/12 | 7/12 | 8/12 | **8**/12 |
| v3 league | 12/12 | 12/12 | 12/12 | 12/12 |
| v6 league champion | 12/12 | 12/12 | 12/12 | 10/12 |

The signature is the first row: the ladder-climber does *worse against the
easiest opponent than against the hardest* (historically 56% vs 72% over
larger panels; see [docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md),
Stage 2). It optimized against its current teacher and forgot the old ones.
Both league checkpoints hold the whole panel: the mixed opponent pool turns
"beat the current teacher" into "beat everyone I have ever met".

## Run it

```
python experiments/03-forgetting/run.py            # full matrix, all shipped checkpoints
python experiments/03-forgetting/run.py --episodes 24
```

## Grade it

```
python -m pytest experiments/03-forgetting/grade.py -m grade_cheap -q
```

The grader asserts the inversion (v1 wins fewer vs L1 than vs L10) and the
league sweep (v3/v6 near-perfect on every level).
