# Experiment 03: curricula climb, leagues retain

*Level: intermediate. In plain English: an agent that always practices against
its current rival gets rusty against old ones; practicing against everyone it
has ever met fixes that, and this experiment measures both in one matrix.*

**Question.** A curriculum trainer advances through opponents L1 -> L10,
always practicing against the current teacher. What happens to its play
against the teachers it left behind, and does league training (a pool of all
past opponents plus frozen self-snapshots) actually fix it?

**Method.** Evaluate shipped era checkpoints against the fixed scripted panel
(L1 / L4 / L7 / L10), each in the rule era it trained under, 12 episodes per
cell on fixed seeds. Inference goes through the pure-Python reference forward
([orbitduel/netpilot.py](../../orbitduel/netpilot.py)), so every cell is a
deterministic integer: two runs on one machine match byte for byte. A few
knife-edge episodes can flip by a single game across environments
(interpreter and math-library differences), which is why the grader asserts
margins rather than the exact matrix. The checkpoints:

| checkpoint | trainer | era |
|---|---|---|
| [duel_ppo_v1](../../agents/models/duel_ppo_v1.json) | curriculum ladder ([agents/ppo_duel.py](../../agents/ppo_duel.py)) | `v1-freewalls` |
| [duel_ppo_v3_league](../../agents/models/duel_ppo_v3_league.json) | league ([agents/league_duel.py](../../agents/league_duel.py)) | `v3-rude` |
| [duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json) | league, full rules | `v6-full` |

**Expected result** (measured 2026-07-30 under the current physics):

| | L1 | L4 | L7 | L10 |
|---|---|---|---|---|
| v1 curriculum | **2**/12 | 4/12 | 9/12 | **9**/12 |
| v3 league | 12/12 | 12/12 | 12/12 | 12/12 |
| v6 league champion | 12/12 | 11/12 | 8/12 | 6/12 |

The signature is the first row: the ladder-climber does *worse against the
easiest opponent than against the hardest* (historically 56% vs 72% over
larger panels; see [docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md),
Stage 2). It optimized against its current teacher and forgot the old ones.
The v3 league row holds the entire panel: the mixed opponent pool turns
"beat the current teacher" into "beat everyone I have ever met". The v6
full-rules champion holds the low ladder but tapers against the top rungs:
it was trained before the 2026-07 physics revision and inherits that gap, so
the retention lesson rests on the era-matched v3 row. `run.py` prints all
six shipped checkpoints; the three rows above are the story.

## Run it

```
python experiments/03-forgetting/run.py            # full matrix, all shipped checkpoints
python experiments/03-forgetting/run.py --episodes 24
python experiments/03-forgetting/run.py --model runs/duel/final.json --rules v6-full
```

The third form is the invitation: train your own checkpoint (any trainer in
[agents/](../../agents)), export it, and your row joins the matrix next to
the shipped ones.

## Grade it

```
python -m pytest experiments/03-forgetting/grade.py -m grade_cheap -q
```

The grader asserts the inversion (v1 wins fewer vs L1 than vs L10), the
league floor (v3 at least 11/12 at every level), and v6's monotone taper
with a floor of 5/12 vs L10.
