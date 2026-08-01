# Stage 6 exercise: measure your own selection bias

The back of the book for the stage 6 exercise in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md): take a training run
with a "save the best model" habit, score its checkpoints twice, and put a
number on how much the picking sample flattered the pick. The sample
solution is [solution.py](solution.py); one run takes a few minutes.

```
python exercises/stage6-selection/solution.py
python exercises/stage6-selection/solution.py --updates 120   # quicker
```

## The walk

**1. Make some checkpoints.** The solution trains one short PPO run against
the scripted L1 and keeps a snapshot every 30 updates, eight in all.
(Training is seeded and deterministic per machine, so it snapshots by
re-running the same seed to longer horizons; a callback would do the same
job in a trainer that offered one.)

**2. Score everything twice.** Every checkpoint plays the same rung on two
seed sets it has never trained on: a SMALL one (12 episodes, the size of a
typical in-training eval, and the sample an argmax would pick by) and a
LARGER FRESH one (48 episodes) that has no say in the pick.

**3. What one run measured.** On the reference platform (world pin
`30f439b240a2`):

| checkpoint | small noisy sample | fresh larger sample |
|---|---|---|
| upd150 | 16.7% | **39.6%** |
| upd180 | 41.7% | 35.4% |
| upd210 | **50.0%** | 29.2% |
| upd240 | 8.3% | 27.1% |

The argmax of the noisy sample ships `upd210`, believing it is a 50%
pilot. Re-scored on seeds that had no vote, it is a 29% pilot: **20.8
points of flattery**. Worse, the pick itself is wrong: the strongest
checkpoint on fresh data is `upd150`, which the noisy sample had ranked
near the bottom at 16.7%. One column, read alone, chose the wrong model
and overstated it, and this is a small, tame instance: your run will
differ (that is rather the point), and the direction survives.

**4. Why it must happen.** Each cell in the noisy column is a coin-flip
estimate with a standard error around 14 points at n=12. Take the maximum
of eight such numbers and you have selected, in roughly equal parts, for
skill and for luck; the luck does not come along to deployment. The same
arithmetic at production scale is measured in the notes' Stage 6: over six
real training runs, 49 of 54 candidates dropped on re-scoring, by 6.1
points on average, and four runs of six would have shipped the wrong
model. Your five-minute version and the lab's are the same effect at two
zoom levels.

**5. The fix costs one evaluation.** Select on one sample, report on
another. [agents/league_duel.py](../../agents/league_duel.py) does it in
`confirm()`: training evals only nominate, a fresh-seed pass over the full
ladder decides, and both numbers land in `selection.csv` so the gap stays
visible. Any "keep the best" line you have ever written can grow the same
second pass for the cost of one more eval.

## Where to go from here

Rerun the solution with `PICK_EPISODES` raised to 48: watch the flattery
shrink but not vanish (the bias falls with sample size; it never reaches
zero while you select and report on the same data). Then find the
`save best` line in any training script you own and give it a second
sample. The gap you measure is the number your last results table
overstated by.
