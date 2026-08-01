# Stage 4 exercise: the simplest league that is still a league

The back of the book for the stage 4 exercise in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md): take a plain PPO
trainer and add the smallest possible opponent pool, then measure what the
variety buys across the whole ladder. The sample solution is
[solution.py](solution.py).

```
python exercises/stage4-league/solution.py                # both arms + grades
python exercises/stage4-league/solution.py --updates 120  # quicker
```

## The walk

**1. Two trainers, one difference.** Both arms run the same PPO loop, the
same 240-update budget, the same rules (`v4-honest`), and the same scripted
teacher (L3). The only difference: the `fixed` arm plays L3 every single
episode, while the `league` arm plays L3 half the time and a random frozen
snapshot of its own recent self the other half (a pool of the last 5,
one added every 40 updates). The pool costs about ten lines.

**2. Grade across the whole ladder, not the teacher.** Both nets then face
all ten scripted rungs, 12 episodes each, none of which they trained
against except L3. Measured on the reference platform:

| arm | L1..L10 wins (of 12 each) | total |
|---|---|---|
| fixed, L3 only | 4 3 1 0 1 0 0 0 0 0 | **9/120** |
| league, L3 + selves | 7 7 7 3 3 0 3 3 1 1 | **35/120** |

**3. Read the shape.** The fixed arm's wins live at the bottom of the
ladder and die out by L5. The league arm is no genius either at this tiny
budget, but it puts wins on nine of the ten rungs, including a game each
at L9 and L10. Same algorithm, same budget, same lone scripted teacher;
the only added ingredient is that half its practice was against opponents
that kept changing as it did. Variety in sparring bought reach, and it
bought reach at rungs neither arm ever met in training.

**4. The honest caveats.** One seed per arm and 12 episodes a rung, so no
single cell of that table is worth quoting; the totals (9 against 35) and
the shapes are the result. This is also self-play at its very smallest:
the pool holds only the agent's own recent past, so it cannot teach what
neither the teacher nor the agent has yet discovered. The full-size
version of this design, with a prioritized scripted mix and a bigger pool,
is [agents/league_duel.py](../../agents/league_duel.py), and
[experiment 03](../../experiments/03-generalization/README.md) measures it
properly, at 48 to 400 episodes a cell.

## Where to go from here

Sweep the mix. The solution's 50% is one point on a dial that runs from
0% (the fixed arm) to 100% (pure self-play, no scripted teacher at all).
Try 25% and 75%, and try pure self-play: watch what happens to the bottom
rungs when the scripted teacher leaves the diet entirely. You are mapping
the tradeoff the full league's 60/40 mix and priority weights were tuned
around.
