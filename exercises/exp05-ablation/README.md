# Experiment 05 "Do it yourself": sweep the ablation across the ladder

The back of the book for the do-it-yourself in
[experiments/05-curved-lasers](../../experiments/05-curved-lasers/README.md):
run the straight-vs-curved ablation at more than one rung and watch the
signal appear, grow, and saturate as the opponent strengthens. No
solution.py; the sweep is the experiment's own snippet with the level in a
loop.

## The quick look (12 episodes a cell, minutes; world pin `30f439b240a2`)

| rung | v4 straight/curved | v5 straight/curved | v6 straight/curved |
|---|---|---|---|
| L4 | 10 / 10 | 10 / 12 | 8 / 8 |
| L7 | 3 / 8 | 4 / 7 | 1 / 8 |
| L10 | 2 / 1 | 0 / 0 | 0 / 0 |

Three regimes in one table. At L4 the opponent is weak enough that the
pilots win with either ballistics: the ablation has nothing to bite on. At
L10 the ported opponent is strong enough that they lose with either: also
nothing to bite on. The signal lives in the middle, where the duel is
competitive enough that wasted shots decide games.

## The settled version (100 episodes a cell at L7)

The 12-episode cells above are a scouting report, not a result; a repo
rule of thumb is that per-rung claims want hundreds of episodes. At n=100
the L7 row sharpens into the experiment's headline table (v4 32/57,
v5 12/60, v6 8/55) and its two-effect reading, which lives in
[the experiment's README](../../experiments/05-curved-lasers/README.md).
Compare your own 12-episode row against it: some cells will be several
wins off, in either direction, and that gap between the scout and the
settled number is worth having felt once.

## How to read the sweep

An ablation needs a competitive readout window: too easy and every variant
saturates high, too hard and every variant saturates low, and in both cases
the difference you are trying to measure is squeezed to nothing. So before
concluding "the flag does not matter", check whether your rung can express
a difference at all. And re-check the window after any change to the
opponent: this experiment originally read out at L10, and the 2026-07
opponent port silently closed that window (the experiment's README tells
that story). A yardstick is part of the apparatus, and apparatus drifts.
