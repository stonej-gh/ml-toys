# Experiment 03 "Break it": whose climb is it?

The back of the book for the Break-it in
[experiments/03-generalization](../../experiments/03-generalization/README.md):
run the matrix with `--era v3-rude`, watch the v3 league row climb, and
answer the question the experiment leaves you with: which part of that climb
is a better policy, and which part is an easier arena?

```
python experiments/03-generalization/run.py --era v3-rude
```

## What the run shows

Judged under today's `v6-full` rules, the v3 league checkpoint reads
36/48 at L7 and 21/48 at L10. Hand it back its own era, `v3-rude`, and the
same weights read 41/48 and 34/48. The wall-touch medians barely move
(1.0 to 3.5 per episode in both worlds), and that is your first answer:
whatever bought the climb, it was not the wall exploit; that habit is mild
and present in both columns.

## The decomposition, measured

The eras differ by more than one thing, so hold `v6-full` still and hand
back ONE v3-era condition at a time (48 episodes a cell, the constructor
kwargs override the preset):

| conditions | L7 | L10 | median walls |
|---|---|---|---|
| `v6-full`, as judged | 36 | 21 | 1.5 / 3.5 |
| `v6-full` + straight lasers | 40 | 31 | 1.0 |
| `v6-full` + no fuel tank | 44 | 33 | 1.5 / 4.0 |
| `v3-rude`, its own era | 41 | 34 | 1.0 / 3.5 |

At L10 the story is even-handed: straight ballistics alone return 10 of the
13 missing wins, and the free tank alone returns 12. Each condition
recovers most of the era gap by itself, because each hands back one thing
the policy fitted itself to: an aim model built for straight shots
([experiment 05](../../experiments/05-curved-lasers/README.md) measures
that fit directly), and a thrust habit built for an engine with no burst
limit ([experiment 04](../../experiments/04-fuel-economics/README.md)
measures that one).

So the honest answer to the experiment's question: the v3 climb is real
skill, but a slice of it is skill AT ITS OWN ERA, roughly ten points of
fitted ballistics and ten of fitted fuel habits at the top rung, and
almost none of it is walls. "Better policy" and "easier arena" turned out
to be separable, and the way to separate them was one flag at a time.

## The transferable habit

When a score jumps after an environment change, resist reading it as one
number. List what actually changed, then re-measure with each difference
restored alone. The counters you need (walls, duty, burn lengths) were
already in the env's `info` dict, which is the quiet lesson: the
instrumentation you add while building is the decomposition you get for
free later.
