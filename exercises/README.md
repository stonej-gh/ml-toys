# Exercises: the back of the book

Every do-it-yourself exercise in this repo has an answer here: a walkthrough
that does the whole thing, with measured results from the reference
platform, and a sample solution you can run and modify. Do the exercise
first; these pages are for checking your work, for getting unstuck, and for
the discussion of what the numbers mean.

The seven stage exercises follow
[docs/LEARNING-NOTES.md](../docs/LEARNING-NOTES.md); the rest answer the
"Break it" / "Do it yourself" / "Shrink it" prompts in the
[experiments](../experiments/README.md).

| exercise | answers | solution | what you build or measure |
|---|---|---|---|
| [stage0-gravity](stage0-gravity/README.md) | notes stage 0 | yes | a 100-line gravity sim, Kepler check, integrator failure |
| [stage1-dqn](stage1-dqn/README.md) | notes stage 1, exp 01 Break-it | yes | DQN from scratch, and the collapse on demand |
| [stage2-reinforce](stage2-reinforce/README.md) | notes stage 2 | yes | the policy gradient on paper, then REINFORCE's variance |
| [stage3-exploits](stage3-exploits/README.md) | notes stage 3 | discussion | the exploit hunt, worked, with the measuring tools |
| [stage4-league](stage4-league/README.md) | notes stage 4 | yes | a 5-snapshot league, graded across the whole ladder |
| [stage5-port](stage5-port/README.md) | notes stage 5 | yes | a hand-rolled forward pass, then int8, with the win-rate bill |
| [stage6-selection](stage6-selection/README.md) | notes stage 6 | yes | your own selection bias, measured in minutes |
| [exp03-break](exp03-break/README.md) | exp 03 Break-it | snippet | v3's era climb, decomposed one flag at a time |
| [exp04-tank-vs-tax](exp04-tank-vs-tax/README.md) | exp 04 DIY | snippet | the duty-vs-price sweep, and where the tank binds |
| [exp05-ablation](exp05-ablation/README.md) | exp 05 DIY | snippet | the ablation swept across rungs: window, signal, saturation |
| [exp06-shrink](exp06-shrink/README.md) | exp 06 Shrink-it | yes | the width sweep: capacity fails one class at a time |

Ground rules, same as everywhere in the lab: the solutions run from the repo
root after the [README quickstart](../README.md#quickstart-the-pilot)
install (stage 0 needs no install at all), every number quoted in a
walkthrough was produced by the command next to it, and training results are
deterministic per machine but only threshold-comparable across machines, so
expect your numbers to differ in the third digit and occasionally in the
first. When a walkthrough's numbers and yours disagree by more than the
page says to expect, that is worth chasing: it is either a real platform
difference or a real bug, and both are the good kind of finding.

Walkthroughs whose numbers depend on the arena cite a **world pin** next to
their tables: a digest of the physics and opponent they were measured under
(see [tools/world_pin.py](../tools/world_pin.py)). A failing
`tests/test_world_pin.py` means the arena has changed since a table was
measured, and the table is due for a rerun before it is quoted.
