# Experiment 06 "Shrink it": how small can the spotter go?

The back of the book for the Shrink-it exercise in
[experiments/06-spotter-port](../../experiments/06-spotter-port/README.md):
shrink the trunk's channel budget, retrain, and find which gate fails first
and on which class. The sample solution is [solution.py](solution.py); it
runs the experiment's own micro flow (85 frames, 12 epochs, ~30 s per size)
at a sweep of widths.

```
python exercises/exp06-shrink/solution.py                 # 1.0x, 0.5x, 0.25x
python exercises/exp06-shrink/solution.py --scales 0.75   # your own points
```

## What one sweep measured

Validation IoU per class at each budget, against the micro grader's floor of
0.15 per foreground class:

| scale | widths | params | interceptor | fighter | laser | hole | verdict |
|---|---|---|---|---|---|---|---|
| 1.00 | 8,16,16,24,24,32 | 28,126 | 0.518 | 0.673 | 0.447 | 0.933 | PASS |
| 0.50 | 4,8,8,12,12,16 | 10,490 | **0.000** | 0.662 | 0.610 | 0.863 | FAIL: interceptor |
| 0.25 | 2,4,4,6,6,8 | 5,380 | 0.237 | **0.001** | **0.000** | 0.761 | FAIL: fighter, laser |

(The full-size row reproduces the experiment's reference numbers exactly,
which is your check that the sweep tool measures what the grader measures.)

## How to read it

**Capacity does not fail gracefully; it fails class by class.** Half the
width did not make everything a little worse. It made three classes slightly
different and one class *disappear*: the interceptor, the narrow dart, fell
to 0.000 IoU while the broad-wedge fighter held at 0.662. At a quarter
width the collapse rotates: this run keeps a little interceptor and loses
the fighter and the laser instead. Which class dies at a given budget is
partly luck of the seed; THAT classes die one at a time, rather than all
fading together, is the durable finding. When a shrunken model "still gets
92% overall", always ask which class paid.

**The hole is nearly free.** The biggest, highest-contrast, most regular
shape in the arena still scores 0.761 at 5,380 parameters. Easy classes
subsidize aggregate scores all the way down, which is one more reason the
spotter's gates are per-class and the honest metric is
[IoU](../../docs/GLOSSARY.md#iou), not accuracy.

**Micro-flow caveat.** These numbers come from the grader's micro edition
(85 training frames, 12 epochs), so they sit far below the full training
flow's (the shipped net's worst class is 0.973 at full size). The absolute
values would all rise with real training; the shape of the failure, one
class at a time and rarest silhouettes first, is what transfers. To run the
sweep at full fidelity, change `WIDTHS` in
[spotter/model.py](../../spotter/model.py) and run the whole stranger flow
per [experiment 06](../../experiments/06-spotter-port/README.md), tens of
minutes per point.

## Where to go from here

Find the knee: sweep 0.9, 0.75, 0.6 and locate the largest saving that
still passes every class. Then try shrinking only the decoder, or only the
trunk, and see which half of the network the small classes actually live
in.
