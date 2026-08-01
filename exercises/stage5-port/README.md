# Stage 5 exercise: port the pilot by hand, then shrink it to int8

The back of the book for the stage 5 exercise in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md): rebuild the
champion's forward pass from nothing but its .json weight file, prove your
copy against the repo's reference, then quantize the weights to 8-bit
integers and measure what the shrink costs in actual wins. The sample
solution is [solution.py](solution.py).

```
python exercises/stage5-port/solution.py                 # all three parts
python exercises/stage5-port/solution.py --episodes 20   # quicker last part
```

## The walk

**1. The port is 20 lines.** Open the champion's .json: three layers, each
a weight grid and a bias list, a field naming the bend (`tanh`), and the
action table. The forward pass is: multiply, add bias, bend, repeat, and
take the biggest of the last layer's twelve numbers. The solution's
`forward()` does it with plain Python lists. If you are porting to C, or a
shader, or a spreadsheet, it is the same 20 lines in another costume.

**2. Prove it before you trust it.** One hundred random observations
through your forward and through the repo's reference
([orbitduel/netpilot.py](../../orbitduel/netpilot.py)): the action picks
must match 100 out of 100. The solution gates on this and stops if it
fails, because every later measurement is meaningless while the port is
wrong. Measured: **100/100**. This tiny check is the same contract as the
spotter's golden bundle ([deploy/](../../deploy)), one size smaller.

**3. Quantize, two ways, and measure both.** Weight-only int8: each weight
becomes a small integer times a scale. The crude version uses one scale per
layer; the better version uses one scale per output row, which is the same
idea as the spotter's per-channel scales. Then the measurement that
actually matters: fly the real duel against the ported L10 on identical
seeds. Measured on the phone champion, 60 episodes per arm (world pin
`30f439b240a2`):

| pilot | random decisions changed (of 1,000) | wins vs L10 (of 60) |
|---|---|---|
| float reference | 0 | 34 |
| int8, one scale per layer | 5 | 27 |
| int8, one scale per row | 3 | 30 |

**4. Read it with the repo's own discipline.** The decision-flip column is
solid: finer scales track the float net more closely, 5 flips falling to 3
per thousand. The wins column is 60 episodes an arm, and this repo's
standing rule is that nothing per-rung below about 400 episodes belongs in
a document as a settled number, so read those wins as one honest sample:
the direction (crude scales cost the most, finer scales cost less, and the
gap to float may not survive a bigger n) rather than as exact prices.
Rerun with more episodes if you want the settled version; the seeds are in
the file. What IS settled: the whole brain now fits in **6,220 bytes**, a
quarter of the float file, and it still beats the top scripted bot almost
half the time.

**5. Why this is the whole edge-AI methodology.** Every deployment of a
small net to cheap hardware walks exactly these stairs: reference
implementation, equality gate, quantize, then measure the METRIC YOU CARE
ABOUT (wins, not weight error) at each precision. The spotter's version of
the same staircase, with activations quantized too and a bit-exact integer
gate at the bottom, is [experiment 06](../../experiments/06-spotter-port/README.md).

## Where to go from here

Push down: try one scale for the whole net, then int4 (round onto -7..7)
per row, and find where the pilot stops winning. Then port your forward to
a second language and check the 100/100 gate holds there too; the .json
plus this page is everything a port needs, which is the point.
