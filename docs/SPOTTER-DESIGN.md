# spotter design

The single reference for the model, the data, and the verification ladder.
The [README](../README.md) carries the pitch; this doc carries the numbers.

## Task

Semantic segmentation of rendered duel frames into five classes (defined in
[`spotter/__init__.py`](../spotter/__init__.py)):

| id | class | drawn as |
|----|-------------|----------|
| 0 | background | space, star clutter, field border, hole glow, letterbox |
| 1 | interceptor | narrow-dart hull, blue |
| 2 | fighter | broad-wedge hull, orange |
| 3 | laser | 3×3-px bright square (stylized ~2× physics size) |
| 4 | hole | the hard event-horizon disk only (soft glow is background) |
| 255 | *ignore* | thrust flames: transient exhaust, excluded from the loss |

The two hulls differ in silhouette, not just color, so the class split does
not lean on color alone. Input is a 320×192 RGB frame (the 16:9 field maps to
320×180 plus 6-px letterbox bars); the exact world→px mapping is
`SCALE = 15/64` in [`spotter/render.py`](../spotter/render.py).

## Labels: exact by construction

[`render.py`](../spotter/render.py) draws every entity twice from the same
geometry: shaded into the RGB frame, flat class ids into the mask. There is
no labeler to trust and no label-review gate; the only human check is a
rendered contact sheet ([`tools/contact_sheet.py`](../tools/contact_sheet.py))
confirming the renderer itself looks right.

Geometry provenance: hull outlines, physics constants (field 768×1365.33 pt,
`HOLE_R` 68, `SHIP_R` 27), and the replay-frame schema come from this repo's
own RL half (the `orbitduel` duel environment; see
[LEARNING-NOTES](LEARNING-NOTES.md)); the replay JSONs in
[`assets/replays/`](../assets/replays/) are recorded league self-play
episodes from that environment. The vision half does not depend on it at
runtime.

## Dataset

[`tools/gen_dataset.py`](../tools/gen_dataset.py) emits compressed npz shards
(frames uint8 NHWC, masks uint8 NHW) + a manifest listing every seed/source.

- **Random scenes** ([`spotter/sample.py`](../spotter/sample.py)): seeded
  uniform coverage, with ship poses anywhere legal (outside the hole
  keep-out, off the walls), 0-8 lasers, 5% dead ships, varied star clutter.
- **Replay frames**: ~30% of train, mixed in for realistic duel correlations
  (orbits, close passes, laser exchanges). One full episode
  (`seed_005.json`) is held out for test only.
- Splits use disjoint seed blocks (train 1M+, val 2M+, test 3M+), so there
  is no leakage by construction.

## Architecture (~28,126 params raw / 27,970 folded)

[`spotter/model.py`](../spotter/model.py). Ops restricted to what a
simple custom kernel can do: 3×3 conv, 1×1 conv, stride-2 conv, 4×4 avg-pool,
×2 nearest-neighbor upsample, elementwise add, ReLU. BatchNorm trains, then
folds into conv weights at export (fold: −2c BN params, +c bias per layer).

| stage | layers | params (raw → folded) |
|-------|--------|-----------------------|
| trunk | 3→8→16→16→24→24→32, all 3×3, strides 1/2/1/2/1/2, BN+ReLU | 19,464 → 19,344 |
| M1 patch head | 4×4 avg-pool + 1×1 conv → 5 logits | 165 |
| M2 decoder | 3× (×2 NN-up + additive 1×1 skip from trunk /4,/2,/1 + 3×3 conv) 32→16→12→8 + 1×1 seg head | 8,497 → 8,461 |

M1 trains the trunk+head **densely on full frames** against the cell-label
grid: heatmap cell (i,j) owns the 8×8 tile at the center of its 32×32
receptive window and is labeled by the rarest non-background class present
in that tile (laser > interceptor > fighter > hole; class-weighted loss).
The head still evaluates single 32×32 patches (that path feeds the golden
vectors), but it is not the training view.

> **Lessons (2026-07-08, M1 iterations):**
> 1. Training on *cropped* 32×32 patches reached 99.9% patch accuracy yet
>    collapsed to all-background when run fully-convolutionally. Isolated
>    patches zero-pad every conv at their borders while full-frame windows
>    see real neighbors, and the trunk's receptive field is wider than
>    32 px; the two views are NOT equivalent. Dense-coarse training
>    removes the mismatch with the same architecture.
> 2. Labeling each cell by its window's center *pixel* made lasers
>    undetectable by construction: a 3-px laser lands on the stride-8 grid
>    of center pixels ~1 time in 7. Tile-*presence* labeling guarantees
>    every entity lights at least one cell.
> 3. Laser tiles bloom ~2× around true positions (held-out episode:
>    precision 0.46 at recall 0.99 with inverse-sqrt class weights).
>    Softer cbrt weights were tried and rejected: precision 0.62 but
>    recall dropped to 0.90, and a missed laser reads worse in the demo
>    than a confidence-dimmed bloom. M2's per-pixel decoder is the real
>    precision fix.
> 4. The original no-skip decoder ("int8-simpler") plateaued at val IoU
>    ~0.59 interceptor / 0.60 laser: an ×8 NN-upsample from /8 features
>    cannot localize an 18-px hull or 3-px laser to pixel precision; a
>    ±1-px ring on objects that small costs exactly the missing IoU.
>    Additive 1×1-projected skips (+1,180 params, still int8-clean:
>    elementwise adds requantize, no concat) fixed it decisively:
>    min val IoU 0.585 → 0.973.
> 5. 99.9-percentile activation calibration failed the int8 IoU gate on
>    small classes (laser 0.963): the rare bright activations a percentile
>    clips ARE the lasers. Max calibration passed everything (laser
>    int8-vs-float IoU 1.000). On near-binary synthetic frames there is
>    no outlier noise to trim, so clipping only destroys signal.

M2 adds the decoder for dense masks, fine-tuning from M1. Training uses
photometric domain randomization (brightness/channel gain, gaussian noise);
a fixed seeded STRESS variant of the test split (stronger perturbation) is
evaluated alongside the clean splits.

## Verification ladder

1. **Float gate first**: pure-numpy reference forward pass vs PyTorch on
   seeded golden vectors. Measured 2026-07-08 on the trained M1 model over
   held-out replay frames: max|Δ| 9.5e-06 on logits, 100% heatmap-argmax
   agreement; **tolerance frozen at 1e-4** (one decade of margin). Gate:
   100% patch-argmax agreement, ≥99.99% per-pixel argmax.
2. **int8 gate**: per-channel symmetric weights, per-tensor symmetric
   activations, **max** calibration (see lesson 5); integer numpy reference
   in the bundle, bit-exact by construction (int8 weights, int32
   accumulators, recorded requantize multipliers). Measured 2026-07-08 on
   the frozen model: ≥99.99% per-pixel argmax agreement vs float,
   dataset-aggregated IoU vs float ≥0.99 every class (gates: ≥99% / ≥0.98).
   In the demo, 2 of 61,440 overlay pixels differ on a busy frame.
3. Training gates: M1 val acc ≥90% (per-class ≥80%): **passed** (99.6%,
   worst class 96.2%). M2 per-class IoU ≥0.70 val, ≥0.60 held-out episode,
   ≥0.50 stress: **passed** (worst class: 0.973 val, 0.965 held-out, 0.965
   stress; dense numpy gate measured max|Δ| 2.4e-05, 100% per-pixel argmax).

## Demo (M2)

Local FastAPI + canvas web app (port 5051): a recorded duel plays while the
net's mask overlay tracks it live, with per-class telemetry curves (ship
pixels, lasers in flight) and a float-vs-int8 A/B toggle. Palette in
[`spotter/__init__.py`](../spotter/__init__.py).

## Milestones

- **M0**: renderer + exact masks + dataset generator + contact sheet + tests
  (this tree).
- **M1**: dense-coarse heatmap training → dense-JSON export → numpy float
  gate → heatmap in the demo app.
- **M2**: decoder (additive skips), dense training + domain randomization,
  live overlay demo + telemetry (this tree).
- **M3**: freeze, PTQ int8, golden bundle ([`deploy/`](../deploy/README.md)),
  float-vs-int8 A/B demo mode, tag `spotter_v1` (this tree). Bundle verified
  relocatable (runs from a bare copy) and reproducible (rebuild → identical
  golden npz sha).
