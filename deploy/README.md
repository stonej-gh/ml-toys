# spotter golden bundle

The portable deliverable of [this repo's vision half](../README.md): a frozen
tiny segmentation CNN, a pure-numpy reference forward pass, and seeded golden
vectors. A downstream port consumes this directory **read-only**: it
re-implements the ops on its target, then proves itself against `golden/`
with [`verify.py`](verify.py). Nothing hardware-specific lives here, and
nothing flows back.

## Contents

| file | what |
|------|------|
| [`models/spotter_dense.json`](models/spotter_dense.json) | frozen float model (~28k params, BN pre-folded): explicit conv / ×2-upsample / skip-add op list |
| [`models/spotter_dense_int8.json`](models/spotter_dense_int8.json) | PTQ int8 twin: int8 weights (per-out-channel symmetric), int32 biases, per-layer requantize multipliers, max-calibrated activation scales |
| [`spotter_forward.py`](spotter_forward.py) | executable spec: numpy-only float + int8 forward passes |
| [`golden/spotter_dense.golden.npz`](golden/spotter_dense.golden.npz) | 8 seeded input frames (held-out episode) + expected float/int8 logit subgrids and full argmax masks |
| [`golden/spotter_dense.golden.json`](golden/spotter_dense.golden.json) | golden metadata: seed, frame indices, tolerances/gates |
| [`verify.py`](verify.py) | the acceptance test (see gates below) |

## Model

Input: RGB frame 320×192, values `x/255` (the int8 model quantizes as
`round(x·127)`). Output: per-pixel logits over
`[background, interceptor, fighter, laser, hole]`. Ops: 3×3 conv (stride
1/2), 1×1 conv, ×2 nearest-neighbor upsample, elementwise add, ReLU, chosen
so a simple custom kernel can implement everything. Architecture and design
history: [`docs/SPOTTER-DESIGN.md`](../docs/SPOTTER-DESIGN.md).

## Verify

```sh
python verify.py        # numpy is the only dependency
```

Gates, in ladder order (float bit-faithfulness first, then int8 agreement):

1. **float**: recomputed logits within `1e-4` of golden (measured deviation
   of the reference itself: ~2e-5); per-pixel argmax matches 100%.
2. **int8**: integer path reproduces the stored argmax **bit-exactly**
   (int8 weights, int32 accumulators: deterministic on any platform);
   argmax agreement vs float ≥99%; per-class IoU vs float ≥0.98.

A port replaces `spotter_forward` with its own implementation and runs the
same script. The bundle is relocatable: copy this directory anywhere.

## Reproducing

Built by [`tools/build_bundle.py`](../tools/build_bundle.py) from the seeded
training pipeline (seed `20260708` end to end: dataset, training, calibration
frames, golden frames). Re-running the build reproduces this bundle
bit-for-bit from the same checkpoint. A full retrain is seeded and passes the
same gates, but GPU training is not bit-deterministic, so a retrained net's
weights (and therefore its goldens) will differ from this frozen one. That is
fine: the bundle freezes ONE verified net, and `verify.py` binds any port to
exactly that net.
