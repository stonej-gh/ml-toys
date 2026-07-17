# Experiment 06, the spotter: retrain from nothing, port to nothing

**Question.** The spotter ships as a frozen golden bundle
([deploy/](../../deploy)): folded-JSON weights, golden vectors, and a
numpy-only verifier. Can a stranger (a) regenerate the dataset bit-for-bit,
(b) retrain the CNN from scratch, and (c) carry the result through the whole
deployment ladder (export -> dependency-free float forward -> int8) with
every hop measured? That ladder is the edge-AI methodology the repo exists to
teach; this experiment is the graded version of it.

**Method.** The grader runs a *micro* edition of the full stranger flow, small
enough for CI but exercising every stage:

1. **Dataset determinism.** Generate 85 training + 16 val frames with
   [tools/gen_dataset.py](../../tools/gen_dataset.py)'s `build_split` (seeded
   scenes + replay frames) and compare the arrays' sha256 against a locked
   constant. The renderer is integer-exact by construction, so this hash must
   match on every platform; a mismatch means the rendering contract broke.
2. **Retrain.** Train `SpotterNet(mode="dense")` from scratch on the micro
   set: 12 epochs, CPU, single thread, seeded (~30 s). Bar: every foreground
   class reaches val IoU >= 0.15 (reference run: interceptor 0.52, fighter
   0.67, laser 0.45, hole 0.93). Torch training is deterministic per-machine,
   threshold-comparable across platforms; the bar sits ~3x under the worst
   reference class.
3. **Port, float.** Export with BN folding
   ([spotter/export.py](../../spotter/export.py)) and replay the val frames
   through the pure-numpy reference
   ([spotter/reference.py](../../spotter/reference.py)): argmax agreement
   >= 99.99% per frame (reference: 100.0%, max logit delta 2.6e-5).
4. **Port, int8.** Post-training quantization
   ([spotter/quantize.py](../../spotter/quantize.py)), then the integer
   forward from the shipped bundle
   ([deploy/spotter_forward.py](../../deploy/spotter_forward.py)): argmax
   agreement vs float >= 98% (reference: 99.94%; the frozen golden bundle's
   own gate is 99%).

**Expected result.** All four bars pass in about a minute of CPU. The frozen
golden bundle itself is verified separately by
[tests/test_bundle.py](../../tests/test_bundle.py) on every test run; this
experiment proves the *path that produced it* still works end to end from a
bare clone.

## Run the full stranger flow

```
python experiments/06-spotter-port/run.py     # full dataset + M1 + M2 + bundle + verify
```

This is the real Phase-2 flow (full-size dataset, both training stages,
bundle rebuild, `deploy/verify.py`), orchestrated; expect tens of minutes on
CPU. It rewrites `deploy/models/` and `deploy/golden/` with YOUR retrain's
bundle (that is the exercise); restore the shipped frozen bundle afterwards
with `git restore deploy`. Training on GPU/MPS is not bit-reproducible, so
your bundle will differ from the shipped one by bits while passing the same
gates; [deploy/README.md](../../deploy/README.md) scopes exactly which
artifacts are bit-reproducible and which are gate-reproducible.

## Grade it

```
python -m pytest experiments/06-spotter-port/grade.py -m grade_cheap -q
```
