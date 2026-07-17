# Experiment 05, curved lasers: train on a lie, learn a lie

*A notebook-style experiment: run the snippet below from the repo root. No
grader; the exact-outcome anchor for shipped checkpoints is
[tests/test_golden_eval.py](../../tests/test_golden_eval.py).*

**Question.** For several eras this sim flew the agent's *own* lasers
straight while gravity curved everything else, including the opponent's
shots. The asymmetry was an oversight, documented in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md), Stage 5. The learned
aim model looked fine in the sim and missed visibly in Spacewar.
`gravity_on_lasers` is now a rule flag: how much of a trained policy's skill
is really an aim model fitted to its training-era ballistics?

**Method.** Ablate one physics bit under otherwise fixed rules: evaluate the
straight-trained [duel_ppo_v4_final](../../agents/models/duel_ppo_v4_final.json)
and the curve-trained [duel_ppo_v5_curved](../../agents/models/duel_ppo_v5_curved.json)
/ [duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json) against L10,
with `gravity_on_lasers` both off and on. Pure-Python inference on fixed
seeds: every cell is exact.

```python
# python - <<'EOF'   (from the repo root)
import sys
sys.path.insert(0, "experiments")
import exputil as X

for model in ("duel_ppo_v4_final.json", "duel_ppo_v5_curved.json",
              "duel_ppo_v6_final.json"):
    for curved in (False, True):
        st = X.duel_stats(X.MODELS / model, 10, 12, "v4-honest",
                          gravity_on_lasers=curved)
        print(f"{model:26s} curved={curved}: wins {X.wins(st):2d}/12  "
              f"losses {sum(1 for s in st if s['outcome'] == 'loss'):2d}")
# EOF
```

**Measured result (wins/12 vs L10, exact on every platform, 2026-07-10):**

| checkpoint | straight lasers | curved lasers |
|---|---|---|
| v4 (straight-trained) | **11** | 10 |
| v5 (curve-trained) | 2 *(10 losses)* | **9** |
| v6 (curve-trained champion) | 9 | **11** |

Each policy is strongest in the ballistics it trained under, and v5 is the
dramatic case: straighten its lasers and it collapses from 9 wins to 2 with
10 losses. Its aim model *leads targets expecting the curve*; fired straight,
those shots miss and every miss is a 15° cone opportunity wasted against an
opponent that shoots back. Physics fidelity is not a graphics nicety: it is
part of what the network knows.

**The sim-to-real punchline.** This flag exists because the "cleaner" v4
generation, dominant in the sim, *lost* in Spacewar to opponents the
ruder v3 beat: the subtler policy leaned harder on the sim's straight-laser
lie. Closing a model gap beats tightening style constraints, and the gap you
close must include your own shots, not just the world's.

**Do it yourself.** Sweep levels (the effect grows with opponent quality:
scripted lead-aim at high levels punishes misses hardest), or retrain
experiment 02's short PPO with `gravity_on_lasers` flipped and watch the
fire-discipline stats diverge.
