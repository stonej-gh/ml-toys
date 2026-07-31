# Experiment 05, curved lasers: train on a lie, learn a lie

*Level: advanced. In plain English: an agent that learned to shoot in a world
where lasers fly straight keeps missing once they curve. How much of its
skill was really an aim table fitted to the old physics? Not the place to
start; read [START-HERE](../../docs/START-HERE.md) and experiment 02 first.*

*A notebook-style experiment: run the snippet below from the repo root. No
grader; the exact-outcome anchor for shipped checkpoints is
[tests/test_golden_eval.py](../../tests/test_golden_eval.py).*

**Question.** For several eras this sim flew the agent's *own* lasers
straight while gravity curved everything else, including the opponent's
shots. The asymmetry was an oversight, documented in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md), Stage 5. The learned
aim model was fitted to that era's ballistics, and the ablation below
measures what the fit was worth. `gravity_on_lasers` is now a rule flag: how much of a trained policy's skill
is really an aim model fitted to its training-era ballistics?

**Method.** Ablate one physics bit under otherwise fixed rules: evaluate the
straight-trained [duel_ppo_v4_final](../../agents/models/duel_ppo_v4_final.json)
and the curve-trained [duel_ppo_v5_curved](../../agents/models/duel_ppo_v5_curved.json)
/ [duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json) against L10,
with `gravity_on_lasers` both off and on. Pure-Python inference on fixed
seeds: deterministic on any one machine, with single-game flips possible
across environments on knife-edge episodes.

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

**Measured result (wins/12 vs L10, measured 2026-07-30):**

| checkpoint | straight lasers | curved lasers |
|---|---|---|
| v4 (straight-trained) | **11** | 11 |
| v5 (curve-trained) | 4 *(8 losses)* | **8** *(4 losses)* |
| v6 (curve-trained champion) | 5 | **8** |

The curve-trained pair tells the fitted-aim story: straighten v5's lasers
and its wins halve while its losses double; v6 shows the same preference (8
curved, 5 straight). An aim model that *leads targets expecting the curve*
misses when shots fly straight, and every miss is a 15° cone opportunity
wasted against an opponent that shoots back. The surprise is v4: the
straight-trained policy now wins 11/12 under both ballistics, so its skill
was not an aim table at all. Whether a policy's skill transfers across the
physics it trained under is a measurable property, not a given, and this
ablation is the measurement. Physics fidelity is not a graphics nicety: it
is part of what the network knows.

**The sim-to-real punchline.** This flag exists because the "cleaner" v4
generation, dominant in the sim, *lost* on the original engine (the arcade
source the physics began in: [PROVENANCE](../../PROVENANCE.md)) to opponents
the ruder v3 beat: the subtler policy leaned harder on the sim's
straight-laser lie. Today's table complicates that era's reading: under the
current physics v4 rides out the ballistics flip alone, so the full transfer
gap held more than aim (see the Stage 5 correction in
[LEARNING-NOTES](../../docs/LEARNING-NOTES.md)). Closing a model gap beats
tightening style constraints, and the gap you
close must include your own shots, not just the world's.

**Do it yourself.** Sweep levels (the effect grows with opponent quality:
scripted lead-aim at high levels punishes misses hardest), or retrain
experiment 02's short PPO with `gravity_on_lasers` flipped and watch the
fire-discipline stats diverge.
