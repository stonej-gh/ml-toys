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
/ [duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json) against the
ported L7, with `gravity_on_lasers` both off and on. L7 rather than L10,
since the 2026-07 opponent port: the ported top rung beats every era
checkpoint under either ballistics (v5 and v6 hold 0-4% there), and an
ablation cannot show its signal in a column of zeros. Pure-Python inference
on fixed seeds; 100 episodes a cell, because this is a statistical claim,
not an exact-outcome pin.

```python
# python - <<'EOF'   (from the repo root; a few minutes)
import sys
sys.path.insert(0, "experiments")
import exputil as X

for model in ("duel_ppo_v4_final.json", "duel_ppo_v5_curved.json",
              "duel_ppo_v6_final.json"):
    for curved in (False, True):
        st = X.duel_stats(X.MODELS / model, 7, 100, "v4-honest",
                          gravity_on_lasers=curved)
        print(f"{model:26s} curved={curved}: wins {X.wins(st):3d}/100")
# EOF
```

**Measured result (wins/100 vs the ported L7, measured 2026-08-01):**

| checkpoint | straight lasers | curved lasers | cost of straightening |
|---|---|---|---|
| v4 (straight-trained) | 32 | **57** | 25 |
| v5 (curve-trained) | 12 | **60** | **48** |
| v6 (curve-trained champion) | 8 | **55** | **47** |

Read the last column, and read it as two stacked effects. Every policy,
even the straight-trained v4, wins more with curved shots against this
opponent, so part of the column is the weapon itself: a shot that bends
with gravity connects where a straight one misses, whoever fires it. v4's
25 points estimate that weapon effect, since v4 never learned to expect
the curve. The curve-trained pair loses that much *and roughly as much
again* (48 and 47 points), and the extra is the fitted aim: a policy that
learned to lead targets expecting the curve misses systematically once its
shots fly straight, and every miss is a 15° cone opportunity wasted
against an opponent that shoots back. A straight-trained control is what
lets you split "the tool got worse" from "the skill was fitted to the
tool"; without the v4 row, the other two rows are ambiguous. Physics
fidelity is not a graphics nicety: it is part of what the network knows.

**What the opponent port did to this experiment.** This page originally
measured at L10 (12 episodes, 2026-07-30) and reported v4 riding out the
ballistics flip at 11/12 either way. The next day's opponent port rebuilt
the scripted ladder around the real robot's moves, and against the ported
L10 the era checkpoints win 0-20% regardless of ballistics: the rung
saturated, taking the old table and its cleanest claim with it. The
measurement above is the same ablation moved down to where the signal still
lives, at a sample size that supports it. An ablation's readout rung is part
of its design, and an opponent change can silently invalidate it: the same
lesson as "never train against a lie", pointed at your own yardstick.

**The sim-to-real punchline.** This flag exists because the "cleaner" v4
generation, dominant in the sim, *lost* on the original engine (the arcade
source the physics began in: [PROVENANCE](../../PROVENANCE.md)) to opponents
the ruder v3 beat: the subtler policy leaned harder on the sim's
straight-laser lie. Closing a model gap beats tightening style constraints,
and the gap you close must include your own shots, not just the world's.

**Do it yourself.** The answer sheet is
[exercises/exp05-ablation](../../exercises/exp05-ablation/README.md): sweep
the ablation across rungs and watch its signal appear, grow, and then
saturate as the opponent strengthens; or retrain experiment 02's short PPO
with `gravity_on_lasers` flipped and watch the fire-discipline stats
diverge.
