# Experiment 04, fuel economics: the tax bounds the mean, the tank bounds the burst

*Level: intermediate. In plain English: charging for fuel makes long burns
expensive; a hard fuel tank makes them impossible. Different rule, different
behavior, and the difference is measurable.*

*A notebook-style experiment: everything here runs from the repo root with
the snippets below. No grader; the graded flagship for this rule family is
[02-reward-hacking](../02-reward-hacking/README.md).*

**Question.** Thrust was originally priced with a per-second tax
(`thrust_cost=0.05` per second lit, in match time). The spawn-ambush era proved the
flaw: a tax bounds *average* consumption but not a single long burn, so a
burn-to-hover ambush was merely expensive rather than impossible (see
[docs/REWARD-SPEC.md](../../docs/REWARD-SPEC.md), exploit 3). What does a
hard burst budget (`FuelTank`: ~1.5 physics-seconds of continuous burn, then
thrust is dead until the tank regenerates) do that the tax cannot?

**Method.** The env reports `thrust_frames / frames` (duty) and
`longest_burn_s` per episode, so burn discipline is measurable, not
aesthetic. The cleanest subject is a policy that *wants* to burn forever: the
fresh wall-rider from experiment 02 trains in a no-fuel world and finishes
with its engine mostly lit (median duty 0.76). Evaluate it with the tank off,
then force the tank on: the `fuel=True` constructor kwarg overrides the era
preset. Run [experiments/02's run.py](../02-reward-hacking/README.md) first;
it writes the `runs/exp02-wallrider/` checkpoint this reads. Then, from the
repo root, paste the whole block into your shell (on Windows, start `python`,
paste the lines between the `EOF` markers, and press Enter):

```
python - <<'EOF'
import sys, statistics
sys.path.insert(0, "experiments")
import exputil as X

for fuel in (False, True):
    st = X.duel_stats("runs/exp02-wallrider/fresh_agent.json", 3, 6,
                      "v1-freewalls", seed0=70_000, fuel=fuel)
    print(f"fuel={fuel}: duty {statistics.median(s['duty'] for s in st):.2f}  "
          f"longest burn {statistics.median(s['longest_burn_s'] for s in st):.2f}s  "
          f"outcomes {[s['outcome'] for s in st]}")
EOF
```

**Measured result (reference platform, 2026-08-01, world pin
`30f439b240a2`).** Tank off: median duty 0.69, median longest burn 1.05 s, five
wall-riding draws and a loss in six episodes. Tank forced on: duty 0.35,
longest burn 0.15 s, and every episode becomes a loss, because the burn
pattern the strategy depends on physically no longer exists. A reward tax
could only have priced that thrust; the tank removes it, which is the
design point: encode *values* in reward, but encode *capabilities* in
mechanics.

**The shipped generation is already disciplined.** The same counters over the
modern checkpoints (12 episodes vs the ported L7, era-matched rules,
re-measured 2026-08-01, world pin `30f439b240a2`):

| checkpoint | duty (median) | longest burn (median) |
|---|---|---|
| [duel_ppo_v3_league](../../agents/models/duel_ppo_v3_league.json) @ `v3-rude` | 0.115 | 0.40 s |
| [duel_ppo_v6_final](../../agents/models/duel_ppo_v6_final.json) @ `v6-full` | 0.007 | 0.03 s |

The scripted bots' hand-tuned governor runs ~5% duty; v3, from the looser
era, burns about twice that, and the champion runs far under it. The
economics came from translating that governor's
burn:regen ratio (0.67/s vs 0.035/s, ~19:1) into both the reward price and
the tank, so "flies like the bots were tuned to fly" is a measured property.

**Do it yourself.** Retrain experiment 02's fresh agent with `--rules
v4-honest` (tank on from birth) and compare its duty trajectory; or sweep
`thrust_cost` in [exputil.train_short_ppo](../exputil.py) and plot duty vs
price to find where the tax alone stops mattering because the tank already
binds. The back of the book, with one sweep measured and read, is
[exercises/exp04-tank-vs-tax](../../exercises/exp04-tank-vs-tax/README.md).
