# Experiment 04 "Do it yourself": the price sweep

The back of the book for the do-it-yourself in
[experiments/04-fuel-economics](../../experiments/04-fuel-economics/README.md):
sweep the thrust price and find where the tax stops mattering because the
tank already binds. No solution.py; the whole sweep is the snippet below,
about two minutes a row.

```python
# python - <<'EOF'   (from the repo root)
import statistics, sys, tempfile
from pathlib import Path
sys.path.insert(0, "experiments")
import exputil as X

for rules in ("v1-freewalls", "v4-honest"):
    for price in (0.0, 0.05, 0.15, 0.30):
        net = X.train_short_ppo(seed=11, rules=rules, level=3, updates=90,
                                thrust_cost=price, log=lambda *_: None)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "net.json"
            X.export_net(net, p)
            st = X.duel_stats(p, 3, 6, rules, seed0=70_000, thrust_cost=price)
        print(rules, price,
              statistics.median(s["duty"] for s in st),
              statistics.median(s["longest_burn_s"] for s in st))
# EOF
```

## What one sweep measured (seed 11, 90 updates)

| rules | price | median duty | median longest burn |
|---|---|---|---|
| `v1-freewalls` (no tank) | 0.00 | 0.94 | 2.92 s |
| `v1-freewalls` | 0.05 | 0.00 | 0.00 s |
| `v1-freewalls` | 0.15 | 0.00 | 0.00 s |
| `v4-honest` (tank on) | 0.00 | 0.44 | 0.27 s |
| `v4-honest` | 0.05 | 0.03 | 0.03 s |
| `v4-honest` | 0.15 | 0.00 | 0.00 s |

## How to read it

**The tank binds the burst even when thrust is free.** Compare the two
price-zero rows: without a tank, a free engine runs 94% of the time in
burns up to 2.9 s; with the tank, the same free engine manages 44% duty
and burns capped at 0.27 s. Nothing about that cap involves the reward.
That is the experiment's design point measured from the other side: the
tank is a capability rule, and it holds whatever the prices do.

**At a short budget, a tax is not a price, it is a ban.** The moment
thrust costs anything at all, this 90-update agent's duty collapses to
about zero in BOTH rulesets. It has not learned to budget fuel; it has
learned "engine bad" before learning what the engine is for, because at
this training length the cost signal arrives long before the benefit
signal does. The shipped trainers charge the same 0.05 over 5,000 to
20,000 updates and end up with pilots at 1-8% duty who still fly
([experiment 04](../../experiments/04-fuel-economics/README.md)'s
discipline table). Same price, different budget, opposite meaning: a
reward term is not a fixed statement, it is a statement whose reading
depends on how long the learner gets to think about it.

**So where does the tax stop mattering?** Everywhere right of the first
column, at this budget: the tank's cap only shows when thrust is free,
because any tax already drives usage below what the tank would allow. The
regime where the two instruments genuinely share the work, tax shaping the
average while the tank clips the burst, needs the long-budget training the
real trainers run.

## Where to go from here

Rerun the sweep at 240 and 600 updates and watch the cliff between price
0.00 and 0.05 turn back into a slope. The budget at which an agent can
afford to learn "thrust is worth its price" is itself a measurable number,
and it is a nice concrete instance of why reward tuning done on short runs
misleads about long ones.
