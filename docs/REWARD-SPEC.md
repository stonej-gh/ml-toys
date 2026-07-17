# The reward spec, and how it got gamed

Every number in the duel's reward function is here, but the numbers are the
least interesting part. This environment's reward design was wrong four times
in instructive ways, and each failure is reproducible in this repo: the exact
rules of each era are a named preset, and the champion that exploited each era
ships as a checkpoint. This document leads with the failures because they are
the point.

## The four exploits

### 1. The draw camper

The first PPO runs plateaued at 100% draws: the agent learned that hiding in
a corner and never engaging avoided the death penalty, and the reward said
nothing against it. Watch it: `viz/watch.html?dir=replays/v1-drawcamper`
(and `v2-drawcamper` for the later, subtler variant).

Fix: a small per-second time cost (`time_cost=0.004` per wall second, about
0.36 over a full 90 s episode) plus potential-based approach shaping
(`pot_coef=0.5`). Potential shaping is the textbook-safe kind: it rewards
approach as `gamma * phi(s') - phi(s)` with `phi = -c * dist / h`, which
cannot change the optimal policy, only the road there.

### 2. The wall rider

The arena walls were free bounces, and the agent noticed before we did. It
learned to carom off the walls like a pinball: 12.2 wall touches per episode,
using rebounds for propulsion and evasion. By the rules we published it was
winning fair and square; specification gaming is exactly this literal.

Reproduce it, counters included:

    python agents/duel_eval.py --model agents/models/duel_ppo_v1.json \
        --rules v1-freewalls --level 10

and compare any modern checkpoint in the same free-walls world (it plays
clean anyway; the habit was learned, not necessary). Watch the curated
episodes at `viz/watch.html?dir=replays/v3-wallrider`.

Fix, in the order that matters: **physics first, then values.** First the
walls got honest: a scrape now imparts a decaying spin that fights your
steering (`spin_kick`, `SPIN_DAMP`), which alone collapsed a random policy's
wall-pump survival trick from 46% to 5%. Then the reward priced what
remained: `wall_penalty=0.8`, about 0.8 deaths per strike. Result across the
retrained generation: 12.2 touches per episode down to 0.42.

### 3. The spawn ambush and the burn-to-hover

Spawns were at fixed positions, so the agent memorized them and learned a
7-second opening ambush. Worse, thrust was taxed per second on average, so a
long continuous burn to hover in the opponent's spawn lane was merely expensive
rather than impossible. The tax bounded the mean, not the burst.

Fix: again mechanics before reward. Spawn PHASE is now randomized (the whole
start configuration rotates, so there is no lane to camp: `spawn_phase`),
and thrust runs on a hard fuel tank (`FuelTank`: about 1.5 physics seconds
of continuous burn, then thrust is dead until the tank regenerates). The
tank level is observation element 18, so the agent can plan around it. A
burn-to-hover ambush is now impossible, not merely costly.

### 4. The straight-laser aim model (a fidelity bug, not a reward bug)

For several eras this simulation flew the agent's own lasers straight while
Spacewar curved them under gravity. The learned aim model looked
fine here and missed visibly there. The lesson generalizes: constraints and
physics gaps do not just lower a policy's score, they change what it learns.
`gravity_on_lasers=True` (the `v6-full` preset) closed the gap, and reading
the curve became the learned pilot's edge over the scripted ladder's
straight-line lead aim.

## The current reward, in full

Terminal: +1 opponent dies, -1 you die (either cause). Per step, with the
defaults the reference agents train under:

| term | value | why it exists |
|---|---|---|
| hit shaping | +/-0.3 per hit dealt/taken | the duel is scored per hit; game-shaped, not hand-authored strategy |
| potential shaping | `pot_coef=0.5` on distance | safe approach incentive (see camper) |
| time cost | 0.004 per wall s | draws are not free |
| wall penalty | 0.8 per own strike | rebound play is not orbit play |
| thrust cost | 0.05 per wall s lit | orbit shifts are deliberate, paid acts |

Constraints as rules, not rewards: the fuel tank (burst budget), the fire
cone (trigger works only within 15 degrees of the bow, in plausible range,
with a clear line past the hole), spawn-phase randomization, and wall spin.
Each one exists because a reward-only version of it was gamed first.

## Era presets

| preset | walls | fuel | spawn | lasers | era it reproduces |
|---|---|---|---|---|---|
| `v1-freewalls` | free | none | fixed lanes | straight | wall rider, draw camper |
| `v3-rude` | spin + penalty | none | fixed lanes | straight | spawn ambush, burn-to-hover |
| `v4-honest` | spin + penalty | tank | randomized | straight | honest but aim-handicapped |
| `v6-full` | spin + penalty | tank | randomized | curved | the shipped rules |

`OrbitDuelEnv(rules="v1-freewalls")` gives you the broken world back;
explicit constructor arguments override any preset field. The `info` dict
reports `wall_touches`, `thrust_frames`, `longest_burn_s`, and the round-end
cause, so every exploit above is measurable, not anecdotal.
