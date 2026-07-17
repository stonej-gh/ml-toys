# Learning notes: how to rebuild this experiment yourself

A living document (started 2026-07-01, updated as the project moves) whose job
is to let you re-create everything in this repo by hand, from scratch,
understanding every step. It's ordered the way the project actually unfolded,
because the failures were the lessons. Each stage names the concepts, the exact
code to read, what went wrong for us, and a do-it-yourself exercise.

Reading list woven through: Sutton & Barto, *Reinforcement Learning: An
Introduction* (free online) is the backbone; chapter pointers below.

---

## Stage 0: the environment IS the experiment

**Concept.** RL code is ~20% algorithm, ~80% environment + reward + evaluation
design. Every interesting result and every failure we hit lived in the
environment, not the neural net.

**What we built.** A headless physics sim
([`orbitduel/physics.py`](../orbitduel/physics.py)). It began life as a twin of
Spacewar: Orbital Duel, an iOS game (RL needs millions of steps; a real game
runs at 1× wall clock), and it runs ~12,000 env-steps/second on CPU.

**The discipline that made it trustworthy:** never assume, measure.
- The gravity constant came from an empirical fit, and coast tests against
  Spacewar confirmed orbital period to 1.8%.
- Thrust acceleration was *verified* by double-differentiating recorded
  positions, which exposed that the original engine's forces and fields used
  different unit conventions. We would have shipped a wrong sim on arithmetic
  alone.
- Analytic self-tests ([`orbitduel/selftest.py`](../orbitduel/selftest.py))
  check Kepler's laws and energy conservation. Note the symplectic-integrator
  subtlety: semi-implicit Euler *wobbles* the orbit radius ~1% but never
  drifts. Test for secular drift, not analytic equality.

**DIY exercise.** Write a 100-line gravity sim (one body, inverse-square
field, semi-implicit Euler). Verify Kepler's third law numerically. Then
break it: switch to naive (explicit) Euler and watch energy blow up.

---

## Stage 1: value-based RL, DQN on the survive task

**Concepts.** Markov Decision Process, Q-function, temporal-difference
bootstrapping, experience replay, target networks, epsilon-greedy
exploration. Sutton & Barto ch. 3, 6, 16.1; the DQN paper (Mnih et al. 2015).

**What we built.** [`agents/dqn_survive.py`](../agents/dqn_survive.py), the
whole algorithm in one file on purpose. The policy is a 2×64 MLP (~5,600
parameters, TD-Gammon's size class, small enough for a microcontroller). The
trained checkpoints are `survive_dqn_v1` and `survive_dqn_v2` in
[`agents/models/`](../agents/models).

**Task design is half the work (our first lesson).** The first spawn
distribution was too easy: *doing nothing* survived 67% of episodes, so there
was nothing to learn. Hardened to doomed sub-circular spawns. Then the metric
lied: random's mean lifetime looked strong because the distribution was
bimodal. Use medians for lifetimes.

**Reward hacking (our second lesson).** Random thrusting survived 46% of
episodes by pumping orbital energy and bouncing off the harmless walls. The
agent could have learned the same exploit; ours happened to circularize
instead. We *audited the behavior* (wall bounces, eccentricity), not just
the score. Later the duel agent DID exploit walls (Stage 3), and the fix was
physics plus reward, not hope.

**Instability (our third lesson).** Vanilla DQN at lr 1e-3 learned the skill
at update ~125k, then collapsed to ~5%: the "deadly triad" (bootstrapping +
function approximation + off-policy) plus max-operator overestimation. Fixes
that worked, in order of importance: **Double DQN** (online net picks the
action, target net prices it), lower lr (5e-4), slower target sync, and
(embarrassing but decisive) **always save the best checkpoint**. We lost our
first good policy by only saving at the end. Second subtlety: rank
checkpoints by *(median, survival-rate)* because the median saturates at the
episode cap.

**DIY exercise.** Reimplement DQN yourself against `SurviveEnv` (~150 lines).
Reproduce the collapse with vanilla DQN at lr 1e-3, then fix it with Double
DQN. Watching the collapse happen on demand teaches more than any paper.

---

## Stage 2: policy-gradient RL, PPO on the duel

**Concepts.** Policy gradients, advantage estimation (GAE), the clipped
surrogate objective, entropy bonus, actor-critic. Sutton & Barto ch. 13;
Schulman et al., *PPO* (2017) and *GAE* (2015). The style to imitate is
CleanRL: one file, no framework.

**What we built.** [`agents/ppo_duel.py`](../agents/ppo_duel.py): tiny
shared-trunk actor-critic, 8 parallel envs, curriculum over the scripted
pilot's levels (advance at 60% eval win rate). The env is `OrbitDuelEnv`
(registered as `OrbitDuel-v0`), and every rule era described below is
reproducible: pass `rules="v1-freewalls"` to get exactly the arena this
stage's agents trained in.

**Reward shaping without lying (fourth lesson).** The agent camped in the
"draw" attractor (safe orbit, 0 reward, forever). You can watch it happen:
the [`replays/v1-drawcamper`](../replays/v1-drawcamper) and
[`replays/v2-drawcamper`](../replays/v2-drawcamper) episodes load in
[`viz/watch.html`](../viz/watch.html) via `?dir=replays/v1-drawcamper`.
Pressure applied:
- a small **time cost** (draws stop being free);
- **potential-based distance shaping**: `γ·φ(s′) − φ(s)` with
  `φ = −c·dist/h`. The theorem (Ng, Harada, Russell 1999): shaping of exactly
  this form provably does NOT change which policy is optimal. Use it instead
  of ad-hoc bonuses wherever possible.
- per-hit rewards, which the game itself scores (game-shaped, not invented).

**Constraints can help learning (fifth, most surprising lesson).** Two runs
plateaued at 100% draws. The run that cleared the whole L1→L10 ladder was the
one where we added the **fire cone** (trigger only works ≤15° off the bow),
added for *aesthetics*, not learning. Hypothesis: gating fire on aim collapses
the exploration problem, because approach, aim, and shoot become one
reinforced chain instead of three independent discoveries.
Constraint-as-scaffold is a real phenomenon; we found it by watching the
replays and reacting to what looked wrong. (One seed: outcome is fact,
mechanism is hypothesis.)

**Curriculum forgetting (sixth lesson).** The L10-specialized final net wins
72% vs L10 but only 56% vs L1. Curricula climb; they don't retain. The fix is
a **league** (train against a pool: past checkpoints plus all scripted
levels). That's the Stage 4 design, and the same reason OpenAI Five and
AlphaStar used leagues.

**DIY exercise.** Derive the policy-gradient theorem for a 2-action bandit on
paper. Then implement REINFORCE (no critic, no clipping, ~60 lines) on
`SurviveEnv` and watch the variance problem PPO exists to solve.

---

## Stage 3: specification design, the wall-rebound episode

**Concept.** "The agent optimizes what you wrote, not what you meant":
specification gaming. Our duel agent adopted wall-rebound trajectories.
Legal, effective, ugly, and (the load-bearing detail) an artifact of a
*modeling gap*: the sim's walls were free, while Spacewar's walls
impart control-fighting spin. See it for yourself in
[`replays/v3-wallrider`](../replays/v3-wallrider), or re-train in that arena
with `rules="v3-rude"`.

**The fix pattern, in order of preference:**
1. **Fix the physics first.** Walls now impart a decaying spin
   (`SPIN_DAMP = 3.0`, matching Spacewar's angular damping) that
   fights steering, so the exploit is genuinely worse, not just taxed.
2. **Then encode the value judgment in reward.** A wall strike costs 0.8× a
   death (matching the hand-tuned bots' planning weight). The game is about
   out-orbiting, and now the reward says so. This ruleset is
   `rules="v4-honest"`, and [`replays/v4-honest`](../replays/v4-honest) shows
   the resulting flying.
3. Keep the *evaluation* metrics (win rate) unshaped, so you can still see
   what the policy actually achieves.

**Measured outcome.** Wall touches: 12.2/episode -> 0.42 (29x). Draws
vanished; clean laser kills went 15 -> 336. And the curriculum deflated from
"L10 in 6 minutes" to "L4 in 9": the exploit had been carrying the blitz.
Expect this. **Removing an exploit usually makes your agent look worse and
your experiment more true.**

**DIY exercise.** Before reading our fix: how ELSE could an agent exploit
this arena spec? (Ideas we've already seen: hole-knock kills via hit
momentum; camping outside the opponent's range gate. What would you do about
each, physics-first?)

---

## Stage 4: the league, self-play and the generalist

**Concepts.** Fictitious self-play, opponent pools, non-transitivity and
strategy cycling, prioritized opponent sampling. Read: AlphaStar league blog
(DeepMind 2019), OpenAI Five writeups; Sutton & Barto ch. 16 for TD-Gammon's
original self-play.

**What we built.** [`agents/league_duel.py`](../agents/league_duel.py):
every episode samples a fresh opponent, 60% a scripted level (weighted
toward the ones we're currently LOSING to, with a floor so nothing is
forgotten) and 40% a frozen snapshot of the agent's own past self. The eval
yardstick is a fixed scripted panel, so "climbing" can't be curriculum
relabeling.

**Results.** Stage 2's ladder-climber forgot L1 while mastering L10 (56% vs
72%). The league generalist: **98-100% vs every level, 99.0% over 2,000
formal episodes**, with a phase transition around 8k updates from a noisy
50-70% band to a sustained sweep. Lesson: **curricula climb, leagues
retain**. The mixed pool converts "beat the current teacher" into "beat
everyone I have ever met, including myself." The champion is `duel_ppo_v6`
in [`agents/models/`](../agents/models) (full ruleset: `rules="v6-full"`),
and [`replays/v6-final`](../replays/v6-final) is its formal eval.

**Also here: constraint economics as behavior design.** The power-hover habit
was removed by translating the hand-tuned bots' fuel governor (an internal
budget: burn 0.67/s vs regen 0.035/s → ~5% duty) into reward economics
(0.05/wall-s of burn). The learned pilot landed at 7-8% thrust duty, inside
the scripted bots' discipline band, without any fuel mechanic in the physics.
Corollary lesson: prices must exceed prizes. At wall penalty 0.8 < win 1.0,
the agent rationally trades a wall scrape for a win (~3 scrapes/min); the
refinement run prices walls at 1.5.

**DIY exercise.** Take your Stage 2-style trainer and add the simplest
league: keep the last 5 checkpoints, play 50% of episodes against a random
one. Compare forgetting curves (final policy vs L1) with and without the
pool.

---

## Stage 5: sim-to-real, what porting the pilot taught us

**Concepts.** Sim-to-real transfer, observation parity, inference without
frameworks.

**What we built.** The trained nets export to plain `.json` weight arrays
(the files in [`agents/models/`](../agents/models)), and a hand-written
~40-line forward pass (two tanh layers + argmax head) is all it takes to fly
one. No frameworks. In this repo that reference forward lives in
[`orbitduel/netpilot.py`](../orbitduel/netpilot.py), and
[`agents/duel_eval.py`](../agents/duel_eval.py) replays any exported `.json`
checkpoint through it, so you can check a hand-rolled port against torch
yourself. We used exactly this recipe to port the league champion into
Spacewar, the game the sim was modeled on.

**Transfer results.** The league champion, in Spacewar vs the
scripted L10: **7-4** over 150 s, despite the sim never modeling curved
lasers, hit-spin, or explosion bodies. Two measured gaps to remember: the
real L10 lands laser kills the sim's opponents never could (curve-aware lead
aim isn't in the scripted-pilot port), and in-game thrust duty ran 21% vs
the sim's 7% (different pressure, different dynamics). Sim-to-real always
costs something; measure it, don't assume it.

**The transfer-ranking lesson.** The complete-ruleset pilot (fuel-budgeted,
wall-clean, spawn-randomized) dominates in the sim with near-zero losses,
and LOST in Spacewar to the scripted L10 that the ruder v3 beat.
Constraining behavior in sim does not preserve real-world ranking: subtler
policies lean harder on sim details (here, straight lasers vs Spacewar's
curved ones). Close the model gap before tightening the style screws
further. `gravity_on_lasers=True` became the next training flag, and
`rules="v6-full"` includes it.

**Correction (2026-07-06).** The laser gap was subtler than "the sim never
modeled curved lasers": the sim already applied gravity to the *opponent's*
shots. It was the *agent's own* shots that had been overlooked and flew
straight. So the agent trained on a lie about its own shots, not about the
world in general; `gravity_on_lasers=True` closed exactly that half of the
asymmetry. (The two paragraphs above predate this correction.)

**DIY exercise.** Write the forward pass by hand from the weights JSON in
any language (C makes the later fixed-point port easy) and verify output
equality against [`orbitduel/netpilot.py`](../orbitduel/netpilot.py) on 100
random observations. Then quantize weights to int8 and measure the win-rate
cost. That curve is the whole edge-AI methodology in one plot.

---

## Standing habits worth copying

- **One readable file per algorithm.** You can't debug what you can't see.
- **Archive everything; render later.** Every eval episode is a JSON replay;
  the live dashboard ([`viz/watch.html`](../viz/watch.html)) and the video
  renderer ([`viz/render_video.py`](../viz/render_video.py)) are both just
  readers of that archive, and the curated era replays in
  [`replays/`](../replays) (open any with `viz/watch.html?dir=replays/<era>`)
  are the archive's highlight reel. Metrics without replays hide reward
  hacking.
- **Fixed eval seeds; gates decided before the run.** Moving goalposts after
  seeing results is how you fool yourself.
- **Watch the replays.** Both behavioral discoveries (spray-fire, wall
  rebounds) were found by *looking*, not by any metric.
- **Tiny nets are a feature.** Full training runs are 3-6 minutes on CPU, so
  every hypothesis gets tested the moment it's stated. That loop speed, not
  any single trick, is why this project moves.

## Why the nets stay this small

The models are deliberately in the quantize-friendly size class: a
5,600-parameter policy fits in a couple of block RAMs on an FPGA, and a GPU
shader could run it. The path from here: plain weight arrays (done, the
`.json` files in [`agents/models/`](../agents/models)) → a hand-rolled
forward pass ([`orbitduel/netpilot.py`](../orbitduel/netpilot.py), or your
own in C) → post-training int8 quantization → quantization-aware training →
a fixed-point reference. Measure win-rate at every precision step;
accuracy-vs-precision curves are the whole edge-deployment methodology,
practiced on a model small enough to inspect by eye.
