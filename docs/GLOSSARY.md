# Glossary

Plain-language definitions for the jargon this repo uses, each with a pointer
to the file where the idea lives in code. Entries are grouped into two ladders
(the pilot's, then the spotter's) with a shared section on training between
them, and each ladder is ordered so that every entry leans only on the ones
above it. Read a ladder top to bottom and you have a working vocabulary; or
arrive here by a link, read one entry, and click through to the code.

New to all of it? [START-HERE.md](START-HERE.md) is the friendlier front door;
this page is the reference.

## The pilot's ladder: reinforcement learning

### Reinforcement learning

Learning by trial and error. A piece of software (the agent) tries actions in
a world, receives a score back, and gradually shifts toward the actions that
score well. Nobody ever shows it a correct answer; it finds behavior by
exploring. In this repo the world is a two-ship orbital duel, and every agent
in [`agents/`](../agents) learned to fly it from scratch.

### Environment

The world the agent lives in, wrapped in a standard interface: `reset()`
starts an episode and returns the first observation; `step(action)` advances
one tick and returns the new observation, the reward, and whether the episode
ended. Gymnasium is the Python convention for that interface, and it is the
whole contract between an agent and its world. In this repo:
[`orbitduel/env.py`](../orbitduel/env.py) defines the duel (`OrbitDuel-v0`)
and [`orbitduel/survive.py`](../orbitduel/survive.py) the one-ship practice
task.

### Observation

What the agent gets to see each step: a short list of numbers describing the
world from its seat. Not pixels here; things like "my radius", "my speed
along the orbit", "which way my nose points". One warning for this repo: the
survive task's observation is 8 numbers and the duel's is 19, so the same
word names two different vectors. The exact meaning of every element is
documented where each is built, in
[`orbitduel/survive.py`](../orbitduel/survive.py) and `obs_from` in
[`orbitduel/env.py`](../orbitduel/env.py).

### Action

What the agent is allowed to do, as a fixed menu it picks from once per
decision. The survive task has 6 actions (turn left/straight/right, thrust
on/off); the duel has 12 (the same, plus fire on/off). The menus are the
`ACTIONS` tables at the top of [`orbitduel/survive.py`](../orbitduel/survive.py)
and [`orbitduel/env.py`](../orbitduel/env.py).

### Reward

The score signal: one number per step, positive for things you want and
negative for things you don't. It is the only channel through which the
designer says what the task IS, which makes reward design the most
error-prone part of the field. This repo's reward went wrong four instructive
times; [REWARD-SPEC.md](REWARD-SPEC.md) tells that story with every number.

### Episode

One round, from `reset()` to an ending: the ship dies, the duel resolves, or
time runs out. Training is thousands of episodes; evaluation is a fixed panel
of them.

### Terminated vs truncated

Two different ways an episode can end. Terminated means the task itself ended
(the ship fell into the hole). Truncated means the clock ran out while the
task was still going. Trainers must treat them differently: after a true
ending there is no future reward to estimate, but after a time-out there is
(see [bootstrapping](#bootstrapping)). The comment at the `buf.push` call in
[`agents/dqn_survive.py`](../agents/dqn_survive.py) is this distinction in
one line.

### Neural network

A stack of layers, each doing the same simple thing: multiply the incoming
numbers by learned weights, add them up, and pass the result through a bend
such as [ReLU](#relu) (the bend is what makes stacking layers worth more than
one big multiply). The final layer's raw outputs are called logits, one per
action, and picking the largest is called argmax. Learning means nudging the
weights so good outputs become more likely. The clearest place in this repo
to see one work is `_forward` in
[`orbitduel/netpilot.py`](../orbitduel/netpilot.py): a complete
neural-network inference in 13 lines of dependency-free Python.

### Discount

Future reward counts slightly less than immediate reward, by a factor gamma
per step (0.99 for the survive DQN, 0.995 for the duel). It keeps totals
finite and encodes mild impatience.
`GAMMA` in [`agents/dqn_survive.py`](../agents/dqn_survive.py).

### Return and value

The return is the total discounted reward from this moment to the end of the
episode. The value of a situation is the return you expect from it, and a
value estimate is a net's learned guess at that number: "how good is my
position, in points". Most of deep RL is machinery for learning value
estimates you can trust.

### Q-value

The value of a specific action in a specific situation: "if I take this
action now and play on as usual, what return do I expect". A Q-network takes
an observation and outputs one Q-value per action on the menu; acting
greedily means taking the argmax. The `QNet` class in
[`agents/dqn_survive.py`](../agents/dqn_survive.py) maps 8 observation
numbers to 6 Q-values.

### Epsilon-greedy

The standard exploration recipe: act greedily most of the time, but with
probability epsilon pick a random action instead. Epsilon starts at 1.0 (all
random) and decays to a small floor, so early training explores and late
training exploits. The three `EPS_*` constants in
[`agents/dqn_survive.py`](../agents/dqn_survive.py) define the schedule.

### Replay buffer

A large memory of past steps, stored as (observation, action, reward, next
observation, ended). Each update trains on a random batch drawn from it,
rather than on the most recent steps, which breaks the correlation between
consecutive frames and lets every experience be reused many times. The
`Replay` class in [`agents/dqn_survive.py`](../agents/dqn_survive.py) is a
ring buffer holding 200,000 steps.

### Bootstrapping

Learning a guess from a guess. The training target for "Q-value now" is
built from the observed one-step reward plus the net's own estimate of the
next state's value (the temporal-difference target). It is what lets an agent
learn long tasks from single steps, and it is also a known source of
instability, because the target moves whenever the net does.

### Target network

A frozen copy of the Q-network used only to compute training targets, synced
to the live net every few thousand steps (`TARGET_SYNC` in
[`agents/dqn_survive.py`](../agents/dqn_survive.py)). Without it,
[bootstrapping](#bootstrapping) chases a target that shifts on every update.

### DQN

Deep Q-Network: the 2015 DeepMind recipe that made Q-learning work with
neural networks by combining a Q-network with a [replay buffer](#replay-buffer)
and a [target network](#target-network).
[`agents/dqn_survive.py`](../agents/dqn_survive.py) is the whole algorithm in
one readable file, trained on the survive task.

### Double DQN

A one-line repair to DQN's known bias. Training targets take a max over the
net's own noisy Q-values, and a max over noise is optimistic, so values
inflate until learning can collapse (this repo's vanilla run learned the
skill, then fell apart; the textbook name for the risk zone is the "deadly
triad": [bootstrapping](#bootstrapping), function approximation, meaning a
net standing in for a lookup table, and off-policy replay, meaning learning
from actions the current policy would no longer take).
Double DQN splits the roles: the online net picks the next action, the target
net prices it. The fix is two lines above the loss in
[`agents/dqn_survive.py`](../agents/dqn_survive.py), and the collapse it
cures is reproducible from
[LEARNING-NOTES.md](LEARNING-NOTES.md) Stage 1.

### Policy gradient

The other big family. Value methods like [DQN](#dqn) learn scores and act by
argmax; policy-gradient methods learn the behavior directly, as a probability
for each action, and nudge probabilities up for actions that led to
better-than-expected outcomes. Smooth, statistical, and the natural fit when
you want stochastic play.

### Actor-critic

A policy-gradient net with two heads on one trunk. The actor holds the policy
(action probabilities); the critic holds a value estimate used as the
baseline that defines "better than expected". Both halves of
[`agents/ppo_duel.py`](../agents/ppo_duel.py)'s network are visible in its
model class.

### Advantage

How much better an action worked out than the critic expected, in points.
Positive advantage nudges the action's probability up, negative down; the
size of the nudge scales with the size of the surprise.

### GAE

Generalized advantage estimation: a way of computing [advantage](#advantage)
that blends one-step and many-step views of the outcome, trading a little
bias for much less noise. In practice it is what makes policy gradients train
smoothly. Implemented in the update loop of
[`agents/ppo_duel.py`](../agents/ppo_duel.py).

### PPO

Proximal Policy Optimization, the field's default policy-gradient algorithm.
Its trademark is the clipped objective: each update is only allowed to move
action probabilities a small ratio away from the policy that collected the
data, so one enthusiastic step cannot wreck the policy.
[`agents/ppo_duel.py`](../agents/ppo_duel.py) is a single-file PPO that
learned the duel through a curriculum.

### Entropy bonus

A small reward for keeping the policy's probabilities spread out. It delays
premature certainty, which keeps exploration alive early in training.
`ENT_COEF` in [`agents/ppo_duel.py`](../agents/ppo_duel.py) sets it.

### Reward shaping

Adding guidance terms to the reward so learning finds the goal sooner.
Dangerous by default, because agents optimize what you wrote instead of what
you meant. One form is provably safe, and it is easier than its formula:
give the agent a running meter (for example, minus the distance to the
opponent) and pay only for *changes* in the meter. Written out, the bonus is
gamma times phi(next) minus phi(now), where phi is the meter; a bonus of
exactly that shape cannot change which policy is optimal, only how quickly
it is found. [REWARD-SPEC.md](REWARD-SPEC.md) shows it used against the
draw-camper exploit.

### Reward hacking

Also called specification gaming: the agent finds a high-scoring behavior the
designer never intended, exploiting the letter of the reward against its
spirit. This repo's wall-riding champion is a preserved specimen, and
[experiments/02-reward-hacking](../experiments/02-reward-hacking/README.md)
lets you re-create it live in about a minute of CPU.

### Curriculum

Training against a ladder of opponents, easiest first, advancing on wins.
Curricula climb fast and forget what they left behind: this repo's
ladder-climber beat the top scripted bot while losing ground against the
easiest one. [experiments/03-forgetting](../experiments/03-forgetting/README.md)
measures the effect in a matrix of exact numbers.

### League

The fix for curriculum forgetting: train against a pool holding every
scripted opponent plus frozen snapshots of the agent's own past selves
(self-play). Strategy quality is not a single ladder (rock beats scissors
beats paper), so retention has to be trained, and the pool converts "beat the
current teacher" into "beat everyone I have ever met, including myself".
[`agents/league_duel.py`](../agents/league_duel.py) built this repo's
champion.

### Sim-to-real

Moving a policy trained in simulation into the world it models. Every gap in
the simulator becomes a gap in behavior: for several eras these agents' own
shots flew straight, and their learned aim collapsed wherever lasers curve.
[LEARNING-NOTES.md](LEARNING-NOTES.md) Stage 5 gives the measured transfer
results, and
[experiments/05-curved-lasers](../experiments/05-curved-lasers/README.md)
isolates the aim-model effect.

## Shared: how training works

### Loss

A single number measuring how wrong the network currently is on a batch of
examples; training is the process of making it smaller. Different jobs use
different losses: the DQN uses a Huber loss on Q-value errors, the spotter a
class-weighted cross-entropy on pixel labels.

### Optimizer

The routine that actually adjusts the weights, following the loss downhill
(the pilots train with Adam, the spotter with its close cousin AdamW). Its
main knob, the learning rate, sets
the step size, and it is a real knob: the survive DQN at learning rate 1e-3
learned the task and then collapsed; at 5e-4 it stayed stable. The whole
story is in [LEARNING-NOTES.md](LEARNING-NOTES.md) Stage 1.

### Epoch and batch

A batch is the handful of examples averaged for one weight update (128
replay steps for the DQN; a few frames for the spotter). An epoch is one full
pass through a fixed dataset, which is supervised-learning vocabulary: the
spotter trains in epochs, while the DQN has no dataset to pass through, only
the stream of its own experience.

### Overfitting

Memorizing the training examples instead of learning the pattern, which shows
up as great training scores and poor scores on anything new. The standard
defenses appear on the spotter side: held-out validation data to detect it,
and [augmentation](#augmentation) to resist it.

## The spotter's ladder: vision

### Images as numbers

To a program, an image is a grid of numbers: height by width, three numbers
per pixel (red, green, blue). The spotter's input is a 320 by 192 by 3 grid.
Everything a vision net does is arithmetic on that grid.

### CNN

Convolutional neural network: the standard net shape for images. Instead of
connecting every pixel to every output, its layers slide small reusable
pattern detectors across the grid (see [convolution](#convolution)), so the
same detector serves every location and the parameter count stays small.
[`spotter/model.py`](../spotter/model.py) defines this repo's 28,126-parameter
CNN.

### Convolution

The sliding-stencil operation. A kernel is a small grid of weights, 3 by 3
here; lay it on the image, multiply corresponding numbers, add them up, and
that is one output pixel; slide one step and repeat. One kernel learns to
detect one local pattern (an edge, a bright dot, a color transition), and a
layer runs many kernels at once.

### Feature map

The output grid a kernel produces: high values where its pattern is present,
low where it is absent. A conv layer's output is a stack of feature maps,
one per kernel, and the stack's depth is called its channels. Deeper layers
run their kernels over earlier feature maps, so later maps respond to
patterns of patterns.

### Stride

How far the stencil steps between applications. Stride 1 keeps the map the
same size; stride 2 halves its width and height, which is how CNNs shrink
maps as they deepen (downsampling). The spotter's trunk downsamples three
times, so its deepest maps are 1/8 scale.

### ReLU

The standard bend between layers: negative values become zero, positive pass
through. Cheap, quantization-friendly, and all this repo's nets use it (the
exported duel policies may use tanh, an S-shaped alternative; the export file
records which).

### Receptive field

How much of the input image one output pixel can see, through the stack of
stencils below it. It grows with every layer and stride. It is load-bearing
here: a net trained on isolated 32-pixel patches saw zero-padding where a
full frame would supply real neighbors, and collapsed when run on full
frames. The docstring of
[`spotter/train_heatmap.py`](../spotter/train_heatmap.py) records the lesson.

### Pooling

Shrinking a map by summarizing windows, for instance averaging each 4 by 4
block. The spotter's patch head ends in exactly that pool.

### Skip connection

A shortcut that adds an early, high-resolution feature map into a later,
upsampled one. Downsampling buys context but blurs detail; the skip hands the
detail back. The spotter's decoder adds three of them, and they were the
difference between mushy and sharp small-object masks
([`spotter/model.py`](../spotter/model.py), `Decoder`).

### Segmentation

Labeling every pixel with a class, rather than giving one label for the whole
image. The spotter's output assigns each of the 61,440 input pixels one of
five classes: background, interceptor, fighter, laser, or hole.
[SPOTTER-DESIGN.md](SPOTTER-DESIGN.md) defines the classes and the numbers.

### Mask

The answer key for segmentation: an image the same size as the input whose
every pixel holds the true class id. Labeling masks by hand is the expensive
part of vision work. This repo skips it: the renderer draws every entity
twice, shaded pixels into the frame and class ids into the mask, so labels
are free and exact by construction
([`spotter/render.py`](../spotter/render.py)).

### Dataset splits

Train, validation, and test: the data the net learns from, the data that
steers decisions (which checkpoint to keep), and the data touched only at
the end for an honest score. The spotter's splits use disjoint random-seed
blocks, so no frame can leak between them
([SPOTTER-DESIGN.md](SPOTTER-DESIGN.md), Dataset).

### Class imbalance

When classes are wildly unequal in frequency. Background pixels outnumber
laser pixels by roughly a thousand to one here, so an unweighted net can
score 99% by answering "background" everywhere. The fix in this repo is a
class-weighted loss, and the honest metric is [IoU](#iou) instead of
accuracy.

### Augmentation

Randomly jittering training inputs (brightness, per-channel gain, noise)
while leaving labels alone, so the net learns the pattern rather than the
exact pixels. Aimed at a simulator-vs-reality gap it is called domain
randomization. The spotter's photometric augmentation lives in
[`spotter/train_dense.py`](../spotter/train_dense.py).

### Fine-tuning

Starting a new training run from already-trained weights instead of from
random ones. The spotter's dense model fine-tunes from the patch model's
trunk, keeping what the trunk already knows about the arena's shapes.

### IoU

Intersection over union, the standard overlap score for masks: the area where
prediction and truth agree, divided by the area covered by either. It runs
from 0 (no overlap) to 1 (perfect), and unlike accuracy it cannot be gamed by
[class imbalance](#class-imbalance). The spotter's quality gates are IoU
thresholds ([SPOTTER-DESIGN.md](SPOTTER-DESIGN.md), Verification ladder).

### Train vs inference

Training needs gradients, an optimizer, and a framework like PyTorch.
Inference, actually using the trained net, is only the forward arithmetic,
and can run anywhere: this repo re-implements both nets' inference with no
framework at all (pure Python for the pilot in
[`orbitduel/netpilot.py`](../orbitduel/netpilot.py), numpy for the spotter in
[`deploy/`](../deploy)), which is what makes the results portable and
checkable.

### BatchNorm folding

BatchNorm is a layer that stabilizes training by normalizing activations,
but at inference it collapses to a fixed multiply-and-add per channel, which
can be folded into the neighboring conv's weights. The deployed spotter has
no BatchNorm layers at all; [`spotter/export.py`](../spotter/export.py) does
the folding and documents the algebra.

### Quantization

Storing weights and activations as 8-bit integers instead of 32-bit floats:
four times smaller, and the arithmetic becomes integer multiply-adds that
tiny chips do well. Done after training (post-training quantization), it
needs calibration: measuring activation ranges on sample inputs to choose the
int8 scales. The cost here is measurably tiny: across the spotter's eight
recorded golden frames, float and int8 disagree on 9 of 491,520 pixels, about
one pixel per frame.
[`spotter/quantize.py`](../spotter/quantize.py) documents the whole scheme.

### Golden bundle

A frozen package of weights, seeded input vectors, and recorded correct
outputs, shipped together so that any re-implementation can prove itself by
reproducing the recorded answers exactly. It is this repo's definition of
"deployed": [`deploy/verify.py`](../deploy/verify.py) checks the float path
to tolerance and the int8 path bit for bit, and a hardware port passes by
running the same script unchanged.

---

Where next: [LEARNING-NOTES.md](LEARNING-NOTES.md) retells the project as
six stages built from these terms, and the
[experiments](../experiments/README.md) put them to work.
