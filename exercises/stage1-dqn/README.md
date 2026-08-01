# Stage 1 exercise: build a DQN, then watch it fall apart

The back of the book for the stage 1 exercise in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md): reimplement DQN
against the survive task, get it working, then remove the Double DQN fix and
turn the learning rate up to reproduce the collapse. The sample solution is
[solution.py](solution.py), about 150 lines against
[orbitduel/survive.py](../../orbitduel/survive.py).

```
python exercises/stage1-dqn/solution.py                     # the stable recipe
python exercises/stage1-dqn/solution.py --vanilla --lr 1e-3 # the collapse
```

Each run is a few minutes on a laptop CPU. New words along the way are in the
[glossary](../../docs/GLOSSARY.md); the survive task itself is packaged as
[experiment 01](../../experiments/01-survive/README.md), whose "Break it"
section is this same exercise applied to the full-size trainer.

## The walk

**1. Five parts, and every one is short.** A Q-network (three `nn.Linear`
lines), a replay memory (a list used as a ring), epsilon-greedy exploration
(one `if random() < eps`), a frozen target network (a copy, synced every
4,000 steps), and the update (build a target value, regress toward it).
That is the entire 2015 DeepMind recipe. If your version is much longer than
150 lines, something snuck in.

**2. The one line beginners get wrong.** When an episode ends because the
clock ran out, the ship did NOT die; there is future reward beyond the cut,
and the target must still include the next state's value. The solution
stores `term` (died), never `term or trunc` (died or timed out), in the
replay buffer. Get this wrong and the agent learns that surviving to the
end of an episode is as bad as falling into the hole.

**3. The Double DQN fix is two lines.** Plain DQN prices the next state with
`tgt(nxt).max(1)`: the target network both picks the best next action and
prices it, and a max over its own noisy estimates is always optimistic.
Double DQN splits the jobs: the online net picks (`argmax`), the target net
prices (`gather`). That is the whole fix.

**4. What the stable recipe measures.** Double DQN at learning rate 5e-4,
400,000 steps, evaluated greedily on 50 fixed seeds every 25,000:

- learns the rescue by step 75,000 (median lifetime 60 s, survival 70%),
- wobbles mid-run (a rough patch near step 275,000 dips to 0%),
- recovers, and finishes at its best: **86% survival at step 375,000**.

Even the good recipe is not a smooth ride at this size, which is why the
run's last line matters more than any single eval.

**5. Now the collapse.** Vanilla target, learning rate 1e-3. The run
*learns faster at first* (84% survival by step 125,000, better than the
stable recipe at the same point) and that is the seductive part. Then it
spends the rest of the run falling apart and half-recovering: 30%, 54%,
18%, 52%, 18%, and it never sustains its peak again. Best was step 125,000;
the final net is 20 points worse. The textbook name for the risk zone is
the deadly triad, and you just watched it: a net learning from its own
guesses, on replayed old data, with a step size big enough that the errors
feed back faster than they wash out.

**6. The embarrassing lesson is the durable one.** Look at the last two
lines either run prints. If you only save the final network, the vanilla
run hands you its worst self and throws its best away. Save the best
checkpoint as you go, ranked by the evaluation, not by training reward.
(Then read [stage 6](../stage6-selection/README.md) for the bias hiding
inside "save the best", and the second sample that fixes it.)

## Where to go from here

Lower the vanilla run's learning rate back to 5e-4 and see how much of the
instability was the rate rather than the missing fix; then re-add Double DQN
at 1e-3 and see how much was the fix rather than the rate. Two runs, and you
have a 2x2 ablation table of your own.
