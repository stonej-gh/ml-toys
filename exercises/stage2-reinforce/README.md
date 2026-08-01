# Stage 2 exercise: the policy gradient on paper, then in 60 lines

The back of the book for the stage 2 exercise in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md): derive the policy
gradient for a two-action bandit with a pencil, then implement REINFORCE
(the policy gradient with nothing else attached) and watch the variance
problem that PPO exists to solve. The sample solution is
[solution.py](solution.py), about 60 lines of algorithm.

```
python exercises/stage2-reinforce/solution.py                # ~3,000 episodes
python exercises/stage2-reinforce/solution.py --episodes 500 # a quick look
```

## Part one: the derivation, with a pencil

Take the smallest possible problem: two actions, A and B. Pulling A pays
out 1 point with some probability; B pays with a different probability.
The policy is one number, p = the probability of choosing A, and we want
to know how expected payout changes as p changes.

Expected payout: J = p·(payout of A) + (1 − p)·(payout of B).

The derivative is just: dJ/dp = payout of A − payout of B. Sensible: if A
pays better, raising p helps, in exact proportion to the gap.

Now the move that makes it an algorithm. We cannot see the payout gap
directly; we can only pull arms and watch. The trick (worth doing slowly on
paper once in your life) is to rewrite the derivative as an average over
what we already do:

    dJ/dp is the AVERAGE, over our own pulls, of
    (payout received) x (d log(probability of the pull) / dp)

Check it for the bandit: choosing A contributes payout·(1/p), choosing B
contributes payout·(−1/(1−p)), and averaging over how often each happens
gives back exactly (payout of A − payout of B). That identity is the
policy-gradient theorem in miniature: **run your own policy, then weight
each choice's "make me more likely" direction by the reward that followed
it, and the average points uphill in expected reward.** No model of the
world required, no answer key, just your own history and the scores.

The full theorem is the same statement with states in it and the payout
replaced by the return. Sutton & Barto chapter 13 has the general proof;
every term in it now has a bandit-sized shadow you have already computed.

## Part two: REINFORCE, and what it measures

The solution is the theorem typed in: play one episode of the survive task,
compute each step's discounted return-to-go, and use
`-log_prob x return` as the loss so the optimizer pushes uphill. Everything
else in the file is evaluation.

Measured over 3,000 episodes (evaluated greedily on 50 fixed seeds every
250): the eval median lifetime bounces between **1.8 s and 6.7 s** and ends
at 2.3 s; survival flickers between 0% and 16% with no steady climb. For
comparison, stage 1's DQN reaches a 60 s median with 70% survival inside
75,000 steps, roughly the same amount of experience.

That bouncing is not a bug in your code (well, verify with the sample, but
it probably is not). It is the variance of the estimator. One episode's
return is a wildly noisy read on how good the policy is, so every update is
a big step in a half-random direction. The lesson to carry out of the
exercise: **the policy-gradient idea is sound and five lines long, and
everything else in a modern trainer exists to cut the noise**: a learned
baseline so actions are judged against expectation (the critic), advantage
estimation to shorten the credit horizon (GAE), and a clip so no single
noisy batch can wreck the policy (PPO). Open
[agents/ppo_duel.py](../../agents/ppo_duel.py) after this and you can now
name the reason each of those parts is there.

## Where to go from here

Add the pieces one at a time to your own copy and measure each: first
collect a batch of episodes per update instead of one, then subtract a
learned value estimate before weighting. You are rebuilding the road from
1992's REINFORCE to today's default algorithm, and your eval curve will
tell you which mile of it bought the most.
