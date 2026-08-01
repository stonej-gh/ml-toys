# Stage 0 exercise: build a gravity sim, then break it

The back of the book for the stage 0 exercise in
[docs/LEARNING-NOTES.md](../../docs/LEARNING-NOTES.md): write a small gravity
sim, check Kepler's third law against it, then swap the integrator and watch
energy conservation fall apart. The sample solution is
[solution.py](solution.py), about 100 lines, standard library only.

```
python exercises/stage0-gravity/solution.py           # the working sim
python exercises/stage0-gravity/solution.py --euler   # break it on purpose
```

## The walk

**1. The world is two functions.** Gravity toward a fixed center, with
strength that falls off as one over distance squared, is four lines
(`accel`). Advancing time is three more (`step_semi_implicit`). Everything
else in the file is measurement.

**2. Pick the step order carefully; it is the whole exercise.** Both
integrators do "add acceleration to velocity, add velocity to position". The
only difference is which copy of the velocity moves the ship:

- *Semi-implicit Euler:* update velocity first, then move with the NEW
  velocity.
- *Explicit Euler:* move with the OLD velocity, then update it.

They look interchangeable. They are not, and the exercise exists so you feel
that once and never forget it.

**3. Check Kepler before trusting anything.** Kepler's third law says the
square of an orbit's period grows with the cube of its size. So: start
circular orbits at four radii, time one full trip each (`measure_period`
adds up swept angle until it reaches a full turn), and fit the slope of log
period against log radius. The solution measures **1.5000** where the law
says 1.5, and the ratio T²/r³ agrees to four digits across all four radii.
Two different orbits, one constant: that is what "the sim obeys the law"
looks like as a number.

**4. Now break it.** Rerun with `--euler`. Same code path, same timestep,
velocity applied one half-step later. Measured over 20 orbit-times on an
eccentric orbit:

| | net energy drift | final radius (start 1.0) |
|---|---|---|
| semi-implicit | **+0.012%** | 0.96 |
| explicit Euler | **+32.07%** | 1.53 |

The explicit version pumps energy every step, so the ship spirals outward
and the "orbit" is a slow escape. Nothing crashes, nothing warns; the
answers are just wrong. This is why the arena's real integrator
([orbitduel/physics.py](../../orbitduel/physics.py)) is semi-implicit.

**5. The subtle part: wobble is not drift.** Look at the two numbers the
solution prints for the good integrator: the energy *wobbles* inside a band
of about 0.1% over each orbit, but the *net* drift after 20 orbits is a
hundred times smaller than the band. Semi-implicit Euler never conserves
energy exactly at any instant; it conserves it on average, forever. So when
you write tests for a sim like this, test for one-way drift over many
orbits, and give the per-orbit wobble some tolerance. A test that demands
exact energy equality every step will fail a perfectly good integrator.
That is the same lesson [orbitduel/selftest.py](../../orbitduel/selftest.py)
encodes for the real arena.

## Where to go from here

Make the timestep ten times bigger and rerun both. Then give the ship a
thruster (a constant push along the velocity direction while a flag is on)
and watch what a burn does to the orbit; you have now built the survive
task's world, which is where [stage 1](../stage1-dqn/README.md) picks up.
