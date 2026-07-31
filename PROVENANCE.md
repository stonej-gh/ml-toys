# Provenance

Everything in this repository was developed independently: on my own time, on
my own hardware, with my own tool licenses, and with no inputs from any
employer or client. It is my work, and I release it under the MIT terms in
[LICENSE](LICENSE).

The environment physics and the recorded gameplay traces derive from a
proprietary iOS game of my own. The game's Swift code, art, and audio are
copyright Jeffrey Stone, all rights reserved, and none of them are part of
this repository. What is here is an independent Python implementation of the
same physics, plus models trained against it, and all of it is MIT.

## Physics fidelity

`orbitduel/physics.py` carries the game's own algorithm rather than
approximating it. The field-strength constant comes from an on-device orbit
fit (v0^2 * r0), re-confirmed on tablet hardware to about 1%, and the same
constant sets the spawn and muzzle speeds so field, spawn, and muzzle stay
consistent. The two hull collision radii (37.52 and 27.16 pt) are the values
measured at physics-body build in the running game, and the thrust
acceleration (666.67 pt/s^2) is cross-checked against position telemetry
(about 667.4).

Deliberately not modeled, so nothing here over-claims: winged versus center
laser hits (every hit costs 1 of 5 damage, matching the game's current rule),
explosion debris, ship-ship collision, edge friction, the human ship's 0.3 s
thrust ramp (the AI has none), and the laser's true thin-rect body (a circle
here). SPIN_KICK and HIT_SPIN are fitted stand-ins for engine contact
dynamics, not measured values.
