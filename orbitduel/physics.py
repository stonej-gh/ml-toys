# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Headless 2D orbital-duel physics: two ships, one gravity well, lasers.

The arena is a rectangle (768 pt reference height) around an inverse-square
gravity well whose event horizon captures on contact. Ships are circles with a
nose direction; thrust is a fixed acceleration along the nose; lasers inherit
the firer's velocity plus a muzzle kick, curve in the field, and live for a
fixed time. Integration is semi-implicit (symplectic) Euler at DT = 1/120
physics seconds, the scheme common 2D engines use, so orbits wobble ~1%
around an invariant circle but never secularly drift.

These values ARE the proprietary arena game's algorithm (see PROVENANCE.md),
carried over exactly rather than approximated: the field is the game's
inverse-square law (strength constant pinned by an on-device orbit fit and
re-confirmed on tablet hardware to about 1%), the spawn and muzzle speeds use
the game's own code formulas, and the two hulls carry their true, DIFFERENT
collision radii as measured at physics-body build in the running game. Where
this file still approximates (documented below), it says so.

TWO FIELDS. The game plays on two geometries: the phone field (the default
here) and the tablet field, a 1.67x zoomed-out arena with a tapered event
horizon and a faster physics clock. Set ORBITDUEL_FIELD=tablet BEFORE
importing to switch the module; ORBITDUEL_TABLET_ASPECT overrides the tablet
screen aspect (default 1180/820). Tablet units divide raw scene units by
1.67 so H stays 768; times and angles are untouched. A policy meant for one
field should be trained on that field's profile.

Two clocks appear throughout: PHYSICS time (integration) and WALL time (what
a player sees); the game runs physics at GAME_SPEED per wall second, so wall
seconds = physics seconds / GAME_SPEED. Units are points (pt) and physics
seconds unless a comment says otherwise.

Deliberately not modeled (documented so nothing over-claims): winged vs
center laser hits (every hit costs 1 of MAX_DAMAGE, matching the game's
current damage rule), explosion debris bodies, ship-ship collision, edge
friction, the human ship's 0.3 s thrust ramp (AI thrust has none), and the
laser's true thin-rect body (a circle here). SPIN_KICK and HIT_SPIN remain
fitted stand-ins for engine contact dynamics.
"""

from dataclasses import dataclass, field
import math
import os

# --- field profile -------------------------------------------------------------
FIELD_SCALE = 1.0                          # 1.0 phone / 1.67 every tablet
_ASPECT = 19.5 / 9.0                       # tall phone; the 16:9 cap engages
if os.environ.get("ORBITDUEL_FIELD") == "tablet":
    FIELD_SCALE = 1.67
    _ASPECT = float(os.environ.get("ORBITDUEL_TABLET_ASPECT", 1180.0 / 820.0))

# --- constants at reference field scale, physics time --------------------------
FPS = 60.0                            # decision/render rate (frames per wall s)
# Physics clock: 0.5 on the phone; the tablet holds the same wall-clock pace
# on its bigger orbit (Kepler: period grows r^1.5) minus a 20% calm-down the
# game applies to tablet live play: 0.5 * 1.67^1.5 * 0.80 = 0.863246.
GAME_SPEED = 0.5 * pow(FIELD_SCALE, 1.5) * (0.80 if FIELD_SCALE > 1 else 1.0)
DT = GAME_SPEED / FPS                 # physics seconds advanced per frame
H = 768.0                             # reference field height (pt)
MAX_PLAY_ASPECT = 16.0 / 9.0          # arena width cap (phones are wider)
PLAY_W = min(_ASPECT, MAX_PLAY_ASPECT) * H    # 1365.33 phone / 1105.17 tablet
# The one gravity constant: the game's inverse-square field, pinned by the
# on-device orbit fit v0^2 * r0 (also the game's own spawn/muzzle constant, so
# field, spawn, and muzzle stay exactly consistent). Distances here divide by
# FIELD_SCALE at unchanged time, so GM divides by FIELD_SCALE^3.
GM = 298.0 * 298.0 * 226.0 / FIELD_SCALE ** 3     # pt^3/s^2
# Event horizon: the tablet field tapers the hole toward the ships (a square-
# root law) so it doesn't loom over the wider arena: 68 / sqrt(FIELD_SCALE).
HOLE_R = 68.0 / math.sqrt(FIELD_SCALE)            # 68.0 phone / 52.62 tablet
SPAWN_R = (HOLE_R + H / 2.0) / 2.0    # spawn-orbit radius: 226.0 / 218.31
SPAWN_V = math.sqrt(GM / SPAWN_R)     # circular speed at spawn: 298.0 / 140.49
SHIP_MASS = 0.03
# The two hulls have DIFFERENT collision circles (measured at physics-body
# build in the running game; identical raw values on both devices): the
# learning agent's seat flies the wider fighter hull, the opponent seat the
# slimmer interceptor. The old single 27.0 under-modeled the agent's own
# hole-capture distance by ~10 pt.
SHIP_R_AGENT = 37.52 / FIELD_SCALE    # fighter hull (ship index 0)
SHIP_R_OPP = 27.16 / FIELD_SCALE      # interceptor hull (ship index 1)
SHIP_R = SHIP_R_AGENT                 # conservative single-radius fallback
                                      # (wall margins, older callers)
THRUST_ACCEL = (20.0 / SHIP_MASS) / FIELD_SCALE
                                      # 666.67 pt/s^2 along the nose (verified
                                      # 667.4 by double-differentiating
                                      # position telemetry, IQR [663, 669])
TURN_RATE = 4.0                       # RL agent's discrete turn rate (rad/phys s)
MAX_TURN_RATE = 4.0 * math.pi / GAME_SPEED
                                      # scripted pilot's slew cap: 2 rev/wall s
HIT_KICK = 0.003 * 5.0 / SHIP_MASS    # laser hit: dv = laser velocity x 0.5
                                      # (the game's contact impulse, transfer 5
                                      # on a bullet of mass ship/10) - exact
RESTITUTION = 0.2                     # wall-bounce velocity retention
SPIN_DAMP = 3.0                       # uncommanded-spin decay rate (/s)
SPIN_KICK = 0.02                      # wall scrape: spin += -v_along_wall x this
HIT_SPIN = 14.0                       # rad/s of hit-spin at a full-lever strike
                                      # (contact-point impulse collapsed to a
                                      # single scalar; a fitted stand-in)
LASER_SPEED = 2.1 * SPAWN_V           # muzzle speed relative to the firer: the
                                      # game's own 2.1 x circular-speed formula
LASER_TTL = 1.4 * GAME_SPEED          # 1.4 WALL s of flight
LASER_R = 4.0                         # collision circle for a slim round
# Laser bodies spawn slightly ahead of the ship center, per hull (the game
# builds the round's body just behind the drawn beam at the nose).
LASER_OFF_AGENT = (22.0 - 5.0) * 0.84 / FIELD_SCALE   # fighter: 14.28
LASER_OFF_OPP = (26.0 - 5.0) * 0.84 / FIELD_SCALE     # interceptor: 17.64
FIRE_COOLDOWN = 0.3 * GAME_SPEED      # base fire cadence (0.3 wall s)
MAX_DAMAGE = 5                        # hits per life


def gravity_accel(x, y, gravity_on=True):
    """Inward inverse-square acceleration of the central field at (x, y)."""
    if not gravity_on:
        return 0.0, 0.0
    r2 = x * x + y * y
    r = math.sqrt(r2) or 1e-9
    a = -GM / (r2 * r)      # -GM/r^2 * (unit radial)
    return a * x, a * y


@dataclass
class Ship:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0    # radians; 0 = nose along +y
    spin: float = 0.0       # uncommanded angular rate (wall scrape), decays
    damage: int = 0
    alive: bool = True
    cooldown: float = 0.0
    thrusting: bool = False # set by the last step's action (for replays)
    r: float = SHIP_R_AGENT       # collision-circle radius (per hull; see reset)
    laser_off: float = LASER_OFF_AGENT   # muzzle body offset ahead of center

    def nose(self):
        """Unit vector along the nose (heading 0 fires +y)."""
        return -math.sin(self.heading), math.cos(self.heading)

    def speed(self):
        return math.hypot(self.vx, self.vy)

    def radius(self):
        return math.hypot(self.x, self.y)


@dataclass
class Laser:
    x: float
    y: float
    vx: float
    vy: float
    owner: int
    ttl: float = LASER_TTL


def _spawn_ships():
    """Both ships on the 180-degrees-apart spawn orbit, same CCW sense.
    Seat 0 = the learning agent = the fighter hull; seat 1 = the opponent =
    the interceptor hull (the seats the game deals them in regular play)."""
    return [Ship(x=SPAWN_R, y=0.0, vy=SPAWN_V, heading=0.0,
                 r=SHIP_R_AGENT, laser_off=LASER_OFF_AGENT),
            Ship(x=-SPAWN_R, y=0.0, vy=-SPAWN_V, heading=math.pi,
                 r=SHIP_R_OPP, laser_off=LASER_OFF_OPP)]


@dataclass
class Arena:
    """The two-ship duel field. step() advances one 60-fps frame (DT physics s).

    spin_kick / hit_spin default to the full modern rules; the era presets in
    env.RULE_PRESETS zero them to reproduce earlier phases of the experiment
    (free walls, no hit-spin) for the reward-hacking case study.
    """
    gravity_on_lasers: bool = True    # the real field pulls on lasers too; the
                                      # era presets turn it off for the
                                      # straight-laser phases (pre curve-reader)
    spin_kick: float = SPIN_KICK      # wall-scrape spin transfer (0 = free walls)
    hit_spin: float = HIT_SPIN        # off-center-hit spin (0 = pre-v6 rules)
    ships: list = field(default_factory=_spawn_ships)
    lasers: list = field(default_factory=list)
    time: float = 0.0                 # physics seconds elapsed
    events: list = field(default_factory=list)   # ('death', ship_idx, cause), ('hit', ...)

    def reset(self):
        self.ships = _spawn_ships()
        self.lasers = []
        self.time = 0.0
        self.events = []
        return self

    # actions: per ship (turn, thrust, fire); turn is a SIGNED RATE in rad per
    # physics second (clamped to MAX_TURN_RATE), thrust/fire in {0, 1}
    def step(self, actions=((0.0, 0, 0), (0.0, 0, 0))):
        self.events = []
        half_w, half_h = PLAY_W / 2.0, H / 2.0

        for i, ship in enumerate(self.ships):
            if not ship.alive:
                continue
            turn, thrust, fire = actions[i]
            # spin fights the commanded steering and decays like angular
            # damping - a wall scrape costs aim, not just points
            ship.heading += (max(-MAX_TURN_RATE, min(MAX_TURN_RATE, turn))
                             + ship.spin) * DT
            ship.spin *= math.exp(-SPIN_DAMP * DT)
            ship.thrusting = bool(thrust)

            ax, ay = gravity_accel(ship.x, ship.y)
            if thrust:
                nx, ny = ship.nose()
                ax += THRUST_ACCEL * nx
                ay += THRUST_ACCEL * ny
            # semi-implicit Euler: velocity first, then position
            ship.vx += ax * DT
            ship.vy += ay * DT
            ship.x += ship.vx * DT
            ship.y += ship.vy * DT

            # arena edge: reflect with restitution; the scrape imparts spin and
            # emits a 'wall' event (rebound play is penalized by the modern
            # rules - the game is about out-orbiting, not bank shots)
            if abs(ship.x) > half_w - ship.r:
                ship.x = math.copysign(half_w - ship.r, ship.x)
                ship.vx = -ship.vx * RESTITUTION
                ship.spin += -ship.vy * self.spin_kick
                self.events.append(('wall', i, round(abs(ship.vy), 1)))
            if abs(ship.y) > half_h - ship.r:
                ship.y = math.copysign(half_h - ship.r, ship.y)
                ship.vy = -ship.vy * RESTITUTION
                ship.spin += -ship.vx * self.spin_kick
                self.events.append(('wall', i, round(abs(ship.vx), 1)))

            # the hole: contact of the two collision circles = death
            if ship.radius() < HOLE_R + ship.r:
                ship.alive = False
                self.events.append(('death', i, 'blackhole'))
                continue

            ship.cooldown = max(0.0, ship.cooldown - DT)
            if fire and ship.cooldown == 0.0:
                nx, ny = ship.nose()
                self.lasers.append(Laser(
                    x=ship.x + nx * ship.laser_off, y=ship.y + ny * ship.laser_off,
                    vx=ship.vx + nx * LASER_SPEED, vy=ship.vy + ny * LASER_SPEED,
                    owner=i))
                ship.cooldown = FIRE_COOLDOWN

        # bullet-vs-bullet: both rounds spent (the in-game "tink")
        spent = set()
        for a in range(len(self.lasers)):
            for b in range(a + 1, len(self.lasers)):
                la, lb = self.lasers[a], self.lasers[b]
                if la.owner != lb.owner and a not in spent and b not in spent \
                   and math.hypot(la.x - lb.x, la.y - lb.y) < 2 * LASER_R:
                    spent.update((a, b))
        if spent:
            self.lasers = [lz for k, lz in enumerate(self.lasers) if k not in spent]

        # lasers fly (curving unless an era preset straightens them), expire, hit
        keep = []
        for lz in self.lasers:
            ax, ay = gravity_accel(lz.x, lz.y, self.gravity_on_lasers)
            lz.vx += ax * DT
            lz.vy += ay * DT
            lz.x += lz.vx * DT
            lz.y += lz.vy * DT
            lz.ttl -= DT
            if lz.ttl <= 0 or abs(lz.x) > half_w or abs(lz.y) > half_h \
               or math.hypot(lz.x, lz.y) < HOLE_R:
                continue
            hit = False
            for i, ship in enumerate(self.ships):
                if i == lz.owner or not ship.alive:
                    continue
                if math.hypot(ship.x - lz.x, ship.y - lz.y) < ship.r + LASER_R:
                    ship.damage += 1
                    ship.vx += lz.vx * HIT_KICK   # momentum transfer: a hit knocks
                    ship.vy += lz.vy * HIT_KICK
                    # off-center hits SPIN the victim (contact-point impulse):
                    # lever = how far off-axis the round struck
                    ox, oy = lz.x - ship.x, lz.y - ship.y
                    ln = math.hypot(ox, oy) * (math.hypot(lz.vx, lz.vy) or 1e-9)
                    ship.spin += self.hit_spin * (ox * lz.vy - oy * lz.vx) / (ln or 1e-9)
                    self.events.append(('hit', i, lz.owner))
                    if ship.damage >= MAX_DAMAGE:
                        ship.alive = False
                        self.events.append(('death', i, 'laser'))
                    hit = True
                    break
            if not hit:
                keep.append(lz)
        self.lasers = keep

        self.time += DT
        return self.events
