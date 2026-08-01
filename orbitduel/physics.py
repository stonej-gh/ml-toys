# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Headless 2D orbital-duel physics: two ships, one gravity well, lasers.

Stdlib only and deterministic, so the golden evals reproduce bit for bit. A
rectangular arena (768 pt tall) surrounds an inverse-square well whose event
horizon captures on contact. Thrust accelerates along the nose; lasers inherit
the firer's velocity plus a muzzle kick and curve in the field. Integration is
semi-implicit (symplectic) Euler, so orbits wobble ~1% but never drift.

Two fields: phone (default) and tablet (set ORBITDUEL_FIELD=tablet before
import). Two clocks: physics time and wall time, wall = physics / GAME_SPEED.
Units are points and physics seconds unless noted. The constants are
measured values, not tuned ones; their sourcing and the deliberately
unmodeled effects are in PROVENANCE.md.
"""

from dataclasses import dataclass, field
import math
import os

# field profile
FIELD_SCALE = 1.0                                    # 1.0 phone / 1.67 every tablet
_ASPECT = 19.5 / 9.0                                 # tall phone; the 16:9 cap engages
if os.environ.get("ORBITDUEL_FIELD") == "tablet":
    FIELD_SCALE = 1.67
    _ASPECT = float(os.environ.get("ORBITDUEL_TABLET_ASPECT", 1180.0 / 820.0))

# clocks and arena
FPS = 60.0                                           # frames per wall second

# the tablet runs its clock in proportion to the field, 0.5 x 1.67, and the
# phone falls out of the same rule. Read off tablet hardware as 0.835000, and
# every wall-clocked constant below (DT, the turn ceiling, laser TTL, fire
# cooldown) is derived from this one number.
GAME_SPEED = 0.5 * FIELD_SCALE                       # 0.5 phone / 0.835 tablet
DT = GAME_SPEED / FPS                                # physics seconds per frame
H = 768.0                                            # reference field height (pt)
MAX_PLAY_ASPECT = 16.0 / 9.0                         # arena width cap (phones are wider)
PLAY_W = min(_ASPECT, MAX_PLAY_ASPECT) * H           # 1365.33 phone / 1105.17 tablet

# gravity well
GM = 298.0 * 298.0 * 226.0 / FIELD_SCALE ** 3        # pt^3/s^2; GM /= FIELD_SCALE^3
HOLE_R = 68.0 / math.sqrt(FIELD_SCALE)               # event horizon, tapered as sqrt: 68.0 / 52.62
SPAWN_R = (HOLE_R + H / 2.0) / 2.0                   # spawn-orbit radius: 226.0 / 218.31
SPAWN_V = math.sqrt(GM / SPAWN_R)                    # circular speed at spawn: 298.0 / 140.49

# ships
SHIP_MASS = 0.03
SHIP_R_AGENT = 37.52 / FIELD_SCALE                   # fighter hull (seat 0), wider circle
SHIP_R_OPP = 27.16 / FIELD_SCALE                     # interceptor hull (seat 1), slimmer
SHIP_R = SHIP_R_AGENT                                # fighter alias; survive.py's lone ship is seat 0
THRUST_ACCEL = (20.0 / SHIP_MASS) / FIELD_SCALE      # 666.67 pt/s^2 along the nose
TURN_RATE = 4.0                                      # agent's discrete turn rate (rad/phys s)
MAX_TURN_RATE = 4.0 * math.pi / GAME_SPEED           # scripted slew cap: 2 rev/wall s

# collisions and spin
HIT_KICK = 0.003 * 5.0 / SHIP_MASS                   # laser-hit dv = 0.5 x laser velocity
RESTITUTION = 0.2                                    # wall-bounce velocity retention
SPIN_DAMP = 3.0                                      # uncommanded-spin decay rate (/s)
SPIN_KICK = 0.02                                     # wall scrape: spin += -v_along_wall x this
HIT_SPIN = 14.0                                      # off-center hit spin (fitted stand-in)

# lasers
LASER_SPEED = 2.1 * SPAWN_V                          # muzzle speed = 2.1 x circular speed
LASER_TTL = 1.4 * GAME_SPEED                         # 1.4 wall s of flight
LASER_R = 4.0                                        # collision circle for a slim round
LASER_OFF_AGENT = (22.0 - 5.0) * 0.84 / FIELD_SCALE  # muzzle offset: fighter 14.28
LASER_OFF_OPP = (26.0 - 5.0) * 0.84 / FIELD_SCALE    # muzzle offset: interceptor 17.64
FIRE_COOLDOWN = 0.3 * GAME_SPEED                     # base fire cadence (0.3 wall s)
MAX_DAMAGE = 5                                       # hits per life


def gravity_accel(x, y, gravity_on=True):
    """Inward inverse-square acceleration of the central field at (x, y)."""
    if not gravity_on:
        return 0.0, 0.0
    r2 = x * x + y * y
    r = math.sqrt(r2) or 1e-9
    a = -GM / (r2 * r)  # -GM/r^2 * (unit radial)
    return a * x, a * y


@dataclass
class Ship:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0                # radians; 0 = nose along +y
    spin: float = 0.0                   # uncommanded angular rate (wall scrape), decays
    damage: int = 0
    alive: bool = True
    cooldown: float = 0.0
    thrusting: bool = False             # set by the last step's action (for replays)
    r: float = SHIP_R_AGENT             # collision-circle radius (per hull; see reset)
    laser_off: float = LASER_OFF_AGENT  # muzzle body offset ahead of center

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
    the interceptor hull (the arena's fixed seat assignment)."""
    return [Ship(x=SPAWN_R, y=0.0, vy=SPAWN_V, heading=0.0,
                 r=SHIP_R_AGENT, laser_off=LASER_OFF_AGENT),
            Ship(x=-SPAWN_R, y=0.0, vy=-SPAWN_V, heading=math.pi,
                 r=SHIP_R_OPP, laser_off=LASER_OFF_OPP)]


@dataclass
class Arena:
    """The two-ship duel field. step() advances one 60-fps frame (DT physics s).

    gravity_on_lasers / spin_kick / hit_spin default to the full modern rules;
    the era presets in env.RULE_PRESETS turn them off to reproduce earlier
    phases of the experiment (straight lasers, free walls, no hit-spin) for the
    reward-hacking case study.
    """
    gravity_on_lasers: bool = True              # the real field pulls on lasers too
    spin_kick: float = SPIN_KICK                # wall-scrape spin transfer (0 = free walls)
    hit_spin: float = HIT_SPIN                  # off-center-hit spin (0 = pre-v6 rules)
    ships: list = field(default_factory=_spawn_ships)
    lasers: list = field(default_factory=list)
    time: float = 0.0                           # physics seconds elapsed
    events: list = field(default_factory=list)  # ('death', ship_idx, cause), ('hit', ...)

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
            # emits a 'wall' event (rebound play is priced by the modern rules)
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

        # bullet-vs-bullet: both rounds spent (the "tink")
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
                    ship.vx += lz.vx * HIT_KICK  # momentum transfer from the round
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
