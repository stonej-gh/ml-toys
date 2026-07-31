# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""T0 curriculum task: SURVIVE. One ship, no opponent, no lasers.

The ship spawns on a DOOMED orbit: radius anywhere in the flyable band but
tangential speed only 0.30-0.75x local circular (plus a radial kick and a
random heading), so the free ellipse's periapsis is almost always inside the
hole's capture radius. A policy that does nothing dies in seconds; the agent must
learn to point prograde and burn: the classic "settle into orbit"
exercise, learned from scratch.

reset()/step() follow the Gymnasium contract but the class works with stdlib
only. Actions are Discrete(6): turn {-1,0,+1} x thrust {0,1}. Observation is a
compact 8-float survival state (all roughly unit-scale):

    0  r / h                         4  cos(heading - CCW tangent)
    1  radial velocity / vc          5  hole clearance (r - capture radius) / h
    2  tangential velocity / vc      6  edge clearance / h
    3  sin(heading - CCW tangent)    7  speed / local circular speed

Reward: +1 per WALL second alive (paid per step), -`wall_penalty` (default
8.0) per wall strike, -10 on death. Episodes truncate (success) at
`max_wall_seconds`.
"""

import math
import random

from . import physics as P

ACTIONS = [(t, th) for t in (-1, 0, 1) for th in (0, 1)]
OBS_DIM = 8
KILL_R = P.HOLE_R + P.SHIP_R


class SurviveEnv:
    def __init__(self, action_repeat=4, max_wall_seconds=60.0, seed=None,
                 wall_penalty=8.0):
        self.action_repeat = action_repeat
        self.max_frames = int(max_wall_seconds * P.FPS)
        self.rng = random.Random(seed)
        self.wall_penalty = wall_penalty  # ~0.8x the death penalty per strike
        self.arena = P.Arena()
        self._frames = 0

    def _obs(self):
        s = self.arena.ships[0]
        r = s.radius() or 1e-9
        vc_local = math.sqrt(P.GM / r)
        ux, uy = s.x / r, s.y / r
        tx, ty = -uy, ux
        vr = s.vx * ux + s.vy * uy
        vt = s.vx * tx + s.vy * ty
        nx, ny = s.nose()
        h_rel = math.atan2(tx * ny - ty * nx, tx * nx + ty * ny)
        edge = min(P.PLAY_W / 2 - abs(s.x), P.H / 2 - abs(s.y)) / P.H
        return [r / P.H, vr / P.SPAWN_V, vt / P.SPAWN_V,
                math.sin(h_rel), math.cos(h_rel),
                (r - KILL_R) / P.H, edge, s.speed() / vc_local]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.arena.reset()
        s = self.arena.ships[0]
        self.arena.ships[1].alive = False  # no opponent in T0
        rng = self.rng
        r0 = rng.uniform(KILL_R + 40.0, P.H / 2 - P.SHIP_R - 20.0)
        ang = rng.uniform(0, 2 * math.pi)
        vc = math.sqrt(P.GM / r0)
        vt = vc * rng.uniform(0.30, 0.75) * (1 if rng.random() < 0.5 else -1)
        vr = vc * rng.uniform(-0.30, 0.30)
        ux, uy = math.cos(ang), math.sin(ang)
        s.x, s.y = r0 * ux, r0 * uy
        s.vx = vr * ux - vt * uy  # radial + CCW-tangential parts
        s.vy = vr * uy + vt * ux
        s.heading = rng.uniform(0, 2 * math.pi)
        self._frames = 0
        return self._obs(), {}

    def step(self, action):
        turn, thrust = ACTIONS[int(action)]
        reward, terminated = 0.0, False
        for _ in range(self.action_repeat):
            events = self.arena.step(((turn * P.TURN_RATE, thrust, 0), (0, 0, 0)))
            self._frames += 1
            reward += 1.0 / P.FPS  # +1 per wall second alive
            reward -= self.wall_penalty * sum(
                1 for ev in events if ev[0] == 'wall' and ev[1] == 0)
            if any(ev[0] == 'death' and ev[1] == 0 for ev in events):
                reward -= 10.0
                terminated = True
                break
        truncated = self._frames >= self.max_frames
        return self._obs(), reward, terminated, truncated, {}

    def wall_seconds(self):
        return self._frames / P.FPS
