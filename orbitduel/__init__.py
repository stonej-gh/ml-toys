# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""orbitduel: a tiny orbital-duel reinforcement-learning environment.

Layer 0 of the lab: physics core (stdlib-pure), Gymnasium duel env with era
rule presets, the scripted opponent ladder, the survive on-ramp task, and a
dependency-free reference forward pass for exported policies. No rendering,
no training frameworks, no network access; see viz/ (Layer 1) and agents/.
"""

__version__ = "0.1.0"

from . import physics
from .physics import Arena, Ship, Laser

__all__ = ["physics", "Arena", "Ship", "Laser", "__version__"]

try:
    import gymnasium as _gym
    _gym.register(id="OrbitDuel-v0", entry_point="orbitduel.env:OrbitDuelEnv")
except ImportError:  # physics core stays usable without gym
    pass
