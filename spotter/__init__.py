# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""spotter: a tiny CNN that sees the orbital duel from raw pixels.

The deployable artifact lives in deploy/ (numpy-only, self-contained); this
package is the training side: the seeded scene renderer (which doubles as the
labeler - frames and masks come from the same draw calls), scene sampling,
training, export, quantization.
"""

__version__ = "0.2.0"

# Class map is fixed for v1 - everything downstream (masks, losses, goldens,
# demo palette) indexes by these ids. 255 marks pixels excluded from the loss
# (thrust flames: transient exhaust, deliberately unlabeled).
CLASSES = ["background", "interceptor", "fighter", "laser", "hole"]
IGNORE = 255

# Overlay palette (RGB) - shared by review renders and the demo overlay so the
# viewer's eye calibrates once. Neon variants of each entity's game color;
# hole gets magenta (its in-game glow is cyan, so the overlay must differ).
PALETTE = {
    0: (0, 0, 0),          # background - rendered transparent in overlays
    1: (60, 170, 255),     # interceptor - neon blue
    2: (255, 130, 40),     # fighter - neon orange
    3: (255, 255, 60),     # laser - neon yellow
    4: (255, 60, 220),     # hole - neon magenta
    IGNORE: (0, 0, 0),     # ignore (flames) - transparent in overlays
}
