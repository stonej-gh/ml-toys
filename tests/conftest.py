# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
import sys
from pathlib import Path

# Tests import orbitduel (installable) and agents/ (repo-local scripts); make
# both importable when running pytest from a bare checkout with no install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
