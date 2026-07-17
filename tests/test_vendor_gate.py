# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The vendor-neutrality gate must pass on the tracked tree, and must actually
detect a planted token (guards against the gate rotting into a no-op)."""

import subprocess
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent
GATE = LAB / "tools" / "check_vendor_neutral.sh"


def _tracked_here():
    r = subprocess.run(["git", "ls-files"], cwd=LAB, capture_output=True,
                       text=True)
    return r.returncode == 0 and r.stdout.strip()


def test_tree_is_vendor_neutral():
    if not _tracked_here():
        import pytest
        pytest.skip("not a git checkout")
    r = subprocess.run(["sh", str(GATE)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
