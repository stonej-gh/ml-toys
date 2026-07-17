# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""The committed golden bundle must verify - run the bundle's own gate."""

import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[1] / "deploy"


@pytest.mark.skipif(not (DEPLOY / "models/spotter_dense.json").exists(),
                    reason="bundle not built")
def test_bundle_verifies():
    r = subprocess.run([sys.executable, str(DEPLOY / "verify.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERIFY: PASS" in r.stdout
