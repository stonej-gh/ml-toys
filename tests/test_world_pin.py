# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Measured tables cite the world they were measured under; this test keeps
those citations honest. tools/world_pin.py digests the arena (physics, rules,
the scripted ladder's behavior, the survive task) from seeded checkpoint-free
probes, and every "measured" table in the experiment READMEs and exercise
walkthroughs carries a `world pin` citation. When the world changes, this
test fails and NAMES every document whose numbers are now unverified, which
is the alarm that was missing when the 2026-07-31 opponent port silently
invalidated three experiments' tables (found 2026-08-01, a day late).

Re-pinning is deliberately manual: re-measure the named tables, update their
numbers and citations, and only then quote the new pin. A re-pin commit that
does not touch the cited documents is the exact mistake this test exists to
block."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CITE_RE = re.compile(r"world pin\s+`([0-9a-f]{12})`")  # \s spans line wraps
MIN_CITATIONS = 12  # 16 tables cite the pin today; slack for future edits


def collect_citations():
    cites = []
    for pattern in ("experiments/*/README.md", "exercises/*/README.md"):
        for path in sorted(REPO.glob(pattern)):
            for pin in CITE_RE.findall(path.read_text(encoding="utf-8")):
                cites.append((path.relative_to(REPO), pin))
    return cites


def test_measured_tables_cite_the_current_world():
    from tools.world_pin import current_pin
    pin, rows = current_pin()
    cites = collect_citations()
    assert len(cites) >= MIN_CITATIONS, \
        f"only {len(cites)} world-pin citations found; the mechanism is rotting"
    stale = sorted({str(p) for p, cited in cites if cited != pin})
    if stale:
        table = "\n".join(f"  {r}" for r in rows)
        raise AssertionError(
            f"the world moved: current pin {pin}, but these documents cite "
            f"an older one and their measured tables are now unverified:\n  "
            + "\n  ".join(stale)
            + "\n\nre-measure each table, update its numbers AND its world-pin "
            "citation, then rerun. Current probe results:\n" + table)
