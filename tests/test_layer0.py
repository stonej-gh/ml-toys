# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Jeffrey Stone
#
# Released under the MIT License. Free to use, modify, and share.
# Attribution appreciated; see LICENSE for details.
"""Layering invariant: Layer 0 (orbitduel/) must not import Layer 1 (viz/),
the agents, or any future llm/ hooks - and must train with those deleted.
Checked structurally here; CI also runs the suite with viz/ removed."""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "orbitduel"
ALLOWED_THIRD_PARTY = {"gymnasium", "numpy"}
FORBIDDEN = {"viz", "agents", "llm", "torch", "PIL"}


def imports_of(path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_layer0_import_graph():
    for py in PKG.glob("*.py"):
        bad = imports_of(py) & FORBIDDEN
        assert not bad, f"{py.name} imports {bad}"


def test_layer0_third_party_is_minimal():
    import sys
    stdlib = set(sys.stdlib_module_names)
    for py in PKG.glob("*.py"):
        third = {m for m in imports_of(py)
                 if m not in stdlib and m != "orbitduel"}
        assert third <= ALLOWED_THIRD_PARTY, f"{py.name}: {third}"
