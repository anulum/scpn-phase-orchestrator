# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public API docstring regressions

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine


def test_source_public_python_surfaces_have_docstrings() -> None:
    missing: list[str] = []
    for path in sorted(Path("src/scpn_phase_orchestrator").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(f"{path}:1 module")
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                if not ast.get_docstring(node):
                    missing.append(f"{path}:{node.lineno} {node.name}")
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if not isinstance(
                            child, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ):
                            continue
                        if child.name.startswith("_"):
                            continue
                        if not ast.get_docstring(child):
                            missing.append(
                                f"{path}:{child.lineno} {node.name}.{child.name}"
                            )

    assert missing == []


@pytest.mark.parametrize(
    ("target", "sections"),
    [
        (UPDEEngine.step, ("Parameters", "Returns", "Raises")),
        (CouplingBuilder.build, ("Parameters", "Returns", "Notes")),
        (SwarmalatorEngine.step, ("Parameters", "Returns", "Notes")),
        (BoundaryObserver.observe, ("Parameters", "Returns", "Notes")),
    ],
)
def test_core_public_methods_use_numpy_style_docstrings(
    target: Callable[..., object], sections: tuple[str, ...]
) -> None:
    doc = inspect.getdoc(target)

    assert doc is not None
    for section in sections:
        assert section in doc
        assert f"{section}\n{'-' * len(section)}" in doc


def test_rust_upde_stepper_constructor_documents_public_contract() -> None:
    rust_source = Path("spo-kernel/crates/spo-ffi/src/lib.rs").read_text(
        encoding="utf-8"
    )
    stepper_start = rust_source.index('pyclass(name = "PyUPDEStepper")')
    constructor_start = rust_source.index("fn new(", stepper_start)
    constructor_doc = rust_source[stepper_start:constructor_start]

    assert "Create a Rust-backed UPDE stepper." in constructor_doc
    assert "Parameters" in constructor_doc
    assert "Raises" in constructor_doc
    assert "method" in constructor_doc
    assert "ValueError" in constructor_doc
