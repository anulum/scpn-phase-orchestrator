# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public API docstring regressions

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

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


# Public API families whose every public callable must carry the applicable
# NumPy-style contract sections, not merely a docstring. Extend this tuple one
# family at a time as M2 (public docstring quality enforcement) closes them.
SECTION_ENFORCED_MODULES = (
    "scpn_phase_orchestrator.api",
    "scpn_phase_orchestrator.binding.channel_algebra",
    "scpn_phase_orchestrator.binding.channel_runtime",
    "scpn_phase_orchestrator.binding.digital_twin",
    "scpn_phase_orchestrator.binding.loader",
    "scpn_phase_orchestrator.binding.resolved",
    "scpn_phase_orchestrator.binding.semantic",
    "scpn_phase_orchestrator.binding.topos_examples",
    "scpn_phase_orchestrator.binding.topos_semantic",
    "scpn_phase_orchestrator.binding.types",
    "scpn_phase_orchestrator.binding.validator",
    "scpn_phase_orchestrator.upde.adjoint",
    "scpn_phase_orchestrator.upde.basin_stability",
    "scpn_phase_orchestrator.upde.bifurcation",
    "scpn_phase_orchestrator.upde.delay",
    "scpn_phase_orchestrator.upde.engine",
    "scpn_phase_orchestrator.upde.envelope",
    "scpn_phase_orchestrator.upde.geometric",
    "scpn_phase_orchestrator.upde.hypergraph",
    "scpn_phase_orchestrator.upde.inertial",
    "scpn_phase_orchestrator.upde.jax_engine",
    "scpn_phase_orchestrator.upde.market",
    "scpn_phase_orchestrator.upde.numerics",
    "scpn_phase_orchestrator.upde.order_params",
    "scpn_phase_orchestrator.upde.pac",
    "scpn_phase_orchestrator.upde.reduction",
    "scpn_phase_orchestrator.upde.sheaf_engine",
    "scpn_phase_orchestrator.upde.simplicial",
    "scpn_phase_orchestrator.upde.sparse_engine",
    "scpn_phase_orchestrator.upde.splitting",
    "scpn_phase_orchestrator.upde.stochastic",
    "scpn_phase_orchestrator.upde.stuart_landau",
    "scpn_phase_orchestrator.upde.swarmalator",
    "scpn_phase_orchestrator.upde.bayesian",
    "scpn_phase_orchestrator.upde.doppler",
    "scpn_phase_orchestrator.upde.moving_frame",
    "scpn_phase_orchestrator.upde.pha_c_acceptance",
    "scpn_phase_orchestrator.upde.pha_c_formal_obligation",
    "scpn_phase_orchestrator.upde.pha_c_handoff",
    "scpn_phase_orchestrator.upde.pha_c_timeline",
    "scpn_phase_orchestrator.upde.prediction",
    "scpn_phase_orchestrator.coupling.attention_residuals",
    "scpn_phase_orchestrator.coupling.connectome",
    "scpn_phase_orchestrator.coupling.ei_balance",
    "scpn_phase_orchestrator.coupling.geometry_constraints",
    "scpn_phase_orchestrator.coupling.hodge",
    "scpn_phase_orchestrator.coupling.infer",
    "scpn_phase_orchestrator.coupling.knm",
    "scpn_phase_orchestrator.coupling.lags",
    "scpn_phase_orchestrator.coupling.plasticity",
    "scpn_phase_orchestrator.coupling.prior",
    "scpn_phase_orchestrator.coupling.spatial_modulator",
    "scpn_phase_orchestrator.coupling.spectral",
    "scpn_phase_orchestrator.coupling.te_adaptive",
    "scpn_phase_orchestrator.coupling.templates",
)


def _unwrap(obj: object) -> object:
    if isinstance(obj, (classmethod, staticmethod)):
        return obj.__func__
    return obj


def _public_callables(
    module: ModuleType,
) -> Iterator[tuple[str, Callable[..., object]]]:
    exported = getattr(
        module, "__all__", [n for n in vars(module) if not n.startswith("_")]
    )
    for name in exported:
        obj = getattr(module, name)
        if inspect.isfunction(obj) and obj.__module__ == module.__name__:
            yield f"{module.__name__}.{name}", obj
        elif inspect.isclass(obj) and obj.__module__ == module.__name__:
            for member_name, raw in vars(obj).items():
                if member_name.startswith("_"):
                    continue
                func = _unwrap(raw)
                if inspect.isfunction(func):
                    yield f"{module.__name__}.{name}.{member_name}", func


def _required_sections(func: Callable[..., object]) -> list[str]:
    sections: list[str] = []
    parameters = [
        param
        for param in inspect.signature(func).parameters.values()
        if param.name not in ("self", "cls")
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if parameters:
        sections.append("Parameters")
    return_annotation = func.__annotations__.get("return")
    if return_annotation not in (None, "None", type(None)):
        sections.append("Returns")
    try:
        source = inspect.getsource(func)
    except OSError:  # pragma: no cover - source always available in-tree
        source = ""
    if "raise " in source:
        sections.append("Raises")
    return sections


def test_section_enforced_families_document_numpy_contracts() -> None:
    problems: list[str] = []
    for module_name in SECTION_ENFORCED_MODULES:
        module = importlib.import_module(module_name)
        for qualified_name, func in _public_callables(module):
            doc = inspect.getdoc(func) or ""
            for section in _required_sections(func):
                header = f"{section}\n{'-' * len(section)}"
                if header not in doc:
                    problems.append(f"{qualified_name}: missing '{section}' section")

    assert problems == []


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
