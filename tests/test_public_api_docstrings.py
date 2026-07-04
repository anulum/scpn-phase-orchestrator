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
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.swarmalator import SwarmalatorEngine

SOURCE_ROOT = Path("src/scpn_phase_orchestrator")


def _tracked_python_sources() -> tuple[Path, ...]:
    """Return Python sources from the committed public source inventory."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git executable is required for docstring inventory")
    result = subprocess.run(
        [git, "ls-files", "--", SOURCE_ROOT.as_posix()],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return tuple(
        Path(path) for path in result.stdout.splitlines() if path.endswith(".py")
    )


def test_source_public_python_surfaces_have_docstrings() -> None:
    missing: list[str] = []
    for path in _tracked_python_sources():
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
def _discover_enforced_modules() -> tuple[str, ...]:
    """Return every importable public module under ``src/`` for section checks.

    Discovery keeps the gate from drifting: a new module is covered the moment
    it lands, with no allow-list to maintain. The only exclusions are
    underscore-prefixed private modules/packages, the generated ``grpc_gen``
    gRPC stubs, and the ``sys.modules`` compatibility aliases (which re-point at
    an already-enforced module).
    """
    modules: list[str] = []
    for path in _tracked_python_sources():
        parts = path.relative_to(SOURCE_ROOT).with_suffix("").parts
        if any(part.startswith("_") for part in parts):
            continue
        if "grpc_gen" in parts:
            continue
        if "sys.modules[__name__]" in path.read_text(encoding="utf-8"):
            continue
        modules.append("scpn_phase_orchestrator." + ".".join(parts))
    return tuple(modules)


SECTION_ENFORCED_MODULES = _discover_enforced_modules()


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
        if name.startswith("_"):
            # Underscore-prefixed names are private even when re-exported in
            # ``__all__`` (e.g. test helpers); the presence gate skips them too.
            continue
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
