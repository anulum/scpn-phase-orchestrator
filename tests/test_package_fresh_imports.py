# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fresh-interpreter import tests

"""Prove every package imports first in a fresh interpreter.

A test process imports many modules before any test runs, so an import cycle
that only breaks when one particular module is imported first stays hidden
from the in-process suite. Each import here runs in its own isolated
interpreter, as a user's first ``import`` would.
"""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import scpn_phase_orchestrator as spo

_PACKAGE_ROOT = Path(spo.__file__).resolve().parent

# Modules through which the binding/supervisor/monitor import cycle was entered
# before the topos examples became a lazy binding export.
_CYCLE_ENTRY_MODULES = (
    "scpn_phase_orchestrator.monitor.boundaries",
    "scpn_phase_orchestrator.monitor.stl.monitor",
    "scpn_phase_orchestrator.supervisor.policy_rules",
)


def _package_names() -> list[str]:
    """Return every importable package of the distribution, root first."""
    names = []
    for init in sorted(_PACKAGE_ROOT.rglob("__init__.py")):
        relative = init.parent.relative_to(_PACKAGE_ROOT.parent)
        names.append(".".join(relative.parts))
    return names


def _first_import_failure(module: str) -> str | None:
    """Import ``module`` first in a fresh isolated interpreter.

    Returns the last stderr line when the import fails, else ``None``.
    """
    completed = subprocess.run(
        [sys.executable, "-I", "-W", "ignore", "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=180.0,
        check=False,
    )
    if completed.returncode == 0:
        return None
    lines = completed.stderr.strip().splitlines()
    return lines[-1] if lines else f"exit status {completed.returncode}"


def test_package_discovery_covers_the_known_subpackages() -> None:
    """The package list is derived from the tree, not typed in by hand."""
    names = _package_names()
    assert names[0] == "scpn_phase_orchestrator"
    for expected in (
        "scpn_phase_orchestrator.binding",
        "scpn_phase_orchestrator.monitor",
        "scpn_phase_orchestrator.monitor.stl",
        "scpn_phase_orchestrator.supervisor",
    ):
        assert expected in names
    assert len(names) == len(set(names))


def test_packages_and_cycle_entries_import_first_in_fresh_interpreters() -> None:
    """No package, and no former cycle entry module, fails as the first import."""
    modules = [*_package_names(), *_CYCLE_ENTRY_MODULES]
    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = dict(
            zip(modules, executor.map(_first_import_failure, modules), strict=True)
        )
    failures = {module: error for module, error in outcomes.items() if error}
    assert failures == {}


def test_lazy_binding_exports_resolve_to_the_topos_module() -> None:
    """The lazily exported topos names are the objects the module defines."""
    import scpn_phase_orchestrator.binding as binding
    from scpn_phase_orchestrator.binding import topos_examples

    for name in (
        "ToposDomainObligation",
        "ToposProofObligation",
        "build_topos_domain_obligation_examples",
    ):
        assert name in binding.__all__
        assert name in dir(binding)
        assert getattr(binding, name) is getattr(topos_examples, name)


def test_binding_rejects_unknown_attributes() -> None:
    """The lazy loader does not turn a typo into a silent import."""
    import scpn_phase_orchestrator.binding as binding

    with pytest.raises(AttributeError, match="ToposDomainObligations"):
        binding.ToposDomainObligations  # noqa: B018
