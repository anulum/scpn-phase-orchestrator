# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — monitor/stl package wiring tests

"""Wiring contract for the ``monitor.stl`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol as the *same object*
defined in its owning submodule, so the split preserves the original flat
``monitor.stl`` import surface with zero consumer churn. The British and American
synthesis spellings stay the same object, and ``HAS_RTAMT`` reports availability.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.monitor.stl as stl

PACKAGE = "scpn_phase_orchestrator.monitor.stl"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.monitor.stl.monitor",
    "scpn_phase_orchestrator.monitor.stl.automaton",
    "scpn_phase_orchestrator.monitor.stl.controller",
    "scpn_phase_orchestrator.monitor.stl.projection",
    "scpn_phase_orchestrator.monitor.stl.actuation_gate",
    "scpn_phase_orchestrator.monitor.stl.closed_loop",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "HAS_RTAMT": "monitor",
    "STLActionProjectionTemplate": "projection",
    "STLAutomatonState": "automaton",
    "STLAutomatonTransition": "automaton",
    "STLClosedLoopSynthesisPlan": "closed_loop",
    "STLControllerCandidate": "controller",
    "STLControllerSynthesis": "controller",
    "STLMonitor": "monitor",
    "STLMonitoringAutomaton": "automaton",
    "STLProjectedActionPlan": "projection",
    "STLRuntimeActuationGate": "actuation_gate",
    "STLTraceResult": "monitor",
    "project_stl_controller_candidates": "projection",
    "synthesise_stl_closed_loop_plan": "closed_loop",
    "synthesise_stl_controller_candidates": "controller",
    "synthesise_stl_monitoring_automaton": "automaton",
    "synthesize_stl_closed_loop_plan": "closed_loop",
    "synthesize_stl_controller_candidates": "controller",
    "synthesize_stl_monitoring_automaton": "automaton",
    "validate_stl_runtime_actuation_gate": "actuation_gate",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(stl.__all__)
    assert len(stl.__all__) == len(set(stl.__all__))


@pytest.mark.parametrize("symbol,module_name", sorted(PUBLIC_SYMBOL_MODULE.items()))
def test_public_symbol_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(stl, symbol) is getattr(owner, symbol)


def test_synthesis_spelling_aliases_are_identical() -> None:
    assert (
        stl.synthesize_stl_monitoring_automaton
        is stl.synthesise_stl_monitoring_automaton
    )
    assert (
        stl.synthesize_stl_controller_candidates
        is stl.synthesise_stl_controller_candidates
    )
    assert stl.synthesize_stl_closed_loop_plan is stl.synthesise_stl_closed_loop_plan


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names
