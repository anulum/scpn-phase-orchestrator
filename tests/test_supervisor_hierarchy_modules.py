# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor/hierarchy package wiring tests

"""Wiring contract for the ``supervisor.hierarchy`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol — and the private
helpers that existing callers import by name — as the *same object* defined in
its owning submodule, so the split preserves the original flat
``supervisor.hierarchy`` import surface with zero consumer churn.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.supervisor.hierarchy as hierarchy

PACKAGE = "scpn_phase_orchestrator.supervisor.hierarchy"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.supervisor.hierarchy.boundary",
    "scpn_phase_orchestrator.supervisor.hierarchy.plan",
    "scpn_phase_orchestrator.supervisor.hierarchy.sync",
    "scpn_phase_orchestrator.supervisor.hierarchy.consensus",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "ChildSupervisorSummary": "boundary",
    "HierarchicalOrchestrationPlan": "plan",
    "HierarchyConsensusRound": "consensus",
    "HierarchyConsensusState": "consensus",
    "HierarchyEscalation": "boundary",
    "HierarchySyncEnvelope": "boundary",
    "HierarchySyncLedger": "sync",
    "HierarchyTransportRuntime": "sync",
    "build_hierarchical_orchestration_plan": "plan",
    "build_hierarchy_sync_envelope": "sync",
    "ingest_hierarchy_sync_envelopes": "sync",
    "load_hierarchy_sync_envelope": "sync",
    "simulate_hierarchy_gossip_consensus": "consensus",
}

# Private helpers imported by name by existing callers; the split must keep them
# resolvable on the package namespace as their owning submodule's object.
PRIVATE_REEXPORT_MODULE = {
    "_child_escalations": "boundary",
    "_is_forbidden_hierarchy_key": "boundary",
    "_load_child_summary": "sync",
    "_load_mapping_record": "sync",
    "_metadata_to_audit_record": "boundary",
    "_normalise_metadata_value": "boundary",
    "_normalise_previous_sequences": "boundary",
    "_reject_raw_hierarchy_keys": "sync",
    "_reject_raw_instance_attributes": "boundary",
    "_reject_unknown_keys": "sync",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(hierarchy.__all__)
    assert len(hierarchy.__all__) == len(set(hierarchy.__all__))


@pytest.mark.parametrize(
    "symbol,module_name",
    sorted({**PUBLIC_SYMBOL_MODULE, **PRIVATE_REEXPORT_MODULE}.items()),
)
def test_symbol_reexport_is_owning_module_object(symbol: str, module_name: str) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(hierarchy, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values()) | set(PRIVATE_REEXPORT_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names
