# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — runtime/cli/_payloads package wiring tests

"""Wiring contract for the ``runtime.cli._payloads`` package.

Asserts that every per-domain loader submodule imports under its dotted path and
that the package ``__init__`` re-exports each loader as the *same object* defined
in its owning submodule, so the split preserves the original flat ``_payloads``
import surface that the ``spo plugins`` command modules and their tests rely on.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.runtime.cli._payloads as payloads

PACKAGE = "scpn_phase_orchestrator.runtime.cli._payloads"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.runtime.cli._payloads._shared",
    "scpn_phase_orchestrator.runtime.cli._payloads.plan",
    "scpn_phase_orchestrator.runtime.cli._payloads.revocation",
    "scpn_phase_orchestrator.runtime.cli._payloads.storage",
    "scpn_phase_orchestrator.runtime.cli._payloads.lifecycle",
    "scpn_phase_orchestrator.runtime.cli._payloads.remediation",
    "scpn_phase_orchestrator.runtime.cli._payloads.scheduler",
)

# Owning submodule for each loader re-exported by the package __init__.
SYMBOL_MODULE = {
    "_PLUGIN_KIND_OPTIONS": "_shared",
    "_build_plan_payload_for_hash": "_shared",
    "_find_capability": "_shared",
    "_find_discovered_plugin": "_shared",
    "_load_approval_from_payload": "plan",
    "_load_json_file": "_shared",
    "_load_lifecycle_from_payload": "lifecycle",
    "_load_lifecycle_multistore_drilldown_payload": "lifecycle",
    "_load_lifecycle_policy_report_payload": "lifecycle",
    "_load_lifecycle_remediation_action_status_payload": "remediation",
    "_load_lifecycle_remediation_deployment_handoff_payload": "remediation",
    "_load_lifecycle_remediation_execution_dashboard_payload": "remediation",
    "_load_lifecycle_remediation_plan_payload": "remediation",
    "_load_lifecycle_remediation_scheduler_acknowledgement_payload": "scheduler",
    "_load_lifecycle_remediation_scheduler_adapter_handoff_payload": "scheduler",
    "_load_lifecycle_remediation_scheduler_queue_payload": "scheduler",
    "_load_lifecycle_remediation_scheduler_telemetry_payload": "scheduler",
    "_load_lifecycle_summary_from_payload": "lifecycle",
    "_load_plan_from_payload": "plan",
    "_load_request_from_payload": "plan",
    "_load_revocation_from_payload": "revocation",
    "_load_revocation_list_from_payload": "revocation",
    "_load_storage_adapter_from_payload": "storage",
    "_load_storage_manifest_from_payload": "storage",
    "_normalize_approved_target_hashes": "_shared",
    "_record_hash": "_shared",
    "_require_sha256": "_shared",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_symbol_module_table_covers_dunder_all() -> None:
    assert set(SYMBOL_MODULE) == set(payloads.__all__)
    assert len(payloads.__all__) == len(set(payloads.__all__))


@pytest.mark.parametrize("symbol,module_name", sorted(SYMBOL_MODULE.items()))
def test_symbol_reexport_is_owning_module_object(symbol: str, module_name: str) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(payloads, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(SYMBOL_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names
