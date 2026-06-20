# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — plugins/registry package wiring tests

"""Wiring contract for the ``plugins.registry`` package.

Asserts that every responsibility submodule imports under its dotted path and
that the package ``__init__`` re-exports each public symbol as the *same object*
defined in its owning submodule, so the split preserves the original flat
``plugins.registry`` import surface with zero consumer churn. ``importlib``,
``metadata``, and ``__version__`` stay resolvable on the package namespace so the
existing monkeypatch contracts keep working.
"""

from __future__ import annotations

import importlib

import pytest

import scpn_phase_orchestrator.plugins.registry as registry

PACKAGE = "scpn_phase_orchestrator.plugins.registry"

# Full dotted paths spelled out so the module-linkage guard
# (tools/check_test_module_linkage.py) finds each submodule's import path.
SUBMODULES = (
    "scpn_phase_orchestrator.plugins.registry._shared",
    "scpn_phase_orchestrator.plugins.registry.policy",
    "scpn_phase_orchestrator.plugins.registry.manifest",
    "scpn_phase_orchestrator.plugins.registry.request",
    "scpn_phase_orchestrator.plugins.registry.storage",
    "scpn_phase_orchestrator.plugins.registry.revocation",
    "scpn_phase_orchestrator.plugins.registry.lifecycle",
    "scpn_phase_orchestrator.plugins.registry.runtime",
    "scpn_phase_orchestrator.plugins.registry.rust_handoff",
)

# Owning submodule for each public symbol re-exported by the package __init__.
PUBLIC_SYMBOL_MODULE = {
    "ExecutedPluginCapability": "runtime",
    "LoadedPluginCapability": "runtime",
    "PluginCapability": "manifest",
    "PluginCompatibilityReport": "manifest",
    "PluginExecutionApproval": "request",
    "PluginExecutionPlan": "runtime",
    "PluginExecutionRequest": "request",
    "PluginExecutionRequestLifecyclePolicyReport": "lifecycle",
    "PluginExecutionRequestLifecycleRecord": "lifecycle",
    "PluginExecutionRequestLifecycleSummary": "lifecycle",
    "PluginExecutionRequestRevocation": "revocation",
    "PluginExecutionRequestRevocationList": "revocation",
    "PluginExecutionRequestStorageAdapterManifest": "storage",
    "PluginExecutionRequestStorageManifest": "storage",
    "PluginManifest": "manifest",
    "PluginRuntimeExecutionPolicy": "policy",
    "PluginRuntimeLoadPolicy": "policy",
    "build_plugin_execution_approval": "request",
    "build_plugin_execution_plan": "runtime",
    "build_plugin_execution_request": "request",
    "build_plugin_execution_request_lifecycle_policy_report": "lifecycle",
    "build_plugin_execution_request_lifecycle_record": "lifecycle",
    "build_plugin_execution_request_lifecycle_summary": "lifecycle",
    "build_plugin_execution_request_revocation": "revocation",
    "build_plugin_execution_request_revocation_list": "revocation",
    "build_plugin_execution_request_storage_adapter_manifest": "storage",
    "build_plugin_execution_request_storage_bundle": "storage",
    "build_plugin_execution_request_storage_manifest": "storage",
    "build_plugin_marketplace_catalog": "manifest",
    "build_rust_plugin_registry": "rust_handoff",
    "build_rust_plugin_runtime_handoff": "rust_handoff",
    "compatibility_report": "manifest",
    "discover_plugin_manifests": "manifest",
    "execute_plugin_capability": "runtime",
    "execute_plugin_execution_request": "runtime",
    "load_plugin_capability": "runtime",
    "validate_plugin_execution_request": "request",
    "validate_plugin_execution_request_lifecycle_record": "lifecycle",
    "validate_plugin_execution_request_lifecycle_summary": "lifecycle",
    "validate_plugin_execution_request_revocation_list": "revocation",
    "validate_plugin_execution_request_storage_adapter_manifest": "storage",
    "validate_plugin_execution_request_storage_bundle": "storage",
    "validate_plugin_execution_request_storage_manifest": "storage",
    "validate_plugin_manifest": "manifest",
    "write_plugin_execution_request_storage_bundle": "storage",
}


@pytest.mark.parametrize("dotted_path", SUBMODULES)
def test_submodule_imports_under_dotted_path(dotted_path: str) -> None:
    assert dotted_path.startswith(f"{PACKAGE}.")
    module = importlib.import_module(dotted_path)
    assert module.__name__ == dotted_path
    assert (module.__doc__ or "").strip(), f"{dotted_path} has no module docstring"


def test_public_symbol_module_table_covers_dunder_all() -> None:
    assert set(PUBLIC_SYMBOL_MODULE) == set(registry.__all__)
    assert len(registry.__all__) == len(set(registry.__all__))


@pytest.mark.parametrize("symbol,module_name", sorted(PUBLIC_SYMBOL_MODULE.items()))
def test_public_symbol_reexport_is_owning_module_object(
    symbol: str, module_name: str
) -> None:
    owner = importlib.import_module(f"{PACKAGE}.{module_name}")
    assert getattr(registry, symbol) is getattr(owner, symbol)


def test_every_reexport_owner_is_a_real_submodule() -> None:
    owners = set(PUBLIC_SYMBOL_MODULE.values())
    submodule_names = {path.rsplit(".", 1)[1] for path in SUBMODULES}
    assert owners <= submodule_names


def test_monkeypatch_attributes_resolve_on_package() -> None:
    # The pre-split tests patch ``registry.importlib``, ``registry.metadata``,
    # and ``registry.__version__``; these must stay resolvable on the package.
    assert registry.importlib.import_module is importlib.import_module
    assert hasattr(registry.metadata, "entry_points")
    assert isinstance(registry.__version__, str)
