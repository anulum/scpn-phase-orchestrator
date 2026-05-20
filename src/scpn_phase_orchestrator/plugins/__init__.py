# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin interface exports

"""Public plugin-interface facade for manifest and compatibility utilities.

The plugins package re-exports manifest dataclasses, capability records,
registry builders, marketplace catalog generation, discovery, and compatibility
reporting. Importing the package does not discover files, load native plugins,
or execute plugin code; validation and registry construction occur only through
explicit calls into ``plugins.registry``.
"""

from __future__ import annotations

from scpn_phase_orchestrator.plugins.registry import (
    ExecutedPluginCapability,
    LoadedPluginCapability,
    PluginCapability,
    PluginCompatibilityReport,
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequest,
    PluginExecutionRequestStorageManifest,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
    build_plugin_execution_request_storage_manifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
    compatibility_report,
    discover_plugin_manifests,
    execute_plugin_capability,
    execute_plugin_execution_request,
    load_plugin_capability,
    validate_plugin_execution_request,
    validate_plugin_execution_request_storage_manifest,
    validate_plugin_manifest,
)

__all__ = [
    "PluginCapability",
    "PluginCompatibilityReport",
    "PluginExecutionApproval",
    "PluginExecutionRequest",
    "PluginExecutionRequestStorageManifest",
    "PluginExecutionPlan",
    "PluginManifest",
    "PluginRuntimeExecutionPolicy",
    "PluginRuntimeLoadPolicy",
    "ExecutedPluginCapability",
    "build_plugin_execution_approval",
    "build_plugin_execution_request",
    "build_plugin_execution_request_storage_manifest",
    "build_plugin_execution_plan",
    "build_plugin_marketplace_catalog",
    "build_rust_plugin_runtime_handoff",
    "build_rust_plugin_registry",
    "compatibility_report",
    "discover_plugin_manifests",
    "LoadedPluginCapability",
    "execute_plugin_capability",
    "execute_plugin_execution_request",
    "load_plugin_capability",
    "validate_plugin_execution_request",
    "validate_plugin_execution_request_storage_manifest",
    "validate_plugin_manifest",
]
