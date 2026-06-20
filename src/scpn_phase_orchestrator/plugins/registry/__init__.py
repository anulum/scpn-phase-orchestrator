# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin manifest registry

"""Plugin manifest registry: validation, execution governance, and runtime loading.

This package splits the plugin registry into responsibility modules (manifest,
policy, request, storage, revocation, lifecycle, runtime, rust hand-off) while
preserving the original public import surface. ``importlib``, ``metadata``, and
``__version__`` are re-exported so tests and callers can resolve them on this
package namespace; runtime loading remains Python-owned and disabled by default.
"""

from __future__ import annotations

import importlib as importlib
from importlib import metadata as metadata

from scpn_phase_orchestrator import __version__ as __version__

from ._shared import (
    PluginKind as PluginKind,
)
from ._shared import (
    _record_hash as _record_hash,
)
from .lifecycle import (
    PluginExecutionRequestLifecyclePolicyReport,
    PluginExecutionRequestLifecycleRecord,
    PluginExecutionRequestLifecycleSummary,
    build_plugin_execution_request_lifecycle_policy_report,
    build_plugin_execution_request_lifecycle_record,
    build_plugin_execution_request_lifecycle_summary,
    validate_plugin_execution_request_lifecycle_record,
    validate_plugin_execution_request_lifecycle_summary,
)
from .manifest import (
    PluginCapability,
    PluginCompatibilityReport,
    PluginManifest,
    build_plugin_marketplace_catalog,
    compatibility_report,
    discover_plugin_manifests,
    validate_plugin_manifest,
)
from .policy import (
    PluginRuntimeExecutionPolicy,
    PluginRuntimeLoadPolicy,
)
from .request import (
    PluginExecutionApproval,
    PluginExecutionRequest,
    build_plugin_execution_approval,
    build_plugin_execution_request,
    validate_plugin_execution_request,
)
from .revocation import (
    PluginExecutionRequestRevocation,
    PluginExecutionRequestRevocationList,
    build_plugin_execution_request_revocation,
    build_plugin_execution_request_revocation_list,
    validate_plugin_execution_request_revocation_list,
)
from .runtime import (
    ExecutedPluginCapability,
    LoadedPluginCapability,
    PluginExecutionPlan,
    build_plugin_execution_plan,
    execute_plugin_capability,
    execute_plugin_execution_request,
    load_plugin_capability,
)
from .rust_handoff import (
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
)
from .storage import (
    PluginExecutionRequestStorageAdapterManifest,
    PluginExecutionRequestStorageManifest,
    build_plugin_execution_request_storage_adapter_manifest,
    build_plugin_execution_request_storage_bundle,
    build_plugin_execution_request_storage_manifest,
    validate_plugin_execution_request_storage_adapter_manifest,
    validate_plugin_execution_request_storage_bundle,
    validate_plugin_execution_request_storage_manifest,
    write_plugin_execution_request_storage_bundle,
)

__all__ = [
    "PluginCapability",
    "PluginCompatibilityReport",
    "PluginManifest",
    "PluginExecutionPlan",
    "PluginExecutionApproval",
    "PluginExecutionRequest",
    "PluginExecutionRequestRevocation",
    "PluginExecutionRequestRevocationList",
    "PluginExecutionRequestLifecycleRecord",
    "PluginExecutionRequestLifecyclePolicyReport",
    "PluginExecutionRequestLifecycleSummary",
    "PluginExecutionRequestStorageAdapterManifest",
    "PluginExecutionRequestStorageManifest",
    "build_plugin_marketplace_catalog",
    "build_plugin_execution_plan",
    "build_plugin_execution_approval",
    "build_plugin_execution_request",
    "build_plugin_execution_request_revocation",
    "build_plugin_execution_request_revocation_list",
    "build_plugin_execution_request_lifecycle_policy_report",
    "build_plugin_execution_request_lifecycle_record",
    "build_plugin_execution_request_lifecycle_summary",
    "build_plugin_execution_request_storage_adapter_manifest",
    "build_plugin_execution_request_storage_manifest",
    "build_plugin_execution_request_storage_bundle",
    "build_rust_plugin_runtime_handoff",
    "build_rust_plugin_registry",
    "compatibility_report",
    "discover_plugin_manifests",
    "LoadedPluginCapability",
    "ExecutedPluginCapability",
    "execute_plugin_capability",
    "execute_plugin_execution_request",
    "load_plugin_capability",
    "validate_plugin_execution_request",
    "validate_plugin_execution_request_lifecycle_record",
    "validate_plugin_execution_request_lifecycle_summary",
    "validate_plugin_execution_request_revocation_list",
    "validate_plugin_execution_request_storage_adapter_manifest",
    "validate_plugin_execution_request_storage_bundle",
    "validate_plugin_execution_request_storage_manifest",
    "validate_plugin_manifest",
    "write_plugin_execution_request_storage_bundle",
    "PluginRuntimeExecutionPolicy",
    "PluginRuntimeLoadPolicy",
]
