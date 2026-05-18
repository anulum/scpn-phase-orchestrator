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
    PluginCapability,
    PluginCompatibilityReport,
    PluginManifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    compatibility_report,
    discover_plugin_manifests,
    validate_plugin_manifest,
)

__all__ = [
    "PluginCapability",
    "PluginCompatibilityReport",
    "PluginManifest",
    "build_plugin_marketplace_catalog",
    "build_rust_plugin_registry",
    "compatibility_report",
    "discover_plugin_manifests",
    "validate_plugin_manifest",
]
