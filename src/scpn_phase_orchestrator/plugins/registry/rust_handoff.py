# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Rust plugin registry and runtime hand-off

"""Builders for the Rust plugin registry and runtime hand-off payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._shared import _VALID_KINDS, _record_hash
from .manifest import build_plugin_marketplace_catalog

if TYPE_CHECKING:
    from .manifest import PluginManifest


def build_rust_plugin_registry(
    manifests: tuple[PluginManifest, ...],
    *,
    include_incompatible: bool = False,
) -> dict[str, object]:
    """Build a flattened metadata registry for Rust-side dispatchers.

    The payload avoids Python object graphs and implementation imports. Rust
    consumers can parse capabilities, targets, channel/knob declarations, and
    compatibility flags from stable JSON before deciding whether to hand a
    target back to Python.

    Parameters
    ----------
    manifests : tuple[PluginManifest, ...]
        The manifest records.
    include_incompatible : bool
        Whether to include incompatible items.

    Returns
    -------
    dict[str, object]
        A flattened metadata registry for Rust-side dispatchers.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    """
    catalog = build_plugin_marketplace_catalog(
        manifests,
        include_incompatible=include_incompatible,
    )
    plugins = catalog["plugins"]
    if not isinstance(plugins, list):
        raise TypeError("plugin catalogue payload is malformed")

    capabilities: list[dict[str, object]] = []
    for plugin in plugins:
        manifest = plugin["manifest"]
        compatible = bool(plugin["compatible"])
        if not isinstance(manifest, dict):
            raise TypeError("plugin manifest payload is malformed")
        manifest_capabilities = manifest["capabilities"]
        if not isinstance(manifest_capabilities, list):
            raise TypeError("plugin capabilities payload is malformed")
        for capability in manifest_capabilities:
            if not isinstance(capability, dict):
                raise TypeError("plugin capability payload is malformed")
            capabilities.append(
                {
                    "plugin": manifest["name"],
                    "plugin_version": manifest["version"],
                    "package": manifest["package"],
                    "kind": capability["kind"],
                    "name": capability["name"],
                    "target": capability["target"],
                    "version": capability["version"],
                    "channels": capability["channels"],
                    "knobs": capability["knobs"],
                    "compatible": compatible,
                }
            )

    capabilities.sort(
        key=lambda item: (
            str(item["plugin"]),
            str(item["kind"]),
            str(item["name"]),
            str(item["version"]),
        )
    )
    return {
        "schema": "scpn_rust_plugin_registry_v1",
        "spo_version": catalog["spo_version"],
        "include_incompatible": include_incompatible,
        "capability_count": len(capabilities),
        "capabilities": capabilities,
        "capability_counts": catalog["capability_counts"],
    }


def build_rust_plugin_runtime_handoff(
    manifests: tuple[PluginManifest, ...],
    *,
    include_incompatible: bool = False,
) -> dict[str, object]:
    """Build a guarded metadata handoff for a future Rust runtime loader.

    The handoff is intentionally non-executing: it groups capabilities by kind,
    hashes every target record, carries compatibility state, and records that
    native/plugin loading remains disabled. Rust can consume this as a stable
    preflight contract before any future loader is allowed to resolve or call
    implementation targets.

    Parameters
    ----------
    manifests : tuple[PluginManifest, ...]
        The manifest records.
    include_incompatible : bool
        Whether to include incompatible items.

    Returns
    -------
    dict[str, object]
        A guarded metadata handoff for a future Rust runtime loader.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    """
    registry = build_rust_plugin_registry(
        manifests,
        include_incompatible=include_incompatible,
    )
    capabilities = registry["capabilities"]
    if not isinstance(capabilities, list):
        raise TypeError("rust plugin registry payload is malformed")

    dispatch_groups: dict[str, list[dict[str, object]]] = {
        kind: [] for kind in sorted(_VALID_KINDS)
    }
    blocked: list[dict[str, object]] = []
    target_hashes: dict[str, str] = {}
    for capability in capabilities:
        if not isinstance(capability, dict):
            raise TypeError("rust plugin capability payload is malformed")
        kind = str(capability["kind"])
        if kind not in _VALID_KINDS:
            raise TypeError("rust plugin capability kind is malformed")
        record = {
            "plugin": capability["plugin"],
            "plugin_version": capability["plugin_version"],
            "package": capability["package"],
            "kind": capability["kind"],
            "name": capability["name"],
            "target": capability["target"],
            "version": capability["version"],
            "channels": capability["channels"],
            "knobs": capability["knobs"],
            "compatible": capability["compatible"],
            "loading_permitted": False,
            "load_policy": "metadata_only_review",
        }
        target_hash = _record_hash(record)
        record["target_hash"] = target_hash
        target_hashes[
            (
                f"{record['plugin']}:{record['kind']}:"
                f"{record['name']}:{record['version']}"
            )
        ] = target_hash
        if record["compatible"] is True:
            dispatch_groups[kind].append(record)
        else:
            blocked.append(
                {
                    **record,
                    "blocked_reason": "incompatible_manifest",
                }
            )

    for records in dispatch_groups.values():
        records.sort(
            key=lambda item: (
                str(item["plugin"]),
                str(item["name"]),
                str(item["version"]),
            )
        )
    blocked.sort(
        key=lambda item: (
            str(item["plugin"]),
            str(item["kind"]),
            str(item["name"]),
            str(item["version"]),
        )
    )
    handoff = {
        "schema": "scpn_rust_plugin_runtime_handoff_v1",
        "registry_schema": registry["schema"],
        "spo_version": registry["spo_version"],
        "include_incompatible": include_incompatible,
        "loading_permitted": False,
        "load_policy": "metadata_only_review",
        "dispatch_groups": dispatch_groups,
        "target_hashes": dict(sorted(target_hashes.items())),
        "compatible_capability_count": sum(
            len(records) for records in dispatch_groups.values()
        ),
        "blocked_capability_count": len(blocked),
        "blocked_capabilities": blocked,
    }
    handoff["handoff_hash"] = _record_hash(handoff)
    return handoff
