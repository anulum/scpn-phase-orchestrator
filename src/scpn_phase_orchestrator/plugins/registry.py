# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin manifest registry

"""Plugin manifest validation and entry-point discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal, TypeAlias

from scpn_phase_orchestrator import __version__

__all__ = [
    "PluginCapability",
    "PluginCompatibilityReport",
    "PluginManifest",
    "build_plugin_marketplace_catalog",
    "build_rust_plugin_runtime_handoff",
    "build_rust_plugin_registry",
    "compatibility_report",
    "discover_plugin_manifests",
    "validate_plugin_manifest",
]

PluginKind: TypeAlias = Literal[
    "domainpack",
    "extractor",
    "monitor",
    "actuator",
    "bridge",
]
_VALID_KINDS = {"domainpack", "extractor", "monitor", "actuator", "bridge"}
_ENTRY_POINT_GROUP = "scpn_phase_orchestrator.plugins"


@dataclass(frozen=True)
class PluginCapability:
    """One declared extension capability."""

    kind: PluginKind
    name: str
    target: str
    version: str = "0.1.0"
    channels: tuple[str, ...] = ()
    knobs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported plugin capability kind: {self.kind}")
        _require_identifier(self.name, "capability name")
        _require_non_empty(self.target, "capability target")
        _validate_version(self.version, "capability version")
        for channel in self.channels:
            _require_identifier(channel, "capability channel")
        for knob in self.knobs:
            _require_identifier(knob, "capability knob")


@dataclass(frozen=True)
class PluginManifest:
    """Versioned plugin manifest for marketplace and CI validation."""

    name: str
    version: str
    package: str
    capabilities: tuple[PluginCapability, ...]
    min_spo_version: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.name, "plugin name")
        _validate_version(self.version, "plugin version")
        _require_non_empty(self.package, "plugin package")
        if not self.capabilities:
            raise ValueError("plugin manifest requires at least one capability")
        if self.min_spo_version is not None:
            _validate_version(self.min_spo_version, "minimum SPO version")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> PluginManifest:
        """Construct a manifest from a JSON/YAML-style mapping."""
        capabilities = tuple(
            PluginCapability(
                kind=item["kind"],
                name=item["name"],
                target=item["target"],
                version=item.get("version", "0.1.0"),
                channels=tuple(item.get("channels", ())),
                knobs=tuple(item.get("knobs", ())),
            )
            for item in payload.get("capabilities", ())
        )
        return cls(
            name=payload["name"],
            version=payload["version"],
            package=payload["package"],
            capabilities=capabilities,
            min_spo_version=payload.get("min_spo_version"),
        )

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable manifest record."""
        return {
            "name": self.name,
            "version": self.version,
            "package": self.package,
            "min_spo_version": self.min_spo_version,
            "capabilities": [
                {
                    "kind": capability.kind,
                    "name": capability.name,
                    "target": capability.target,
                    "version": capability.version,
                    "channels": list(capability.channels),
                    "knobs": list(capability.knobs),
                }
                for capability in self.capabilities
            ],
        }


@dataclass(frozen=True)
class PluginCompatibilityReport:
    """Compatibility result for one plugin manifest."""

    manifest: PluginManifest
    compatible: bool
    reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable compatibility record."""
        return {
            "manifest": self.manifest.to_audit_record(),
            "compatible": self.compatible,
            "reasons": list(self.reasons),
        }


def validate_plugin_manifest(manifest: PluginManifest) -> PluginManifest:
    """Validate and return a plugin manifest.

    Dataclass construction performs structural validation; this function
    exists as a stable public compatibility gate for tooling.
    """
    compatibility = compatibility_report(manifest)
    if not compatibility.compatible:
        raise ValueError("; ".join(compatibility.reasons))
    return manifest


def compatibility_report(manifest: PluginManifest) -> PluginCompatibilityReport:
    """Return a non-throwing compatibility report for a manifest."""
    reasons: list[str] = []
    if manifest.min_spo_version is not None and _version_tuple(
        __version__
    ) < _version_tuple(manifest.min_spo_version):
        reasons.append(
            f"requires SPO >= {manifest.min_spo_version}, current {__version__}"
        )
    seen: set[tuple[str, str]] = set()
    for capability in manifest.capabilities:
        key = (capability.kind, capability.name)
        if key in seen:
            reasons.append(f"duplicate capability {capability.kind}:{capability.name}")
        seen.add(key)
        if capability.kind == "extractor" and not capability.channels:
            reasons.append(f"extractor {capability.name} must declare channels")
        if capability.kind == "monitor" and not capability.channels:
            reasons.append(f"monitor {capability.name} must declare channels")
        if capability.kind == "actuator" and not capability.knobs:
            reasons.append(f"actuator {capability.name} must declare knobs")
    return PluginCompatibilityReport(
        manifest=manifest,
        compatible=not reasons,
        reasons=tuple(reasons),
    )


def discover_plugin_manifests(
    entry_point_group: str = _ENTRY_POINT_GROUP,
) -> tuple[PluginManifest, ...]:
    """Discover plugin manifests from Python entry points."""
    entry_points = metadata.entry_points()
    selected = entry_points.select(group=entry_point_group)

    manifests: list[PluginManifest] = []
    for entry_point in selected:
        loaded = entry_point.load()
        payload = loaded() if callable(loaded) else loaded
        manifest = (
            payload
            if isinstance(payload, PluginManifest)
            else PluginManifest.from_mapping(payload)
        )
        validate_plugin_manifest(manifest)
        manifests.append(manifest)
    return tuple(manifests)


def build_plugin_marketplace_catalog(
    manifests: tuple[PluginManifest, ...],
    *,
    include_incompatible: bool = False,
) -> dict[str, object]:
    """Build a deterministic catalogue payload for marketplace tooling.

    The catalogue is metadata-only: it uses manifest declarations and
    compatibility reports, and it never imports plugin implementation targets.
    """
    reports = tuple(compatibility_report(manifest) for manifest in manifests)
    selected = (
        reports
        if include_incompatible
        else tuple(report for report in reports if report.compatible)
    )
    sorted_reports = tuple(
        sorted(
            selected,
            key=lambda report: (report.manifest.name, report.manifest.version),
        )
    )
    return {
        "schema_version": "1.0.0",
        "spo_version": __version__,
        "plugins": [report.to_audit_record() for report in sorted_reports],
        "plugin_count": len(sorted_reports),
        "compatible_count": sum(1 for report in reports if report.compatible),
        "incompatible_count": sum(1 for report in reports if not report.compatible),
        "capability_counts": _capability_counts(sorted_reports),
    }


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


def _require_identifier(value: str, label: str) -> None:
    _require_non_empty(value, label)
    if any(char.isspace() for char in value):
        raise ValueError(f"{label} must not contain whitespace")


def _require_non_empty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")


def _validate_version(value: str, label: str) -> None:
    parts = value.split(".")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"{label} must use MAJOR.MINOR.PATCH")


def _version_tuple(value: str) -> tuple[int, int, int]:
    core = value.split("+", maxsplit=1)[0]
    parts = core.split(".")
    if len(parts) < 3 or any(not part.isdigit() for part in parts[:3]):
        return (0, 0, 0)
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _capability_counts(
    reports: tuple[PluginCompatibilityReport, ...],
) -> dict[str, int]:
    counts = dict.fromkeys(sorted(_VALID_KINDS), 0)
    for report in reports:
        for capability in report.manifest.capabilities:
            counts[capability.kind] += 1
    return counts


def _record_hash(record: dict[str, object]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
