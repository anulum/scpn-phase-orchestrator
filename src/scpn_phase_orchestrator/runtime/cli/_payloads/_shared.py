# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI payload shared loaders and hashing

"""Shared JSON loading, hashing, and plugin lookup for CLI payload loaders."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

import click

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginManifest,
)

_PLUGIN_KIND_OPTIONS: tuple[str, ...] = (
    "actuator",
    "bridge",
    "domainpack",
    "extractor",
    "monitor",
)


def _find_discovered_plugin(
    manifests: tuple[PluginManifest, ...],
    plugin_name: str,
) -> PluginManifest:
    matches = tuple(manifest for manifest in manifests if manifest.name == plugin_name)
    if not matches:
        raise click.ClickException(f"plugin {plugin_name!r} is not discovered")
    if len(matches) > 1:
        raise click.ClickException(
            f"multiple discovered plugin manifests matched {plugin_name!r}; "
            "selection is ambiguous"
        )
    return matches[0]


def _find_capability(
    manifest: PluginManifest,
    kind: str,
    capability_name: str,
) -> PluginCapability:
    matches = tuple(
        capability
        for capability in manifest.capabilities
        if capability.kind == kind and capability.name == capability_name
    )
    if not matches:
        raise click.ClickException(
            f"plugin {manifest.name!r} does not expose {kind}:{capability_name!r}"
        )
    if len(matches) > 1:
        raise click.ClickException(
            f"plugin {manifest.name!r} declares {kind}:{capability_name!r} "
            "more than once"
        )
    return matches[0]


def _normalize_approved_target_hashes(
    approved_target_hashes: tuple[str, ...],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for approved_hash in approved_target_hashes:
        if re.fullmatch(r"[0-9a-fA-F]{64}", approved_hash) is None:
            raise click.ClickException(
                f"approved target hash {approved_hash!r} is not a valid SHA-256 digest"
            )
        normalized.append(approved_hash.lower())
    return tuple(dict.fromkeys(normalized))


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise click.ClickException(
            f"{field_name} must be a 64-character SHA-256 digest"
        )
    if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise click.ClickException(
            f"{field_name} {value!r} is not a valid SHA-256 digest"
        )
    return value.lower()


def _load_json_file(path: Path, *, artifact: str = "plan") -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"cannot read plan file {path!s}: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"malformed {artifact} JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise click.ClickException(f"{artifact} payload must be a JSON object")
    return payload


def _record_hash(record: Mapping[str, object]) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _build_plan_payload_for_hash(plan_payload: dict[str, object]) -> dict[str, object]:
    if "plan_hash" not in plan_payload:
        raise click.ClickException("plan payload is missing required field plan_hash")
    if "target_hash" not in plan_payload:
        raise click.ClickException("plan payload is missing required field target_hash")
    payload_without_plan_hash = dict(plan_payload)
    payload_without_plan_hash.pop("plan_hash", None)
    payload_without_plan_hash.pop("manifest", None)
    payload_without_plan_hash.pop("capability", None)
    payload_without_plan_hash.pop("compatible", None)
    payload_without_plan_hash.pop("compatibility_reasons", None)
    return payload_without_plan_hash
