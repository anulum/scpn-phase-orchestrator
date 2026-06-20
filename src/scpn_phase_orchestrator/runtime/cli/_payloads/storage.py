# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI storage manifest payload loaders

"""Storage manifest and storage-adapter payload loaders for the plugins CLI."""

from __future__ import annotations

from typing import Literal, cast

import click

from scpn_phase_orchestrator.plugins import (
    PluginExecutionRequestStorageAdapterManifest,
    PluginExecutionRequestStorageManifest,
)

from ._shared import _PLUGIN_KIND_OPTIONS, _require_sha256


def _load_storage_manifest_from_payload(
    manifest_payload: dict[str, object],
) -> PluginExecutionRequestStorageManifest:
    if (
        manifest_payload.get("schema")
        != "scpn_plugin_execution_request_storage_manifest_v1"
    ):
        raise click.ClickException(
            "storage manifest schema mismatch: expected "
            "scpn_plugin_execution_request_storage_manifest_v1"
        )
    request_hash = _require_sha256(manifest_payload.get("request_hash"), "request_hash")
    plan_hash = _require_sha256(manifest_payload.get("plan_hash"), "plan_hash")
    approval_hash = _require_sha256(
        manifest_payload.get("approval_hash"), "approval_hash"
    )
    target_hash = _require_sha256(manifest_payload.get("target_hash"), "target_hash")
    revocation_hash = _require_sha256(
        manifest_payload.get("revocation_hash"), "revocation_hash"
    )
    manifest_hash = _require_sha256(
        manifest_payload.get("manifest_hash"), "manifest_hash"
    )
    plugin = manifest_payload.get("plugin")
    kind = manifest_payload.get("kind")
    name = manifest_payload.get("name")
    operator_identity = manifest_payload.get("operator_identity")
    approval_reference = manifest_payload.get("approval_reference")
    storage_uri = manifest_payload.get("storage_uri")
    storage_backend = manifest_payload.get("storage_backend")
    retention_policy = manifest_payload.get("retention_policy")
    created_by = manifest_payload.get("created_by")
    revoked_request_hashes = manifest_payload.get("revoked_request_hashes")
    version = manifest_payload.get("version")
    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("storage_uri", storage_uri),
        ("storage_backend", storage_backend),
        ("retention_policy", retention_policy),
        ("created_by", created_by),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"storage manifest schema mismatch: {field_name} must be non-empty"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"storage manifest schema mismatch: unsupported kind {kind!r}"
        )
    if not isinstance(revoked_request_hashes, list) or not all(
        isinstance(item, str) for item in revoked_request_hashes
    ):
        raise click.ClickException(
            "storage manifest revoked_request_hashes must be a string list"
        )
    normalized_revocations = tuple(
        _require_sha256(item, "revoked request hash") for item in revoked_request_hashes
    )
    return PluginExecutionRequestStorageManifest(
        schema="scpn_plugin_execution_request_storage_manifest_v1",
        version=str(version),
        request_hash=request_hash,
        plan_hash=plan_hash,
        approval_hash=approval_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        storage_uri=str(storage_uri),
        storage_backend=str(storage_backend),
        retention_policy=str(retention_policy),
        created_by=str(created_by),
        revoked_request_hashes=normalized_revocations,
        revocation_hash=revocation_hash,
        manifest_hash=manifest_hash,
        audit_record=manifest_payload,
    )


def _load_storage_adapter_from_payload(
    adapter_payload: dict[str, object],
) -> PluginExecutionRequestStorageAdapterManifest:
    if (
        adapter_payload.get("schema")
        != "scpn_plugin_execution_request_storage_adapter_v1"
    ):
        raise click.ClickException(
            "storage adapter schema mismatch: expected "
            "scpn_plugin_execution_request_storage_adapter_v1"
        )
    request_hash = _require_sha256(adapter_payload.get("request_hash"), "request_hash")
    storage_manifest_hash = _require_sha256(
        adapter_payload.get("storage_manifest_hash"),
        "storage_manifest_hash",
    )
    bundle_hash = _require_sha256(adapter_payload.get("bundle_hash"), "bundle_hash")
    adapter_hash = _require_sha256(adapter_payload.get("adapter_hash"), "adapter_hash")
    storage_backend = adapter_payload.get("storage_backend")
    storage_uri = adapter_payload.get("storage_uri")
    storage_scheme = adapter_payload.get("storage_scheme")
    adapter_mode = adapter_payload.get("adapter_mode")
    write_performed = adapter_payload.get("write_performed")
    created_by = adapter_payload.get("created_by")
    version = adapter_payload.get("version")
    for field_name, value in (
        ("storage_backend", storage_backend),
        ("storage_uri", storage_uri),
        ("storage_scheme", storage_scheme),
        ("adapter_mode", adapter_mode),
        ("created_by", created_by),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"storage adapter schema mismatch: {field_name} must be non-empty"
            )
    if not isinstance(write_performed, bool):
        raise click.ClickException(
            "storage adapter schema mismatch: write_performed must be boolean"
        )
    return PluginExecutionRequestStorageAdapterManifest(
        schema="scpn_plugin_execution_request_storage_adapter_v1",
        version=str(version),
        request_hash=request_hash,
        storage_manifest_hash=storage_manifest_hash,
        storage_backend=str(storage_backend),
        storage_uri=str(storage_uri),
        storage_scheme=str(storage_scheme),
        adapter_mode=str(adapter_mode),
        bundle_hash=bundle_hash,
        write_performed=write_performed,
        created_by=str(created_by),
        adapter_hash=adapter_hash,
        audit_record=adapter_payload,
    )
