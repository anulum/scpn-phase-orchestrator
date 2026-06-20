# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI revocation payload loaders

"""Revocation and revocation-list payload loaders for the plugins CLI."""

from __future__ import annotations

from typing import Literal, cast

import click

from scpn_phase_orchestrator.plugins import (
    PluginExecutionRequestRevocation,
    PluginExecutionRequestRevocationList,
    validate_plugin_execution_request_revocation_list,
)

from ._shared import _PLUGIN_KIND_OPTIONS, _require_sha256


def _load_revocation_from_payload(
    revocation_payload: dict[str, object],
) -> PluginExecutionRequestRevocation:
    if (
        revocation_payload.get("schema")
        != "scpn_plugin_execution_request_revocation_v1"
    ):
        raise click.ClickException(
            "revocation schema mismatch: expected "
            "scpn_plugin_execution_request_revocation_v1"
        )
    request_hash = _require_sha256(
        revocation_payload.get("request_hash"), "request_hash"
    )
    plan_hash = _require_sha256(revocation_payload.get("plan_hash"), "plan_hash")
    approval_hash = _require_sha256(
        revocation_payload.get("approval_hash"), "approval_hash"
    )
    target_hash = _require_sha256(revocation_payload.get("target_hash"), "target_hash")
    revocation_hash = _require_sha256(
        revocation_payload.get("revocation_hash"), "revocation_hash"
    )
    plugin = revocation_payload.get("plugin")
    kind = revocation_payload.get("kind")
    name = revocation_payload.get("name")
    operator_identity = revocation_payload.get("operator_identity")
    approval_reference = revocation_payload.get("approval_reference")
    revoked_by = revocation_payload.get("revoked_by")
    revocation_reference = revocation_payload.get("revocation_reference")
    revocation_reason = revocation_payload.get("revocation_reason")
    revoked = revocation_payload.get("revoked")
    version = revocation_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("revoked_by", revoked_by),
        ("revocation_reference", revocation_reference),
        ("revocation_reason", revocation_reason),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"revocation schema mismatch: {field_name} must be non-empty"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"revocation schema mismatch: unsupported kind {kind!r}"
        )
    if revoked is not True:
        raise click.ClickException("revocation schema mismatch: revoked must be true")

    return PluginExecutionRequestRevocation(
        schema="scpn_plugin_execution_request_revocation_v1",
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
        revoked_by=str(revoked_by),
        revocation_reference=str(revocation_reference),
        revocation_reason=str(revocation_reason),
        revoked=True,
        revocation_hash=revocation_hash,
        audit_record=revocation_payload,
    )


def _load_revocation_list_from_payload(
    revocation_list_payload: dict[str, object],
) -> PluginExecutionRequestRevocationList:
    if (
        revocation_list_payload.get("schema")
        != "scpn_plugin_execution_request_revocation_list_v1"
    ):
        raise click.ClickException(
            "revocation list schema mismatch: expected "
            "scpn_plugin_execution_request_revocation_list_v1"
        )
    request_hashes = revocation_list_payload.get("request_hashes")
    revocation_hashes = revocation_list_payload.get("revocation_hashes")
    revocation_count = revocation_list_payload.get("revocation_count")
    created_by = revocation_list_payload.get("created_by")
    revocation_list_hash = _require_sha256(
        revocation_list_payload.get("revocation_list_hash"),
        "revocation_list_hash",
    )
    version = revocation_list_payload.get("version")
    if not isinstance(version, str) or not version:
        raise click.ClickException("revocation list version must be non-empty")
    if not isinstance(created_by, str) or not created_by:
        raise click.ClickException("revocation list created_by must be non-empty")
    if not isinstance(revocation_count, int) or revocation_count < 1:
        raise click.ClickException(
            "revocation list revocation_count must be a positive integer"
        )
    if not isinstance(request_hashes, list) or not all(
        isinstance(item, str) for item in request_hashes
    ):
        raise click.ClickException(
            "revocation list request_hashes must be a string list"
        )
    if not isinstance(revocation_hashes, list) or not all(
        isinstance(item, str) for item in revocation_hashes
    ):
        raise click.ClickException(
            "revocation list revocation_hashes must be a string list"
        )
    normalized_request_hashes = tuple(
        _require_sha256(item, "revoked request hash") for item in request_hashes
    )
    normalized_revocation_hashes = tuple(
        _require_sha256(item, "revocation hash") for item in revocation_hashes
    )
    return validate_plugin_execution_request_revocation_list(
        PluginExecutionRequestRevocationList(
            schema="scpn_plugin_execution_request_revocation_list_v1",
            version=version,
            request_hashes=normalized_request_hashes,
            revocation_hashes=normalized_revocation_hashes,
            revocation_count=revocation_count,
            created_by=created_by,
            revocation_list_hash=revocation_list_hash,
            audit_record=revocation_list_payload,
        )
    )
