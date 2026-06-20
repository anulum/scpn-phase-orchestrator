# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI lifecycle payload loaders

"""Lifecycle record, summary, policy-report, and drilldown payload loaders."""

from __future__ import annotations

from typing import Literal, cast

import click

from scpn_phase_orchestrator.plugins import (
    PluginExecutionRequestLifecycleRecord,
    PluginExecutionRequestLifecycleSummary,
)

from ._shared import _PLUGIN_KIND_OPTIONS, _require_sha256


def _load_lifecycle_from_payload(
    lifecycle_payload: dict[str, object],
) -> PluginExecutionRequestLifecycleRecord:
    if lifecycle_payload.get("schema") != "scpn_plugin_execution_request_lifecycle_v1":
        raise click.ClickException(
            "lifecycle schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_v1"
        )
    request_hash = _require_sha256(
        lifecycle_payload.get("request_hash"), "request_hash"
    )
    lifecycle_hash = _require_sha256(
        lifecycle_payload.get("lifecycle_hash"), "lifecycle_hash"
    )
    status = lifecycle_payload.get("status")
    plugin = lifecycle_payload.get("plugin")
    kind = lifecycle_payload.get("kind")
    name = lifecycle_payload.get("name")
    operator_identity = lifecycle_payload.get("operator_identity")
    approval_reference = lifecycle_payload.get("approval_reference")
    storage_manifest_hash = lifecycle_payload.get("storage_manifest_hash")
    storage_backend = lifecycle_payload.get("storage_backend")
    storage_uri = lifecycle_payload.get("storage_uri")
    revoked = lifecycle_payload.get("revoked")
    revocation_list_hash = lifecycle_payload.get("revocation_list_hash")
    revocation_hash = lifecycle_payload.get("revocation_hash")
    revoked_by = lifecycle_payload.get("revoked_by")
    revocation_reference = lifecycle_payload.get("revocation_reference")
    created_by = lifecycle_payload.get("created_by")
    version = lifecycle_payload.get("version")
    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("created_by", created_by),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"lifecycle schema mismatch: {field_name} must be non-empty"
            )
    if status not in {"approved", "stored", "revoked"}:
        raise click.ClickException("lifecycle schema mismatch: unsupported status")
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"lifecycle schema mismatch: unsupported kind {kind!r}"
        )
    if not isinstance(revoked, bool):
        raise click.ClickException("lifecycle schema mismatch: revoked must be boolean")

    def optional_hash(value: object, field_name: str) -> str | None:
        if value is None:
            return None
        return _require_sha256(value, field_name)

    return PluginExecutionRequestLifecycleRecord(
        schema="scpn_plugin_execution_request_lifecycle_v1",
        version=str(version),
        request_hash=request_hash,
        status=cast(Literal["approved", "stored", "revoked"], str(status)),
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        storage_manifest_hash=optional_hash(
            storage_manifest_hash, "storage_manifest_hash"
        ),
        storage_backend=str(storage_backend) if storage_backend is not None else None,
        storage_uri=str(storage_uri) if storage_uri is not None else None,
        revoked=revoked,
        revocation_list_hash=optional_hash(
            revocation_list_hash, "revocation_list_hash"
        ),
        revocation_hash=optional_hash(revocation_hash, "revocation_hash"),
        revoked_by=str(revoked_by) if revoked_by is not None else None,
        revocation_reference=(
            str(revocation_reference) if revocation_reference is not None else None
        ),
        created_by=str(created_by),
        lifecycle_hash=lifecycle_hash,
        audit_record=lifecycle_payload,
    )


def _load_lifecycle_summary_from_payload(
    summary_payload: dict[str, object],
) -> PluginExecutionRequestLifecycleSummary:
    if (
        summary_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_summary_v1"
    ):
        raise click.ClickException(
            "lifecycle summary schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_summary_v1"
        )
    summary_hash = _require_sha256(summary_payload.get("summary_hash"), "summary_hash")
    request_count = summary_payload.get("request_count")
    status_counts = summary_payload.get("status_counts")
    lifecycle_hashes = summary_payload.get("lifecycle_hashes")
    approved_request_hashes = summary_payload.get("approved_request_hashes")
    stored_request_hashes = summary_payload.get("stored_request_hashes")
    revoked_request_hashes = summary_payload.get("revoked_request_hashes")
    storage_missing_request_hashes = summary_payload.get(
        "storage_missing_request_hashes"
    )
    renewal_required_request_hashes = summary_payload.get(
        "renewal_required_request_hashes"
    )
    created_by = summary_payload.get("created_by")
    version = summary_payload.get("version")
    if not isinstance(version, str) or not version:
        raise click.ClickException("lifecycle summary version must be non-empty")
    if not isinstance(created_by, str) or not created_by:
        raise click.ClickException("lifecycle summary created_by must be non-empty")
    if not isinstance(request_count, int) or request_count < 1:
        raise click.ClickException(
            "lifecycle summary request_count must be a positive integer"
        )
    if not isinstance(status_counts, dict) or not all(
        isinstance(key, str) and isinstance(value, int)
        for key, value in status_counts.items()
    ):
        raise click.ClickException("lifecycle summary status_counts is malformed")

    def hash_list(value: object, field_name: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise click.ClickException(
                f"lifecycle summary {field_name} must be a string list"
            )
        return tuple(_require_sha256(item, field_name) for item in value)

    return PluginExecutionRequestLifecycleSummary(
        schema="scpn_plugin_execution_request_lifecycle_summary_v1",
        version=version,
        request_count=request_count,
        status_counts=cast(dict[str, int], status_counts),
        lifecycle_hashes=hash_list(lifecycle_hashes, "lifecycle_hashes"),
        approved_request_hashes=hash_list(
            approved_request_hashes, "approved_request_hashes"
        ),
        stored_request_hashes=hash_list(stored_request_hashes, "stored_request_hashes"),
        revoked_request_hashes=hash_list(
            revoked_request_hashes, "revoked_request_hashes"
        ),
        storage_missing_request_hashes=hash_list(
            storage_missing_request_hashes,
            "storage_missing_request_hashes",
        ),
        renewal_required_request_hashes=hash_list(
            renewal_required_request_hashes,
            "renewal_required_request_hashes",
        ),
        created_by=created_by,
        summary_hash=summary_hash,
        audit_record=summary_payload,
    )


def _load_lifecycle_policy_report_payload(
    policy_payload: dict[str, object],
) -> dict[str, object]:
    if (
        policy_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_policy_v1"
    ):
        raise click.ClickException(
            "lifecycle policy schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_policy_v1"
        )
    _require_sha256(policy_payload.get("summary_hash"), "summary_hash")
    _require_sha256(policy_payload.get("policy_hash"), "policy_hash")
    request_count = policy_payload.get("request_count")
    if not isinstance(request_count, int) or request_count < 1:
        raise click.ClickException(
            "lifecycle policy schema mismatch: request_count must be a positive integer"
        )
    for field_name in (
        "renewal_required_request_hashes",
        "missing_adapter_request_hashes",
        "external_write_followup_request_hashes",
    ):
        value = policy_payload.get(field_name)
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise click.ClickException(
                f"lifecycle policy schema mismatch: {field_name} must be a string list"
            )
        for item in value:
            _require_sha256(item, field_name)
    return policy_payload


def _load_lifecycle_multistore_drilldown_payload(
    drilldown_payload: dict[str, object],
) -> dict[str, object]:
    if (
        drilldown_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1"
    ):
        raise click.ClickException(
            "multi-store drilldown schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1"
        )
    _require_sha256(drilldown_payload.get("drilldown_hash"), "drilldown_hash")
    policy_count = drilldown_payload.get("policy_count")
    stores = drilldown_payload.get("stores")
    global_flagged = drilldown_payload.get("global_flagged_request_hashes")
    if not isinstance(policy_count, int) or policy_count < 1:
        raise click.ClickException(
            "multi-store drilldown schema mismatch: policy_count must be positive"
        )
    if not isinstance(stores, list) or not stores:
        raise click.ClickException(
            "multi-store drilldown schema mismatch: stores must be non-empty list"
        )
    if not isinstance(global_flagged, list) or not all(
        isinstance(item, str) for item in global_flagged
    ):
        raise click.ClickException(
            "multi-store drilldown schema mismatch: global_flagged_request_hashes "
            "must be a string list"
        )
    for item in global_flagged:
        _require_sha256(item, "global_flagged_request_hash")
    for store in stores:
        if not isinstance(store, dict):
            raise click.ClickException(
                "multi-store drilldown schema mismatch: store record must be object"
            )
        _require_sha256(store.get("store_hash"), "store_hash")
        _require_sha256(store.get("policy_hash"), "policy_hash")
        _require_sha256(store.get("summary_hash"), "summary_hash")
        for field_name in (
            "renewal_required_request_hashes",
            "storage_missing_request_hashes",
            "missing_adapter_request_hashes",
            "external_write_followup_request_hashes",
        ):
            value = store.get(field_name)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                raise click.ClickException(
                    f"multi-store drilldown schema mismatch: {field_name} "
                    "must be a string list"
                )
            for item in value:
                _require_sha256(item, field_name)
    return drilldown_payload
