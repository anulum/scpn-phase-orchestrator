# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI payload and plan loaders

"""JSON payload loaders and hash helpers shared by the plugins CLI commands.

These functions parse and validate the on-disk JSON artifacts the ``spo plugins``
commands consume — execution plans and approvals, execution requests, revocations
and revocation lists, storage and storage-adapter manifests, and the lifecycle,
remediation, and scheduler records — reconstructing the corresponding plugin
domain objects and raising :class:`click.ClickException` on malformed input. They
also provide the canonical record hashing (:func:`_record_hash`), SHA-256 digest
validation, plan-payload canonicalisation, and plugin/capability lookups the
command layer reuses. Pure functions with no Click command surface of their own.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import click

from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequest,
    PluginExecutionRequestLifecycleRecord,
    PluginExecutionRequestLifecycleSummary,
    PluginExecutionRequestRevocation,
    PluginExecutionRequestRevocationList,
    PluginExecutionRequestStorageAdapterManifest,
    PluginExecutionRequestStorageManifest,
    PluginManifest,
    validate_plugin_execution_request_revocation_list,
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


def _load_plan_from_payload(
    plan_payload: dict[str, object],
) -> tuple[PluginExecutionPlan, dict[str, object]]:
    if plan_payload.get("schema") != "scpn_plugin_runtime_execution_plan_v1":
        raise click.ClickException(
            "plan schema mismatch: expected scpn_plugin_runtime_execution_plan_v1"
        )

    manifest_payload = plan_payload.get("manifest")
    capability_payload = plan_payload.get("capability")
    if not isinstance(manifest_payload, dict):
        raise click.ClickException("plan payload is missing manifest object")
    if not isinstance(capability_payload, dict):
        raise click.ClickException("plan payload is missing capability object")

    try:
        manifest = PluginManifest.from_mapping(manifest_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"manifest schema mismatch: {exc}") from exc

    kind = capability_payload.get("kind")
    name = capability_payload.get("name")
    if not isinstance(kind, str) or not isinstance(name, str):
        raise click.ClickException(
            "capability schema mismatch: kind and name are required"
        )

    try:
        capability = _find_capability(manifest, kind, name)
    except click.ClickException as exc:
        raise click.ClickException(f"capability schema mismatch: {exc}") from exc

    raw_argument_count = plan_payload.get("argument_count")
    if not isinstance(raw_argument_count, int) or raw_argument_count < 0:
        raise click.ClickException(
            "plan schema mismatch: argument_count must be a non-negative integer"
        )
    raw_keyword_names = plan_payload.get("keyword_names")
    if not isinstance(raw_keyword_names, list):
        raise click.ClickException("plan schema mismatch: keyword_names must be a list")
    if not all(isinstance(name, str) for name in raw_keyword_names):
        raise click.ClickException(
            "plan schema mismatch: keyword_names must contain strings"
        )

    expected_plan_hash = _record_hash(
        _build_plan_payload_for_hash(plan_payload),
    )
    plan_hash = _require_sha256(plan_payload.get("plan_hash"), "plan_hash")
    if expected_plan_hash != plan_hash:
        raise click.ClickException("plan hash mismatch")

    target_hash = _require_sha256(plan_payload.get("target_hash"), "target_hash")

    audit_record = dict(plan_payload)
    audit_record["target_hash"] = target_hash
    execution_permitted = plan_payload.get("execution_permitted")
    if not isinstance(execution_permitted, bool):
        raise click.ClickException(
            "plan schema mismatch: execution_permitted must be a boolean"
        )
    if not execution_permitted:
        raise click.ClickException(
            "plugin runtime execution must be permitted for approval"
        )

    if plan_payload.get("require_target_hash_approval") is True:
        target_hash_approved = plan_payload.get("target_hash_approved")
        if target_hash_approved is not True:
            raise click.ClickException(
                f"plugin runtime target hash {target_hash} is not approved"
            )

    return PluginExecutionPlan(
        manifest=manifest,
        capability=capability,
        argument_count=raw_argument_count,
        keyword_names=tuple(raw_keyword_names),
        target_hash=target_hash,
        plan_hash=plan_hash,
        audit_record=audit_record,
    ), audit_record


def _load_approval_from_payload(
    approval_payload: dict[str, object],
) -> PluginExecutionApproval:
    if approval_payload.get("schema") != "scpn_plugin_execution_approval_v1":
        raise click.ClickException(
            "approval schema mismatch: expected scpn_plugin_execution_approval_v1"
        )

    plan_hash = _require_sha256(approval_payload.get("plan_hash"), "plan_hash")
    target_hash = _require_sha256(approval_payload.get("target_hash"), "target_hash")
    approval_hash = _require_sha256(
        approval_payload.get("approval_hash"), "approval_hash"
    )
    plugin = approval_payload.get("plugin")
    kind = approval_payload.get("kind")
    name = approval_payload.get("name")
    operator_identity = approval_payload.get("operator_identity")
    approval_reference = approval_payload.get("approval_reference")
    approval_reason = approval_payload.get("approval_reason")
    approved = approval_payload.get("approved")
    execution_permitted = approval_payload.get("execution_permitted")
    version = approval_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("approval_reason", approval_reason),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"approval schema mismatch: {field_name} must be a non-empty string"
            )

    if not isinstance(approved, bool):
        raise click.ClickException(
            "approval schema mismatch: approved must be a boolean"
        )
    if not isinstance(execution_permitted, bool):
        raise click.ClickException(
            "approval schema mismatch: execution_permitted must be a boolean"
        )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"approval schema mismatch: unsupported kind {kind!r}"
        )

    return PluginExecutionApproval(
        schema="scpn_plugin_execution_approval_v1",
        version=str(version),
        plan_hash=plan_hash,
        target_hash=target_hash,
        plugin=str(plugin),
        kind=cast(
            Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
            str(kind),
        ),
        name=str(name),
        operator_identity=str(operator_identity),
        approval_reference=str(approval_reference),
        approval_reason=str(approval_reason),
        approved=bool(approved),
        execution_permitted=bool(execution_permitted),
        approval_hash=approval_hash,
        audit_record=approval_payload,
    )


def _load_request_from_payload(
    request_payload: dict[str, object],
) -> PluginExecutionRequest:
    if request_payload.get("schema") != "scpn_plugin_runtime_execution_request_v1":
        raise click.ClickException(
            "request schema mismatch: expected scpn_plugin_runtime_execution_request_v1"
        )

    plan_hash = _require_sha256(request_payload.get("plan_hash"), "plan_hash")
    target_hash = _require_sha256(request_payload.get("target_hash"), "target_hash")
    approval_hash = _require_sha256(
        request_payload.get("approval_hash"), "approval_hash"
    )
    plugin = request_payload.get("plugin")
    kind = request_payload.get("kind")
    name = request_payload.get("name")
    operator_identity = request_payload.get("operator_identity")
    approval_reference = request_payload.get("approval_reference")
    loading_permitted = request_payload.get("loading_permitted")
    execution_permitted = request_payload.get("execution_permitted")
    require_target_hash_approval = request_payload.get("require_target_hash_approval")
    require_package_target = request_payload.get("require_package_target")
    approved_target_hashes = request_payload.get("approved_target_hashes")
    allowed_kinds = request_payload.get("allowed_kinds")
    version = request_payload.get("version")

    for field_name, value in (
        ("plugin", plugin),
        ("kind", kind),
        ("name", name),
        ("operator_identity", operator_identity),
        ("approval_reference", approval_reference),
        ("version", version),
    ):
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                f"request schema mismatch: {field_name} must be a non-empty string"
            )
    if kind not in _PLUGIN_KIND_OPTIONS:
        raise click.ClickException(
            f"request schema mismatch: unsupported kind {kind!r}"
        )
    for field_name, value in (
        ("loading_permitted", loading_permitted),
        ("execution_permitted", execution_permitted),
        ("require_target_hash_approval", require_target_hash_approval),
        ("require_package_target", require_package_target),
    ):
        if not isinstance(value, bool):
            raise click.ClickException(
                f"request schema mismatch: {field_name} must be a boolean"
            )
    if not isinstance(approved_target_hashes, list) or not all(
        isinstance(item, str) for item in approved_target_hashes
    ):
        raise click.ClickException(
            "request schema mismatch: approved_target_hashes must be a string list"
        )
    if not isinstance(allowed_kinds, list) or not all(
        isinstance(item, str) and item in _PLUGIN_KIND_OPTIONS for item in allowed_kinds
    ):
        raise click.ClickException(
            "request schema mismatch: allowed_kinds must be valid kind strings"
        )
    normalized_target_hashes = tuple(
        _require_sha256(item, "approved_target_hash") for item in approved_target_hashes
    )

    return PluginExecutionRequest(
        schema="scpn_plugin_runtime_execution_request_v1",
        version=str(version),
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
        loading_permitted=bool(loading_permitted),
        execution_permitted=bool(execution_permitted),
        require_target_hash_approval=bool(require_target_hash_approval),
        approved_target_hashes=normalized_target_hashes,
        allowed_kinds=cast(
            tuple[
                Literal["actuator", "bridge", "domainpack", "extractor", "monitor"],
                ...,
            ],
            tuple(allowed_kinds),
        ),
        require_package_target=bool(require_package_target),
        audit_record=request_payload,
    )


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


def _load_lifecycle_remediation_plan_payload(
    plan_payload: dict[str, object],
) -> dict[str, object]:
    if (
        plan_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
    ):
        raise click.ClickException(
            "remediation plan schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
        )
    _require_sha256(plan_payload.get("plan_hash"), "plan_hash")
    _require_sha256(plan_payload.get("drilldown_hash"), "drilldown_hash")
    action_count = plan_payload.get("action_count")
    actions = plan_payload.get("actions")
    if not isinstance(action_count, int) or action_count < 0:
        raise click.ClickException(
            "remediation plan schema mismatch: action_count must be non-negative"
        )
    if not isinstance(actions, list):
        raise click.ClickException(
            "remediation plan schema mismatch: actions must be a list"
        )
    if action_count != len(actions):
        raise click.ClickException(
            "remediation plan schema mismatch: action_count does not match actions"
        )
    for action in actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation plan schema mismatch: action must be an object"
            )
        _require_sha256(action.get("action_hash"), "action_hash")
        _require_sha256(action.get("request_hash"), "request_hash")
        _require_sha256(action.get("store_hash"), "store_hash")
        _require_sha256(action.get("policy_hash"), "policy_hash")
        _require_sha256(action.get("summary_hash"), "summary_hash")
        action_type = action.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation plan schema mismatch: unsupported action_type"
            )
        priority = action.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation plan schema mismatch: priority must be a positive integer"
            )
    return plan_payload


def _load_lifecycle_remediation_action_status_payload(
    status_payload: dict[str, object],
) -> dict[str, object]:
    if (
        status_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
    ):
        raise click.ClickException(
            "remediation action status schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
        )
    _require_sha256(status_payload.get("status_hash"), "status_hash")
    _require_sha256(status_payload.get("action_hash"), "action_hash")
    _require_sha256(status_payload.get("plan_hash"), "plan_hash")
    state = status_payload.get("state")
    if state not in {"pending", "in_progress", "completed", "blocked"}:
        raise click.ClickException(
            "remediation action status schema mismatch: unsupported state"
        )
    return status_payload


def _load_lifecycle_remediation_execution_dashboard_payload(
    dashboard_payload: dict[str, object],
) -> dict[str, object]:
    if (
        dashboard_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
    ):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
        )
    _require_sha256(dashboard_payload.get("execution_hash"), "execution_hash")
    _require_sha256(dashboard_payload.get("plan_hash"), "plan_hash")
    action_count = dashboard_payload.get("action_count")
    rows = dashboard_payload.get("rows")
    if not isinstance(action_count, int) or action_count < 0:
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: "
            "action_count must be non-negative"
        )
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: rows must be a list"
        )
    if action_count != len(rows):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: "
            "action_count does not match rows"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: row must be object"
            )
        _require_sha256(row.get("action_hash"), "action_hash")
        _require_sha256(row.get("request_hash"), "request_hash")
        state = row.get("state")
        if state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: unsupported state"
            )
        action_type = row.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: "
                "unsupported action_type"
            )
        priority = row.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: "
                "priority must be a positive integer"
            )
    return dashboard_payload


def _load_lifecycle_remediation_deployment_handoff_payload(
    handoff_payload: dict[str, object],
) -> dict[str, object]:
    if (
        handoff_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
    ):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
        )
    _require_sha256(handoff_payload.get("handoff_hash"), "handoff_hash")
    _require_sha256(handoff_payload.get("plan_hash"), "plan_hash")
    _require_sha256(handoff_payload.get("execution_hash"), "execution_hash")
    unresolved_count = handoff_payload.get("unresolved_action_count")
    handoff_actions = handoff_payload.get("handoff_actions")
    if not isinstance(unresolved_count, int) or unresolved_count < 0:
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "unresolved_action_count must be non-negative"
        )
    if not isinstance(handoff_actions, list):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "handoff_actions must be a list"
        )
    if unresolved_count != len(handoff_actions):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "unresolved_action_count does not match handoff_actions"
        )
    for action in handoff_actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: action must be object"
            )
        _require_sha256(action.get("handoff_action_hash"), "handoff_action_hash")
        _require_sha256(action.get("action_hash"), "action_hash")
        _require_sha256(action.get("request_hash"), "request_hash")
        action_type = action.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "unsupported action_type"
            )
        priority = action.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "priority must be a positive integer"
            )
        template = action.get("deployment_command_template")
        if not isinstance(template, str) or not template:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "deployment_command_template must be non-empty"
            )
    return handoff_payload


def _load_lifecycle_remediation_scheduler_queue_payload(
    queue_payload: dict[str, object],
) -> dict[str, object]:
    if (
        queue_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
    ):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
        )
    _require_sha256(queue_payload.get("scheduler_hash"), "scheduler_hash")
    _require_sha256(queue_payload.get("plan_hash"), "plan_hash")
    _require_sha256(queue_payload.get("execution_hash"), "execution_hash")
    _require_sha256(queue_payload.get("handoff_hash"), "handoff_hash")
    queue_entry_count = queue_payload.get("queue_entry_count")
    queue_entries = queue_payload.get("queue_entries")
    if not isinstance(queue_entry_count, int) or queue_entry_count < 0:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "queue_entry_count must be non-negative"
        )
    if not isinstance(queue_entries, list):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: queue_entries must be a list"
        )
    if queue_entry_count != len(queue_entries):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "queue_entry_count does not match queue_entries"
        )
    for entry in queue_entries:
        if not isinstance(entry, dict):
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: entry must be object"
            )
        _require_sha256(entry.get("entry_hash"), "entry_hash")
        _require_sha256(entry.get("handoff_action_hash"), "handoff_action_hash")
        _require_sha256(entry.get("action_hash"), "action_hash")
        _require_sha256(entry.get("request_hash"), "request_hash")
        action_type = entry.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: unsupported action_type"
            )
        priority = entry.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "priority must be a positive integer"
            )
        schedule_epoch = entry.get("schedule_epoch")
        if not isinstance(schedule_epoch, int) or schedule_epoch < 0:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "schedule_epoch must be non-negative integer"
            )
        template = entry.get("scheduler_command_template")
        if not isinstance(template, str) or not template:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "scheduler_command_template must be non-empty"
            )
    return queue_payload


def _load_lifecycle_remediation_scheduler_telemetry_payload(
    telemetry_payload: dict[str, object],
) -> dict[str, object]:
    if (
        telemetry_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
    ):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
        )
    _require_sha256(telemetry_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(telemetry_payload.get("plan_hash"), "plan_hash")
    _require_sha256(telemetry_payload.get("execution_hash"), "execution_hash")
    _require_sha256(telemetry_payload.get("handoff_hash"), "handoff_hash")
    _require_sha256(telemetry_payload.get("scheduler_hash"), "scheduler_hash")
    queue_entry_count = telemetry_payload.get("queue_entry_count")
    rows = telemetry_payload.get("rows")
    if not isinstance(queue_entry_count, int) or queue_entry_count < 0:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "queue_entry_count must be non-negative"
        )
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: rows must be a list"
        )
    if queue_entry_count != len(rows):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "queue_entry_count does not match rows"
        )
    return telemetry_payload


def _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
    handoff_payload: dict[str, object],
) -> dict[str, object]:
    if handoff_payload.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
    ):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        )
    _require_sha256(handoff_payload.get("adapter_handoff_hash"), "adapter_handoff_hash")
    _require_sha256(handoff_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(handoff_payload.get("plan_hash"), "plan_hash")
    _require_sha256(handoff_payload.get("execution_hash"), "execution_hash")
    entries = handoff_payload.get("entries")
    entry_count = handoff_payload.get("entry_count")
    if not isinstance(entry_count, int) or entry_count < 0:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "entry_count must be non-negative"
        )
    if not isinstance(entries, list):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: entries must be "
            "list"
        )
    if entry_count != len(entries):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "entry_count does not match entries"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise click.ClickException(
                "remediation scheduler adapter handoff schema mismatch: entry must be "
                "object"
            )
        _require_sha256(entry.get("adapter_entry_hash"), "adapter_entry_hash")
        _require_sha256(entry.get("entry_hash"), "entry_hash")
        _require_sha256(entry.get("action_hash"), "action_hash")
        _require_sha256(entry.get("request_hash"), "request_hash")
    return handoff_payload


def _load_lifecycle_remediation_scheduler_acknowledgement_payload(
    acknowledgement_payload: dict[str, object],
) -> dict[str, object]:
    if acknowledgement_payload.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_v1"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_v1"
        )
    _require_sha256(
        acknowledgement_payload.get("acknowledgement_hash"),
        "acknowledgement_hash",
    )
    _require_sha256(
        acknowledgement_payload.get("adapter_handoff_hash"),
        "adapter_handoff_hash",
    )
    _require_sha256(acknowledgement_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(acknowledgement_payload.get("plan_hash"), "plan_hash")
    _require_sha256(acknowledgement_payload.get("execution_hash"), "execution_hash")
    _require_sha256(
        acknowledgement_payload.get("adapter_entry_hash"),
        "adapter_entry_hash",
    )
    _require_sha256(acknowledgement_payload.get("entry_hash"), "entry_hash")
    _require_sha256(acknowledgement_payload.get("action_hash"), "action_hash")
    _require_sha256(acknowledgement_payload.get("request_hash"), "request_hash")
    state = acknowledgement_payload.get("state")
    if state not in {"in_progress", "completed", "blocked"}:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: unsupported state"
        )
    for field_name in ("acknowledged_by", "external_reference"):
        value = acknowledgement_payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                "remediation scheduler acknowledgement schema mismatch: "
                f"{field_name} must be non-empty"
            )
    return acknowledgement_payload
