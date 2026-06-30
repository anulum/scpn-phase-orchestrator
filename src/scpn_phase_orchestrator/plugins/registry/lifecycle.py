# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin execution request lifecycle records

"""Plugin execution request lifecycle records, summaries, and policy reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._shared import _VALID_KINDS, _record_hash, _require_identifier, _validate_sha256
from .request import validate_plugin_execution_request
from .revocation import validate_plugin_execution_request_revocation_list
from .storage import (
    validate_plugin_execution_request_storage_adapter_manifest,
    validate_plugin_execution_request_storage_manifest,
)

if TYPE_CHECKING:
    from ._shared import PluginKind
    from .request import PluginExecutionRequest
    from .revocation import PluginExecutionRequestRevocationList
    from .storage import (
        PluginExecutionRequestStorageAdapterManifest,
        PluginExecutionRequestStorageManifest,
    )


@dataclass(frozen=True)
class PluginExecutionRequestLifecycleRecord:
    """Operator-facing lifecycle status for one approved execution request."""

    schema: str
    version: str
    request_hash: str
    status: Literal["approved", "stored", "revoked"]
    plugin: str
    kind: PluginKind
    name: str
    operator_identity: str
    approval_reference: str
    storage_manifest_hash: str | None
    storage_backend: str | None
    storage_uri: str | None
    revoked: bool
    revocation_list_hash: str | None
    revocation_hash: str | None
    revoked_by: str | None
    revocation_reference: str | None
    created_by: str
    lifecycle_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_lifecycle_v1":
            raise ValueError(
                "lifecycle schema must be scpn_plugin_execution_request_lifecycle_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("lifecycle version must be 1.0.0")
        _validate_sha256(self.request_hash, "lifecycle request hash")
        _validate_sha256(self.lifecycle_hash, "lifecycle hash")
        if self.status not in {"approved", "stored", "revoked"}:
            raise ValueError("unsupported lifecycle status")
        _require_identifier(self.plugin, "plugin")
        _require_identifier(self.name, "capability name")
        _require_identifier(self.created_by, "lifecycle creator")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported plugin capability kind: {self.kind}")
        if not isinstance(self.revoked, bool):
            raise TypeError("revoked must be a boolean")
        for value, field_name in (
            (self.storage_manifest_hash, "storage manifest hash"),
            (self.revocation_list_hash, "revocation list hash"),
            (self.revocation_hash, "revocation hash"),
        ):
            if value is not None:
                _validate_sha256(value, field_name)
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")


@dataclass(frozen=True)
class PluginExecutionRequestLifecycleSummary:
    """Deterministic operator summary for lifecycle-review batches."""

    schema: str
    version: str
    request_count: int
    status_counts: dict[str, int]
    lifecycle_hashes: tuple[str, ...]
    approved_request_hashes: tuple[str, ...]
    stored_request_hashes: tuple[str, ...]
    revoked_request_hashes: tuple[str, ...]
    storage_missing_request_hashes: tuple[str, ...]
    renewal_required_request_hashes: tuple[str, ...]
    created_by: str
    summary_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_lifecycle_summary_v1":
            raise ValueError(
                "lifecycle summary schema must be "
                "scpn_plugin_execution_request_lifecycle_summary_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("lifecycle summary version must be 1.0.0")
        if self.request_count < 1:
            raise ValueError("lifecycle summary requires at least one request")
        _require_identifier(self.created_by, "lifecycle summary creator")
        _validate_sha256(self.summary_hash, "lifecycle summary hash")
        for lifecycle_hash in self.lifecycle_hashes:
            _validate_sha256(lifecycle_hash, "lifecycle hash")
        for request_hash in (
            *self.approved_request_hashes,
            *self.stored_request_hashes,
            *self.revoked_request_hashes,
            *self.storage_missing_request_hashes,
            *self.renewal_required_request_hashes,
        ):
            _validate_sha256(request_hash, "lifecycle summary request hash")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")


@dataclass(frozen=True)
class PluginExecutionRequestLifecyclePolicyReport:
    """Deterministic operator policy report for plugin lifecycle batches."""

    schema: str
    version: str
    summary_hash: str
    request_count: int
    policy_action_counts: dict[str, int]
    storage_missing_request_hashes: tuple[str, ...]
    renewal_required_request_hashes: tuple[str, ...]
    missing_adapter_request_hashes: tuple[str, ...]
    local_storage_request_hashes: tuple[str, ...]
    non_local_storage_request_hashes: tuple[str, ...]
    external_write_followup_request_hashes: tuple[str, ...]
    storage_adapter_hashes: tuple[str, ...]
    created_by: str
    policy_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_lifecycle_policy_v1":
            raise ValueError(
                "lifecycle policy schema must be "
                "scpn_plugin_execution_request_lifecycle_policy_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("lifecycle policy version must be 1.0.0")
        if self.request_count < 1:
            raise ValueError("lifecycle policy requires at least one request")
        _validate_sha256(self.summary_hash, "lifecycle summary hash")
        _validate_sha256(self.policy_hash, "lifecycle policy hash")
        _require_identifier(self.created_by, "lifecycle policy creator")
        for adapter_hash in self.storage_adapter_hashes:
            _validate_sha256(adapter_hash, "storage adapter hash")
        for request_hash in (
            *self.storage_missing_request_hashes,
            *self.renewal_required_request_hashes,
            *self.missing_adapter_request_hashes,
            *self.local_storage_request_hashes,
            *self.non_local_storage_request_hashes,
            *self.external_write_followup_request_hashes,
        ):
            _validate_sha256(request_hash, "lifecycle policy request hash")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")


def validate_plugin_execution_request_lifecycle_record(
    lifecycle: PluginExecutionRequestLifecycleRecord,
) -> PluginExecutionRequestLifecycleRecord:
    """Validate a stored plugin execution request lifecycle record.

    Parameters
    ----------
    lifecycle : PluginExecutionRequestLifecycleRecord
        The lifecycle status record.

    Returns
    -------
    PluginExecutionRequestLifecycleRecord
        A stored plugin execution request lifecycle record.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    record = lifecycle.audit_record
    if record.get("schema") != "scpn_plugin_execution_request_lifecycle_v1":
        raise ValueError("lifecycle schema mismatch")
    if record.get("version") != "1.0.0":
        raise ValueError("lifecycle version must be 1.0.0")
    expected_hash = record.get("lifecycle_hash")
    if not isinstance(expected_hash, str):
        raise ValueError("lifecycle record is missing lifecycle_hash")
    _validate_sha256(expected_hash, "lifecycle hash")
    payload = dict(record)
    payload.pop("lifecycle_hash", None)
    if _record_hash(payload) != expected_hash:
        raise ValueError("lifecycle hash mismatch")
    field_checks = {
        "request_hash": lifecycle.request_hash,
        "status": lifecycle.status,
        "plugin": lifecycle.plugin,
        "kind": lifecycle.kind,
        "name": lifecycle.name,
        "operator_identity": lifecycle.operator_identity,
        "approval_reference": lifecycle.approval_reference,
        "storage_manifest_hash": lifecycle.storage_manifest_hash,
        "storage_backend": lifecycle.storage_backend,
        "storage_uri": lifecycle.storage_uri,
        "revoked": lifecycle.revoked,
        "revocation_list_hash": lifecycle.revocation_list_hash,
        "revocation_hash": lifecycle.revocation_hash,
        "revoked_by": lifecycle.revoked_by,
        "revocation_reference": lifecycle.revocation_reference,
        "created_by": lifecycle.created_by,
    }
    for field_name, expected in field_checks.items():
        if record.get(field_name) != expected:
            raise ValueError(f"lifecycle {field_name} field mismatch")
    return lifecycle


def validate_plugin_execution_request_lifecycle_summary(
    summary: PluginExecutionRequestLifecycleSummary,
) -> PluginExecutionRequestLifecycleSummary:
    """Validate a stored plugin execution request lifecycle summary.

    Parameters
    ----------
    summary : PluginExecutionRequestLifecycleSummary
        The lifecycle batch summary.

    Returns
    -------
    PluginExecutionRequestLifecycleSummary
        A stored plugin execution request lifecycle summary.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    record = summary.audit_record
    if record.get("schema") != "scpn_plugin_execution_request_lifecycle_summary_v1":
        raise ValueError("lifecycle summary schema mismatch")
    if record.get("version") != "1.0.0":
        raise ValueError("lifecycle summary version must be 1.0.0")
    expected_hash = record.get("summary_hash")
    if not isinstance(expected_hash, str):
        raise ValueError("lifecycle summary is missing summary_hash")
    _validate_sha256(expected_hash, "lifecycle summary hash")
    payload = dict(record)
    payload.pop("summary_hash", None)
    if _record_hash(payload) != expected_hash:
        raise ValueError("lifecycle summary hash mismatch")
    for field_name, expected in (
        ("request_count", summary.request_count),
        ("status_counts", summary.status_counts),
        ("lifecycle_hashes", list(summary.lifecycle_hashes)),
        ("approved_request_hashes", list(summary.approved_request_hashes)),
        ("stored_request_hashes", list(summary.stored_request_hashes)),
        ("revoked_request_hashes", list(summary.revoked_request_hashes)),
        (
            "storage_missing_request_hashes",
            list(summary.storage_missing_request_hashes),
        ),
        (
            "renewal_required_request_hashes",
            list(summary.renewal_required_request_hashes),
        ),
        ("created_by", summary.created_by),
    ):
        if record.get(field_name) != expected:
            raise ValueError(f"lifecycle summary {field_name} field mismatch")
    return summary


def build_plugin_execution_request_lifecycle_record(
    request: PluginExecutionRequest,
    *,
    created_by: str,
    storage_manifest: PluginExecutionRequestStorageManifest | None = None,
    revocation_list: PluginExecutionRequestRevocationList | None = None,
) -> PluginExecutionRequestLifecycleRecord:
    """Build a deterministic operator lifecycle status record.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    created_by : str
        Identifier of the creating actor.
    storage_manifest : PluginExecutionRequestStorageManifest | None
        The storage manifest.
    revocation_list : PluginExecutionRequestRevocationList | None
        The aggregate revocation list.

    Returns
    -------
    PluginExecutionRequestLifecycleRecord
        A deterministic operator lifecycle status record.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    validate_plugin_execution_request(request)
    _require_identifier(created_by, "lifecycle creator")
    request_hash = str(request.audit_record["request_hash"])
    storage_manifest_hash: str | None = None
    storage_backend: str | None = None
    storage_uri: str | None = None
    if storage_manifest is not None:
        validate_plugin_execution_request_storage_manifest(request, storage_manifest)
        storage_manifest_hash = storage_manifest.manifest_hash
        storage_backend = storage_manifest.storage_backend
        storage_uri = storage_manifest.storage_uri

    revocation_list_hash: str | None = None
    revocation_hash: str | None = None
    revoked_by: str | None = None
    revocation_reference: str | None = None
    if revocation_list is not None:
        validate_plugin_execution_request_revocation_list(revocation_list)
        revocation_list_hash = revocation_list.revocation_list_hash
        revocations = revocation_list.audit_record.get("revocations")
        # Defensive: validate_..._revocation_list above already guarantees
        # ``revocations`` is a list of dict records.
        if not isinstance(revocations, (list, tuple)):  # pragma: no cover
            raise ValueError("revocation list revocations must be a sequence")
        for revocation in revocations:
            if not isinstance(revocation, dict):  # pragma: no cover
                raise ValueError("revocation list revocations must be object records")
            if revocation.get("request_hash") == request_hash:
                revocation_hash = str(revocation["revocation_hash"])
                revoked_by = str(revocation["revoked_by"])
                revocation_reference = str(revocation["revocation_reference"])
                break

    revoked = revocation_hash is not None
    status: Literal["approved", "stored", "revoked"]
    if revoked:
        status = "revoked"
    elif storage_manifest is not None:
        status = "stored"
    else:
        status = "approved"
    audit_record: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_v1",
        "version": "1.0.0",
        "request_hash": request_hash,
        "plan_hash": request.plan_hash,
        "approval_hash": request.approval_hash,
        "target_hash": request.target_hash,
        "plugin": request.plugin,
        "kind": request.kind,
        "name": request.name,
        "operator_identity": request.operator_identity,
        "approval_reference": request.approval_reference,
        "status": status,
        "loading_permitted": request.loading_permitted,
        "execution_permitted": request.execution_permitted,
        "storage_manifest_hash": storage_manifest_hash,
        "storage_backend": storage_backend,
        "storage_uri": storage_uri,
        "revoked": revoked,
        "revocation_list_hash": revocation_list_hash,
        "revocation_hash": revocation_hash,
        "revoked_by": revoked_by,
        "revocation_reference": revocation_reference,
        "created_by": created_by,
    }
    audit_record["lifecycle_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestLifecycleRecord(
        schema="scpn_plugin_execution_request_lifecycle_v1",
        version="1.0.0",
        request_hash=request_hash,
        status=status,
        plugin=request.plugin,
        kind=request.kind,
        name=request.name,
        operator_identity=request.operator_identity,
        approval_reference=request.approval_reference,
        storage_manifest_hash=storage_manifest_hash,
        storage_backend=storage_backend,
        storage_uri=storage_uri,
        revoked=revoked,
        revocation_list_hash=revocation_list_hash,
        revocation_hash=revocation_hash,
        revoked_by=revoked_by,
        revocation_reference=revocation_reference,
        created_by=created_by,
        lifecycle_hash=str(audit_record["lifecycle_hash"]),
        audit_record=audit_record,
    )


def build_plugin_execution_request_lifecycle_summary(
    lifecycle_records: tuple[PluginExecutionRequestLifecycleRecord, ...],
    *,
    created_by: str,
) -> PluginExecutionRequestLifecycleSummary:
    """Build a deterministic batch summary for operator lifecycle review.

    Parameters
    ----------
    lifecycle_records : tuple[PluginExecutionRequestLifecycleRecord, ...]
        Lifecycle status records.
    created_by : str
        Identifier of the creating actor.

    Returns
    -------
    PluginExecutionRequestLifecycleSummary
        A deterministic batch summary for operator lifecycle review.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    _require_identifier(created_by, "lifecycle summary creator")
    if not lifecycle_records:
        raise ValueError("lifecycle summary requires at least one record")
    validated = tuple(
        validate_plugin_execution_request_lifecycle_record(record)
        for record in lifecycle_records
    )
    request_hashes = tuple(record.request_hash for record in validated)
    if len(set(request_hashes)) != len(request_hashes):
        raise ValueError("lifecycle summary contains duplicate request hashes")
    records = tuple(sorted(validated, key=lambda item: item.request_hash))
    approved = tuple(
        record.request_hash for record in records if record.status == "approved"
    )
    stored = tuple(
        record.request_hash for record in records if record.status == "stored"
    )
    revoked = tuple(
        record.request_hash for record in records if record.status == "revoked"
    )
    status_counts = {
        "approved": len(approved),
        "stored": len(stored),
        "revoked": len(revoked),
    }
    audit_record: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_summary_v1",
        "version": "1.0.0",
        "request_count": len(records),
        "status_counts": status_counts,
        "lifecycle_hashes": [record.lifecycle_hash for record in records],
        "request_hashes": [record.request_hash for record in records],
        "approved_request_hashes": list(approved),
        "stored_request_hashes": list(stored),
        "revoked_request_hashes": list(revoked),
        "storage_missing_request_hashes": list(approved),
        "renewal_required_request_hashes": list(revoked),
        "created_by": created_by,
    }
    audit_record["summary_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestLifecycleSummary(
        schema="scpn_plugin_execution_request_lifecycle_summary_v1",
        version="1.0.0",
        request_count=len(records),
        status_counts=status_counts,
        lifecycle_hashes=tuple(record.lifecycle_hash for record in records),
        approved_request_hashes=approved,
        stored_request_hashes=stored,
        revoked_request_hashes=revoked,
        storage_missing_request_hashes=approved,
        renewal_required_request_hashes=revoked,
        created_by=created_by,
        summary_hash=str(audit_record["summary_hash"]),
        audit_record=audit_record,
    )


def build_plugin_execution_request_lifecycle_policy_report(
    summary: PluginExecutionRequestLifecycleSummary,
    *,
    created_by: str,
    storage_adapters: tuple[PluginExecutionRequestStorageAdapterManifest, ...] = (),
) -> PluginExecutionRequestLifecyclePolicyReport:
    """Build a deterministic operator policy report for lifecycle batches.

    Parameters
    ----------
    summary : PluginExecutionRequestLifecycleSummary
        The lifecycle summary.
    created_by : str
        Identifier of the creating actor.
    storage_adapters : tuple[PluginExecutionRequestStorageAdapterManifest, ...]
        Storage-adapter manifests.

    Returns
    -------
    PluginExecutionRequestLifecyclePolicyReport
        A deterministic operator policy report for lifecycle batches.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    validate_plugin_execution_request_lifecycle_summary(summary)
    _require_identifier(created_by, "lifecycle policy creator")
    request_hashes = set(summary.approved_request_hashes)
    request_hashes.update(summary.stored_request_hashes)
    request_hashes.update(summary.revoked_request_hashes)
    adapter_by_request: dict[str, PluginExecutionRequestStorageAdapterManifest] = {}
    for adapter in storage_adapters:
        validate_plugin_execution_request_storage_adapter_manifest(adapter)
        if adapter.request_hash not in request_hashes:
            raise ValueError("storage adapter request hash is not in lifecycle summary")
        if adapter.request_hash in adapter_by_request:
            raise ValueError("duplicate storage adapter request hash")
        adapter_by_request[adapter.request_hash] = adapter
    storage_relevant = tuple(
        sorted(
            (*summary.storage_missing_request_hashes, *summary.stored_request_hashes)
        )
    )
    missing_adapters = tuple(
        request_hash
        for request_hash in storage_relevant
        if request_hash not in adapter_by_request
    )
    local_storage = tuple(
        sorted(
            request_hash
            for request_hash, adapter in adapter_by_request.items()
            if adapter.storage_backend == "local_file"
        )
    )
    non_local_storage = tuple(
        sorted(
            request_hash
            for request_hash, adapter in adapter_by_request.items()
            if adapter.storage_backend != "local_file"
        )
    )
    external_followup = tuple(
        sorted(
            request_hash
            for request_hash, adapter in adapter_by_request.items()
            if adapter.storage_backend != "local_file" and not adapter.write_performed
        )
    )
    action_counts = {
        "confirm_external_write": len(external_followup),
        "persist_request": len(summary.storage_missing_request_hashes),
        "register_storage_adapter": len(missing_adapters),
        "renew_approval": len(summary.renewal_required_request_hashes),
    }
    audit_record: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_policy_v1",
        "version": "1.0.0",
        "summary_hash": summary.summary_hash,
        "request_count": summary.request_count,
        "status_counts": dict(summary.status_counts),
        "policy_action_counts": action_counts,
        "storage_missing_request_hashes": list(summary.storage_missing_request_hashes),
        "renewal_required_request_hashes": list(
            summary.renewal_required_request_hashes
        ),
        "missing_adapter_request_hashes": list(missing_adapters),
        "local_storage_request_hashes": list(local_storage),
        "non_local_storage_request_hashes": list(non_local_storage),
        "external_write_followup_request_hashes": list(external_followup),
        "storage_adapter_hashes": sorted(
            adapter.adapter_hash for adapter in storage_adapters
        ),
        "created_by": created_by,
    }
    audit_record["policy_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestLifecyclePolicyReport(
        schema="scpn_plugin_execution_request_lifecycle_policy_v1",
        version="1.0.0",
        summary_hash=summary.summary_hash,
        request_count=summary.request_count,
        policy_action_counts=action_counts,
        storage_missing_request_hashes=summary.storage_missing_request_hashes,
        renewal_required_request_hashes=summary.renewal_required_request_hashes,
        missing_adapter_request_hashes=missing_adapters,
        local_storage_request_hashes=local_storage,
        non_local_storage_request_hashes=non_local_storage,
        external_write_followup_request_hashes=external_followup,
        storage_adapter_hashes=tuple(
            sorted(adapter.adapter_hash for adapter in storage_adapters)
        ),
        created_by=created_by,
        policy_hash=str(audit_record["policy_hash"]),
        audit_record=audit_record,
    )
