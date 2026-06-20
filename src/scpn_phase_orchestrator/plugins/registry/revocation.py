# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin execution request revocation records

"""Plugin execution request revocation and revocation-list records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ._shared import (
    _VALID_KINDS,
    _record_hash,
    _require_identifier,
    _require_non_empty,
    _validate_sha256,
)
from .request import validate_plugin_execution_request

if TYPE_CHECKING:
    from ._shared import PluginKind
    from .request import PluginExecutionRequest


@dataclass(frozen=True)
class PluginExecutionRequestRevocation:
    """Operator lifecycle artefact that revokes an execution request hash."""

    schema: str
    version: str
    request_hash: str
    plan_hash: str
    approval_hash: str
    target_hash: str
    plugin: str
    kind: PluginKind
    name: str
    operator_identity: str
    approval_reference: str
    revoked_by: str
    revocation_reference: str
    revocation_reason: str
    revoked: bool
    revocation_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_revocation_v1":
            raise ValueError(
                "revocation schema must be scpn_plugin_execution_request_revocation_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("revocation version must be 1.0.0")
        _validate_sha256(self.request_hash, "revocation request hash")
        _validate_sha256(self.plan_hash, "revocation plan hash")
        _validate_sha256(self.approval_hash, "revocation approval hash")
        _validate_sha256(self.target_hash, "revocation target hash")
        _validate_sha256(self.revocation_hash, "revocation hash")
        _require_identifier(self.plugin, "plugin")
        _require_identifier(self.name, "capability name")
        _require_identifier(self.revoked_by, "revocation actor")
        _require_identifier(self.revocation_reference, "revocation reference")
        _require_non_empty(self.revocation_reason, "revocation reason")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported plugin capability kind: {self.kind}")
        if self.revoked is not True:
            raise PermissionError("revocation artefact must mark request revoked")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")


@dataclass(frozen=True)
class PluginExecutionRequestRevocationList:
    """Deployment-owned aggregate of request revocation artefacts."""

    schema: str
    version: str
    request_hashes: tuple[str, ...]
    revocation_hashes: tuple[str, ...]
    revocation_count: int
    created_by: str
    revocation_list_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_revocation_list_v1":
            raise ValueError(
                "revocation list schema must be "
                "scpn_plugin_execution_request_revocation_list_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("revocation list version must be 1.0.0")
        if self.revocation_count != len(self.request_hashes):
            raise ValueError("revocation count must match request hash count")
        if self.revocation_count != len(self.revocation_hashes):
            raise ValueError("revocation count must match revocation hash count")
        _require_identifier(self.created_by, "revocation list creator")
        _validate_sha256(self.revocation_list_hash, "revocation list hash")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")
        for request_hash in self.request_hashes:
            _validate_sha256(request_hash, "revoked request hash")
        for revocation_hash in self.revocation_hashes:
            _validate_sha256(revocation_hash, "revocation hash")

    def as_revoked_request_hashes(self) -> tuple[str, ...]:
        """Return the revoked request hash set for validation calls.

        Returns
        -------
        tuple[str, ...]
            The revoked request hash set for validation calls.
        """
        return self.request_hashes


def build_plugin_execution_request_revocation(
    request: PluginExecutionRequest,
    *,
    revoked_by: str,
    revocation_reference: str,
    revocation_reason: str,
) -> PluginExecutionRequestRevocation:
    """Build a deterministic revocation artefact for an execution request.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    revoked_by : str
        Identifier of the revoking actor.
    revocation_reference : str
        External revocation reference.
    revocation_reason : str
        Reason recorded with the revocation.

    Returns
    -------
    PluginExecutionRequestRevocation
        A deterministic revocation artefact for an execution request.
    """
    validate_plugin_execution_request(request)
    _require_identifier(revoked_by, "revocation actor")
    _require_identifier(revocation_reference, "revocation reference")
    _require_non_empty(revocation_reason, "revocation reason")
    request_hash = str(request.audit_record["request_hash"])
    audit_record = {
        "schema": "scpn_plugin_execution_request_revocation_v1",
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
        "revoked_by": revoked_by,
        "revocation_reference": revocation_reference,
        "revocation_reason": revocation_reason,
        "revoked": True,
    }
    audit_record["revocation_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestRevocation(
        schema="scpn_plugin_execution_request_revocation_v1",
        version="1.0.0",
        request_hash=request_hash,
        plan_hash=request.plan_hash,
        approval_hash=request.approval_hash,
        target_hash=request.target_hash,
        plugin=request.plugin,
        kind=request.kind,
        name=request.name,
        operator_identity=request.operator_identity,
        approval_reference=request.approval_reference,
        revoked_by=revoked_by,
        revocation_reference=revocation_reference,
        revocation_reason=revocation_reason,
        revoked=True,
        revocation_hash=str(audit_record["revocation_hash"]),
        audit_record=audit_record,
    )


def build_plugin_execution_request_revocation_list(
    revocations: tuple[PluginExecutionRequestRevocation, ...],
    *,
    created_by: str,
) -> PluginExecutionRequestRevocationList:
    """Build a deterministic deployment revocation-list artefact.

    Parameters
    ----------
    revocations : tuple[PluginExecutionRequestRevocation, ...]
        The revocation artefacts.
    created_by : str
        Identifier of the creating actor.

    Returns
    -------
    PluginExecutionRequestRevocationList
        A deterministic deployment revocation-list artefact.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    _require_identifier(created_by, "revocation list creator")
    if not revocations:
        raise ValueError("revocation list requires at least one revocation")
    records: list[dict[str, object]] = []
    for revocation in revocations:
        expected_record = dict(revocation.audit_record)
        expected_record.pop("revocation_hash", None)
        if _record_hash(expected_record) != revocation.revocation_hash:
            raise ValueError("revocation audit record mismatch")
        records.append(dict(revocation.audit_record))
    records.sort(
        key=lambda record: (
            str(record["request_hash"]),
            str(record["revocation_hash"]),
        )
    )
    request_hashes = tuple(str(record["request_hash"]) for record in records)
    if len(set(request_hashes)) != len(request_hashes):
        raise ValueError("revocation list contains duplicate request hashes")
    revocation_hashes = tuple(str(record["revocation_hash"]) for record in records)
    audit_record = {
        "schema": "scpn_plugin_execution_request_revocation_list_v1",
        "version": "1.0.0",
        "created_by": created_by,
        "revocation_count": len(records),
        "request_hashes": list(request_hashes),
        "revocation_hashes": list(revocation_hashes),
        "revocations": records,
    }
    audit_record["revocation_list_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestRevocationList(
        schema="scpn_plugin_execution_request_revocation_list_v1",
        version="1.0.0",
        request_hashes=request_hashes,
        revocation_hashes=revocation_hashes,
        revocation_count=len(records),
        created_by=created_by,
        revocation_list_hash=str(audit_record["revocation_list_hash"]),
        audit_record=audit_record,
    )


def validate_plugin_execution_request_revocation_list(
    revocation_list: PluginExecutionRequestRevocationList,
) -> PluginExecutionRequestRevocationList:
    """Validate a stored aggregate request-revocation list.

    Parameters
    ----------
    revocation_list : PluginExecutionRequestRevocationList
        The aggregate revocation list.

    Returns
    -------
    PluginExecutionRequestRevocationList
        A stored aggregate request-revocation list.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
    record = revocation_list.audit_record
    if record.get("schema") != "scpn_plugin_execution_request_revocation_list_v1":
        raise ValueError("revocation list schema mismatch")
    if record.get("version") != "1.0.0":
        raise ValueError("revocation list version must be 1.0.0")
    expected_hash = record.get("revocation_list_hash")
    if not isinstance(expected_hash, str):
        raise ValueError("revocation list is missing revocation_list_hash")
    _validate_sha256(expected_hash, "revocation list hash")
    payload = dict(record)
    payload.pop("revocation_list_hash", None)
    if _record_hash(payload) != expected_hash:
        raise ValueError("revocation list hash mismatch")
    request_hashes = record.get("request_hashes")
    revocation_hashes = record.get("revocation_hashes")
    revocations = record.get("revocations")
    if not isinstance(request_hashes, list) or not all(
        isinstance(item, str) for item in request_hashes
    ):
        raise ValueError("revocation list request_hashes must be a string list")
    if not isinstance(revocation_hashes, list) or not all(
        isinstance(item, str) for item in revocation_hashes
    ):
        raise ValueError("revocation list revocation_hashes must be a string list")
    if not isinstance(revocations, list) or not all(
        isinstance(item, dict) for item in revocations
    ):
        raise ValueError("revocation list revocations must be object records")
    if len(set(request_hashes)) != len(request_hashes):
        raise ValueError("revocation list contains duplicate request hashes")
    if revocation_list.request_hashes != tuple(request_hashes):
        raise ValueError("revocation list request hash field mismatch")
    if revocation_list.revocation_hashes != tuple(revocation_hashes):
        raise ValueError("revocation list revocation hash field mismatch")
    if revocation_list.revocation_count != len(request_hashes):
        raise ValueError("revocation list count mismatch")
    for revocation in revocations:
        revocation_hash = revocation.get("revocation_hash")
        if not isinstance(revocation_hash, str):
            raise ValueError("revocation record is missing revocation_hash")
        _validate_sha256(revocation_hash, "revocation hash")
        revocation_payload = dict(revocation)
        revocation_payload.pop("revocation_hash", None)
        if _record_hash(revocation_payload) != revocation_hash:
            raise ValueError("revocation audit record mismatch")
    return revocation_list
