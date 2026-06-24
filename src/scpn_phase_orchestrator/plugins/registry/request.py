# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin execution request and approval records

"""Plugin execution request and approval records with deterministic audit hashing."""

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
from .policy import PluginRuntimeExecutionPolicy

if TYPE_CHECKING:
    from ._shared import PluginKind
    from .runtime import PluginExecutionPlan


@dataclass(frozen=True)
class PluginExecutionApproval:
    """Operator approval artefact for a plugin execution plan."""

    schema: str
    version: str
    plan_hash: str
    target_hash: str
    plugin: str
    kind: PluginKind
    name: str
    operator_identity: str
    approval_reference: str
    approval_reason: str
    approved: bool
    execution_permitted: bool
    approval_hash: str
    audit_record: dict[str, object]


@dataclass(frozen=True)
class PluginExecutionRequest:
    """Operator-approved request artefact consumed by runtime execution."""

    schema: str
    version: str
    plan_hash: str
    approval_hash: str
    target_hash: str
    plugin: str
    kind: PluginKind
    name: str
    operator_identity: str
    approval_reference: str
    loading_permitted: bool
    execution_permitted: bool
    require_target_hash_approval: bool
    approved_target_hashes: tuple[str, ...]
    allowed_kinds: tuple[PluginKind, ...]
    require_package_target: bool
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_runtime_execution_request_v1":
            raise ValueError(
                "request schema must be scpn_plugin_runtime_execution_request_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("request version must be 1.0.0")
        _validate_sha256(self.plan_hash, "request plan hash")
        _validate_sha256(self.target_hash, "request target hash")
        _validate_sha256(self.approval_hash, "request approval hash")
        _require_identifier(self.plugin, "plugin")
        _require_identifier(self.name, "capability name")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported plugin capability kind: {self.kind}")
        if not isinstance(self.loading_permitted, bool):
            raise TypeError("loading_permitted must be a boolean")
        if not isinstance(self.execution_permitted, bool):
            raise TypeError("execution_permitted must be a boolean")
        if not isinstance(self.require_target_hash_approval, bool):
            raise TypeError("require_target_hash_approval must be a boolean")
        if not isinstance(self.require_package_target, bool):
            raise TypeError("require_package_target must be a boolean")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")
        for target_kind in self.allowed_kinds:
            if target_kind not in _VALID_KINDS:
                raise ValueError(f"unsupported runtime load kind: {target_kind}")
        for target_hash in self.approved_target_hashes:
            _validate_sha256(target_hash, "approved target hash")

    def to_execution_policy(self) -> PluginRuntimeExecutionPolicy:
        """Construct an explicit runtime execution policy from request fields.

        Returns
        -------
        PluginRuntimeExecutionPolicy
            An explicit runtime execution policy from request fields.
        """
        return PluginRuntimeExecutionPolicy(
            loading_permitted=self.loading_permitted,
            execution_permitted=self.execution_permitted,
            allowed_kinds=self.allowed_kinds,
            require_package_target=self.require_package_target,
            require_target_hash_approval=self.require_target_hash_approval,
            approved_target_hashes=self.approved_target_hashes,
        )


def validate_plugin_execution_request(
    request: PluginExecutionRequest,
    *,
    revoked_request_hashes: tuple[str, ...] = (),
) -> PluginExecutionRequest:
    """Validate a stored plugin execution request before runtime consumption.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.

    Returns
    -------
    PluginExecutionRequest
        A stored plugin execution request before runtime consumption.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    PermissionError
        If the operation is not permitted by policy.
    """
    request_hash = request.audit_record.get("request_hash")
    if not isinstance(request_hash, str):
        raise ValueError("request audit record is missing request_hash")
    _validate_sha256(request_hash, "request hash")
    normalised_revocations: set[str] = set()
    for revoked_hash in revoked_request_hashes:
        _validate_sha256(revoked_hash, "revoked request hash")
        normalised_revocations.add(revoked_hash.lower())
    if request_hash.lower() in normalised_revocations:
        raise PermissionError("plugin execution request has been revoked")
    if not request.require_target_hash_approval:
        raise PermissionError(
            "plugin execution request must require target hash approval"
        )
    if request.approved_target_hashes != (request.target_hash,):
        raise PermissionError(
            "plugin execution request must approve exactly the request target hash"
        )

    expected_record = _runtime_execution_request_audit_record(
        plan_hash=request.plan_hash,
        approval_hash=request.approval_hash,
        target_hash=request.target_hash,
        plugin=request.plugin,
        kind=request.kind,
        name=request.name,
        operator_identity=request.operator_identity,
        approval_reference=request.approval_reference,
        loading_permitted=request.loading_permitted,
        execution_permitted=request.execution_permitted,
        require_target_hash_approval=request.require_target_hash_approval,
        approved_target_hashes=request.approved_target_hashes,
        allowed_kinds=request.allowed_kinds,
        require_package_target=request.require_package_target,
    )
    if request.audit_record != expected_record:
        raise ValueError("request audit record mismatch")
    return request


def build_plugin_execution_approval(
    plan: PluginExecutionPlan,
    *,
    operator_identity: str,
    approval_reference: str,
    approval_reason: str,
) -> PluginExecutionApproval:
    """Build a deterministic operator approval artefact for an execution plan.

    Parameters
    ----------
    plan : PluginExecutionPlan
        The execution plan.
    operator_identity : str
        Identifier of the operator.
    approval_reference : str
        External approval reference.
    approval_reason : str
        Reason recorded with the approval.

    Returns
    -------
    PluginExecutionApproval
        A deterministic operator approval artefact for an execution plan.

    Raises
    ------
    PermissionError
        If the operation is not permitted by policy.
    """
    _validate_sha256(plan.plan_hash, "plan hash")
    _validate_sha256(plan.target_hash, "target hash")
    _require_identifier(operator_identity, "operator identity")
    _require_identifier(approval_reference, "approval reference")
    _require_non_empty(approval_reason, "approval reason")

    execution_permitted = bool(plan.audit_record.get("execution_permitted"))
    if not execution_permitted:
        raise PermissionError("plugin runtime execution must be permitted for approval")

    require_target_hash_approval = bool(
        plan.audit_record.get("require_target_hash_approval")
    )
    if require_target_hash_approval and not bool(
        plan.audit_record.get("target_hash_approved")
    ):
        raise PermissionError(
            f"plugin runtime target hash {plan.target_hash} is not approved"
        )

    audit_record = {
        "schema": "scpn_plugin_execution_approval_v1",
        "version": "1.0.0",
        "plan_hash": plan.plan_hash,
        "target_hash": plan.target_hash,
        "plugin": plan.manifest.name,
        "kind": plan.capability.kind,
        "name": plan.capability.name,
        "operator_identity": operator_identity,
        "approval_reference": approval_reference,
        "approval_reason": approval_reason,
        "approved": True,
        "execution_permitted": execution_permitted,
    }
    audit_record["approval_hash"] = _record_hash(audit_record)
    return PluginExecutionApproval(
        schema="scpn_plugin_execution_approval_v1",
        version="1.0.0",
        plan_hash=plan.plan_hash,
        target_hash=plan.target_hash,
        plugin=plan.manifest.name,
        kind=plan.capability.kind,
        name=plan.capability.name,
        operator_identity=operator_identity,
        approval_reference=approval_reference,
        approval_reason=approval_reason,
        approved=True,
        execution_permitted=execution_permitted,
        approval_hash=str(audit_record["approval_hash"]),
        audit_record=audit_record,
    )


def build_plugin_execution_request(
    plan: PluginExecutionPlan,
    approval: PluginExecutionApproval,
) -> PluginExecutionRequest:
    """Build a deterministic, non-importing request artefact for execution.

    Parameters
    ----------
    plan : PluginExecutionPlan
        The execution plan.
    approval : PluginExecutionApproval
        The execution-plan approval artefact.

    Returns
    -------
    PluginExecutionRequest
        A deterministic, non-importing request artefact for execution.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    PermissionError
        If the operation is not permitted by policy.
    """
    if approval.schema != "scpn_plugin_execution_approval_v1":
        raise ValueError("approval schema must be scpn_plugin_execution_approval_v1")
    if not approval.approved:
        raise PermissionError("approval must be granted before request construction")
    if not approval.execution_permitted:
        raise PermissionError(
            "approval execution_permitted must be true before request construction"
        )
    _validate_sha256(plan.plan_hash, "plan hash")
    _validate_sha256(plan.target_hash, "target hash")
    _validate_sha256(approval.plan_hash, "approval plan hash")
    _validate_sha256(approval.target_hash, "approval target hash")
    if plan.plan_hash != approval.plan_hash:
        raise ValueError("plan hash mismatch")
    if plan.target_hash != approval.target_hash:
        raise ValueError("target hash mismatch")
    if plan.manifest.name != approval.plugin:
        raise ValueError("plugin name mismatch")
    if plan.capability.kind != approval.kind:
        raise ValueError("capability kind mismatch")
    if plan.capability.name != approval.name:
        raise ValueError("capability name mismatch")

    loading_permitted = bool(plan.audit_record.get("loading_permitted"))
    execution_permitted = bool(plan.audit_record.get("execution_permitted"))
    if not loading_permitted:
        raise PermissionError("plugin runtime loading must be permitted before request")
    if not execution_permitted:
        raise PermissionError(
            "plugin runtime execution must be permitted before request"
        )

    require_target_hash_approval = bool(
        plan.audit_record.get("require_target_hash_approval")
    )
    if require_target_hash_approval and not bool(
        plan.audit_record.get("target_hash_approved")
    ):
        raise PermissionError(
            f"plugin runtime target hash {plan.target_hash} is not approved"
        )

    raw_allowed_kinds = plan.audit_record.get("allowed_kinds")
    if not isinstance(raw_allowed_kinds, (list, tuple)):
        raise ValueError("plan audit record allowed_kinds must be a sequence")
    validated_allowed_kinds: list[PluginKind] = []
    if not raw_allowed_kinds:
        raise ValueError("plan audit record is missing allowed_kinds")
    for raw_kind in raw_allowed_kinds:
        if raw_kind == "domainpack":
            validated_allowed_kinds.append("domainpack")
        elif raw_kind == "extractor":
            validated_allowed_kinds.append("extractor")
        elif raw_kind == "monitor":
            validated_allowed_kinds.append("monitor")
        elif raw_kind == "actuator":
            validated_allowed_kinds.append("actuator")
        elif raw_kind == "bridge":
            validated_allowed_kinds.append("bridge")
        else:
            raise ValueError(f"unsupported runtime load kind: {raw_kind}")
    allowed_kinds: tuple[PluginKind, ...] = tuple(validated_allowed_kinds)

    require_package_target = bool(plan.audit_record.get("require_package_target", True))
    approved_target_hashes = (plan.target_hash,)
    audit_record = _runtime_execution_request_audit_record(
        plan_hash=plan.plan_hash,
        approval_hash=approval.approval_hash,
        target_hash=plan.target_hash,
        plugin=plan.manifest.name,
        kind=plan.capability.kind,
        name=plan.capability.name,
        operator_identity=approval.operator_identity,
        approval_reference=approval.approval_reference,
        loading_permitted=loading_permitted,
        execution_permitted=execution_permitted,
        require_target_hash_approval=require_target_hash_approval,
        approved_target_hashes=approved_target_hashes,
        allowed_kinds=allowed_kinds,
        require_package_target=require_package_target,
    )
    return PluginExecutionRequest(
        schema="scpn_plugin_runtime_execution_request_v1",
        version="1.0.0",
        plan_hash=plan.plan_hash,
        approval_hash=approval.approval_hash,
        target_hash=plan.target_hash,
        plugin=plan.manifest.name,
        kind=plan.capability.kind,
        name=plan.capability.name,
        operator_identity=approval.operator_identity,
        approval_reference=approval.approval_reference,
        loading_permitted=loading_permitted,
        execution_permitted=execution_permitted,
        require_target_hash_approval=require_target_hash_approval,
        approved_target_hashes=approved_target_hashes,
        allowed_kinds=tuple(allowed_kinds),
        require_package_target=require_package_target,
        audit_record=audit_record,
    )


def _runtime_execution_request_audit_record(
    *,
    plan_hash: str,
    approval_hash: str,
    target_hash: str,
    plugin: str,
    kind: PluginKind,
    name: str,
    operator_identity: str,
    approval_reference: str,
    loading_permitted: bool,
    execution_permitted: bool,
    require_target_hash_approval: bool,
    approved_target_hashes: tuple[str, ...],
    allowed_kinds: tuple[str, ...],
    require_package_target: bool,
) -> dict[str, object]:
    """Build the audit record for a runtime execution request."""
    record = {
        "schema": "scpn_plugin_runtime_execution_request_v1",
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "approval_hash": approval_hash,
        "target_hash": target_hash,
        "plugin": plugin,
        "kind": kind,
        "name": name,
        "operator_identity": operator_identity,
        "approval_reference": approval_reference,
        "loading_permitted": loading_permitted,
        "execution_permitted": execution_permitted,
        "require_target_hash_approval": require_target_hash_approval,
        "approved_target_hashes": list(approved_target_hashes),
        "allowed_kinds": list(allowed_kinds),
        "require_package_target": require_package_target,
    }
    record["request_hash"] = _record_hash(record)
    return record
