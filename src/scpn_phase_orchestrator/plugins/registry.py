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
import importlib
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

from scpn_phase_orchestrator import __version__

__all__ = [
    "PluginCapability",
    "PluginCompatibilityReport",
    "PluginManifest",
    "PluginExecutionPlan",
    "PluginExecutionApproval",
    "PluginExecutionRequest",
    "PluginExecutionRequestRevocation",
    "PluginExecutionRequestRevocationList",
    "PluginExecutionRequestLifecycleRecord",
    "PluginExecutionRequestLifecyclePolicyReport",
    "PluginExecutionRequestLifecycleSummary",
    "PluginExecutionRequestStorageAdapterManifest",
    "PluginExecutionRequestStorageManifest",
    "build_plugin_marketplace_catalog",
    "build_plugin_execution_plan",
    "build_plugin_execution_approval",
    "build_plugin_execution_request",
    "build_plugin_execution_request_revocation",
    "build_plugin_execution_request_revocation_list",
    "build_plugin_execution_request_lifecycle_policy_report",
    "build_plugin_execution_request_lifecycle_record",
    "build_plugin_execution_request_lifecycle_summary",
    "build_plugin_execution_request_storage_adapter_manifest",
    "build_plugin_execution_request_storage_manifest",
    "build_plugin_execution_request_storage_bundle",
    "build_rust_plugin_runtime_handoff",
    "build_rust_plugin_registry",
    "compatibility_report",
    "discover_plugin_manifests",
    "LoadedPluginCapability",
    "ExecutedPluginCapability",
    "execute_plugin_capability",
    "execute_plugin_execution_request",
    "load_plugin_capability",
    "validate_plugin_execution_request",
    "validate_plugin_execution_request_lifecycle_record",
    "validate_plugin_execution_request_revocation_list",
    "validate_plugin_execution_request_storage_adapter_manifest",
    "validate_plugin_execution_request_storage_bundle",
    "validate_plugin_execution_request_storage_manifest",
    "validate_plugin_manifest",
    "write_plugin_execution_request_storage_bundle",
    "PluginRuntimeExecutionPolicy",
    "PluginRuntimeLoadPolicy",
]

PluginKind: TypeAlias = Literal[
    "domainpack",
    "extractor",
    "monitor",
    "actuator",
    "bridge",
]
_VALID_KINDS = {"domainpack", "extractor", "monitor", "actuator", "bridge"}
_DEFAULT_RUNTIME_LOAD_KINDS: tuple[PluginKind, ...] = (
    "actuator",
    "bridge",
    "extractor",
    "monitor",
)
_DEFAULT_RUNTIME_LOAD_POLICY: PluginRuntimeLoadPolicy
_ENTRY_POINT_GROUP = "scpn_phase_orchestrator.plugins"
_STORAGE_BACKEND_SCHEMES: dict[str, tuple[str, ...]] = {
    "local_file": ("", "file"),
    "s3_object": ("s3",),
    "gcs_object": ("gs",),
    "azure_blob": ("az", "azure"),
    "oci_object": ("oci",),
    "https_api": ("https",),
}
_NON_LOCAL_STORAGE_BACKENDS = frozenset(
    backend for backend in _STORAGE_BACKEND_SCHEMES if backend != "local_file"
)


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


@dataclass(frozen=True)
class PluginRuntimeLoadPolicy:
    """Explicit policy gate for Python-owned plugin runtime loading."""

    loading_permitted: bool = False
    allowed_kinds: tuple[PluginKind, ...] = _DEFAULT_RUNTIME_LOAD_KINDS
    require_package_target: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.loading_permitted, bool):
            raise TypeError("loading_permitted must be a boolean")
        if not isinstance(self.require_package_target, bool):
            raise TypeError("require_package_target must be a boolean")
        if not self.allowed_kinds:
            raise ValueError("allowed_kinds must not be empty")
        for kind in self.allowed_kinds:
            if kind not in _VALID_KINDS:
                raise ValueError(f"unsupported runtime load kind: {kind}")


@dataclass(frozen=True)
class LoadedPluginCapability:
    """Resolved plugin runtime target with deterministic audit evidence."""

    manifest: PluginManifest
    capability: PluginCapability
    target_object: object
    audit_record: dict[str, object]


@dataclass(frozen=True)
class PluginRuntimeExecutionPolicy:
    """Explicit policy gate for invoking Python-owned plugin runtime targets."""

    loading_permitted: bool = False
    execution_permitted: bool = False
    allowed_kinds: tuple[PluginKind, ...] = _DEFAULT_RUNTIME_LOAD_KINDS
    require_package_target: bool = True
    approved_target_hashes: tuple[str, ...] = ()
    require_target_hash_approval: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.execution_permitted, bool):
            raise TypeError("execution_permitted must be a boolean")
        if not isinstance(self.require_target_hash_approval, bool):
            raise TypeError("require_target_hash_approval must be a boolean")
        for target_hash in self.approved_target_hashes:
            _validate_sha256(target_hash, "approved target hash")
        if self.require_target_hash_approval and not self.approved_target_hashes:
            raise ValueError(
                "approved_target_hashes must not be empty when target hash "
                "approval is required"
            )
        PluginRuntimeLoadPolicy(
            loading_permitted=self.loading_permitted,
            allowed_kinds=self.allowed_kinds,
            require_package_target=self.require_package_target,
        )

    def to_load_policy(self) -> PluginRuntimeLoadPolicy:
        """Return the corresponding load policy for target resolution."""
        return PluginRuntimeLoadPolicy(
            loading_permitted=self.loading_permitted,
            allowed_kinds=self.allowed_kinds,
            require_package_target=self.require_package_target,
        )


@dataclass(frozen=True)
class ExecutedPluginCapability:
    """Result of an explicitly approved plugin runtime invocation."""

    loaded: LoadedPluginCapability
    result: object
    audit_record: dict[str, object]


@dataclass(frozen=True)
class PluginExecutionPlan:
    """Prepared plugin-capability invocation plan without executing it."""

    manifest: PluginManifest
    capability: PluginCapability
    argument_count: int
    keyword_names: tuple[str, ...]
    target_hash: str
    plan_hash: str
    audit_record: dict[str, object]


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
        """Construct an explicit runtime execution policy from request fields."""
        return PluginRuntimeExecutionPolicy(
            loading_permitted=self.loading_permitted,
            execution_permitted=self.execution_permitted,
            allowed_kinds=self.allowed_kinds,
            require_package_target=self.require_package_target,
            require_target_hash_approval=self.require_target_hash_approval,
            approved_target_hashes=self.approved_target_hashes,
        )


@dataclass(frozen=True)
class PluginExecutionRequestStorageManifest:
    """Deployment-owned storage manifest for an approved execution request."""

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
    storage_uri: str
    storage_backend: str
    retention_policy: str
    created_by: str
    revoked_request_hashes: tuple[str, ...]
    revocation_hash: str
    manifest_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_storage_manifest_v1":
            raise ValueError(
                "storage manifest schema must be "
                "scpn_plugin_execution_request_storage_manifest_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("storage manifest version must be 1.0.0")
        _validate_sha256(self.request_hash, "storage manifest request hash")
        _validate_sha256(self.plan_hash, "storage manifest plan hash")
        _validate_sha256(self.approval_hash, "storage manifest approval hash")
        _validate_sha256(self.target_hash, "storage manifest target hash")
        _validate_sha256(self.revocation_hash, "storage manifest revocation hash")
        _validate_sha256(self.manifest_hash, "storage manifest hash")
        _require_identifier(self.plugin, "plugin")
        _require_identifier(self.name, "capability name")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported plugin capability kind: {self.kind}")
        _require_non_empty(self.storage_uri, "storage URI")
        _require_identifier(self.storage_backend, "storage backend")
        _require_identifier(self.retention_policy, "retention policy")
        _require_identifier(self.created_by, "storage manifest creator")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")
        for revoked_hash in self.revoked_request_hashes:
            _validate_sha256(revoked_hash, "revoked request hash")


@dataclass(frozen=True)
class PluginExecutionRequestStorageAdapterManifest:
    """Deterministic handoff record for deployment-owned request storage."""

    schema: str
    version: str
    request_hash: str
    storage_manifest_hash: str
    storage_backend: str
    storage_uri: str
    storage_scheme: str
    adapter_mode: str
    bundle_hash: str
    write_performed: bool
    created_by: str
    adapter_hash: str
    audit_record: dict[str, object]

    def __post_init__(self) -> None:
        if self.schema != "scpn_plugin_execution_request_storage_adapter_v1":
            raise ValueError(
                "storage adapter schema must be "
                "scpn_plugin_execution_request_storage_adapter_v1"
            )
        if self.version != "1.0.0":
            raise ValueError("storage adapter version must be 1.0.0")
        _validate_sha256(self.request_hash, "storage adapter request hash")
        _validate_sha256(
            self.storage_manifest_hash,
            "storage adapter manifest hash",
        )
        _validate_sha256(self.bundle_hash, "storage adapter bundle hash")
        _validate_sha256(self.adapter_hash, "storage adapter hash")
        _require_identifier(self.storage_backend, "storage backend")
        _require_non_empty(self.storage_uri, "storage URI")
        _require_identifier(self.created_by, "storage adapter creator")
        if not isinstance(self.write_performed, bool):
            raise TypeError("write_performed must be a boolean")
        if self.audit_record is None:
            raise ValueError("audit_record must be provided")


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
        """Return the revoked request hash set for validation calls."""
        return self.request_hashes


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


def validate_plugin_manifest(manifest: PluginManifest) -> PluginManifest:
    """Validate and return a plugin manifest.

    Dataclass construction performs structural validation; this function
    exists as a stable public compatibility gate for tooling.
    """
    compatibility = compatibility_report(manifest)
    if not compatibility.compatible:
        raise ValueError("; ".join(compatibility.reasons))
    return manifest


def validate_plugin_execution_request(
    request: PluginExecutionRequest,
    *,
    revoked_request_hashes: tuple[str, ...] = (),
) -> PluginExecutionRequest:
    """Validate a stored plugin execution request before runtime consumption."""
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


def validate_plugin_execution_request_lifecycle_record(
    lifecycle: PluginExecutionRequestLifecycleRecord,
) -> PluginExecutionRequestLifecycleRecord:
    """Validate a stored plugin execution request lifecycle record."""
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
    """Validate a stored lifecycle batch summary."""
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


def build_plugin_execution_request_storage_manifest(
    request: PluginExecutionRequest,
    *,
    storage_uri: str,
    storage_backend: str,
    retention_policy: str,
    created_by: str,
    revoked_request_hashes: tuple[str, ...] = (),
) -> PluginExecutionRequestStorageManifest:
    """Build a deterministic storage manifest for an execution request."""
    validate_plugin_execution_request(
        request,
        revoked_request_hashes=revoked_request_hashes,
    )
    _require_non_empty(storage_uri, "storage URI")
    _require_identifier(storage_backend, "storage backend")
    storage_scheme = _validate_storage_backend_uri(storage_backend, storage_uri)
    _require_identifier(retention_policy, "retention policy")
    _require_identifier(created_by, "storage manifest creator")
    normalised_revocations = tuple(
        sorted(revoked_hash.lower() for revoked_hash in revoked_request_hashes)
    )
    for revoked_hash in normalised_revocations:
        _validate_sha256(revoked_hash, "revoked request hash")
    revocation_hash = _request_revocation_hash(normalised_revocations)
    request_hash = str(request.audit_record["request_hash"])
    audit_record: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_storage_manifest_v1",
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
        "storage_uri": storage_uri,
        "storage_backend": storage_backend,
        "storage_scheme": storage_scheme,
        "retention_policy": retention_policy,
        "created_by": created_by,
        "revoked_request_hashes": list(normalised_revocations),
        "revocation_hash": revocation_hash,
    }
    audit_record["manifest_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestStorageManifest(
        schema="scpn_plugin_execution_request_storage_manifest_v1",
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
        storage_uri=storage_uri,
        storage_backend=storage_backend,
        retention_policy=retention_policy,
        created_by=created_by,
        revoked_request_hashes=normalised_revocations,
        revocation_hash=revocation_hash,
        manifest_hash=str(audit_record["manifest_hash"]),
        audit_record=audit_record,
    )


def build_plugin_execution_request_revocation(
    request: PluginExecutionRequest,
    *,
    revoked_by: str,
    revocation_reference: str,
    revocation_reason: str,
) -> PluginExecutionRequestRevocation:
    """Build a deterministic revocation artefact for an execution request."""
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
    """Build a deterministic deployment revocation-list artefact."""
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
    """Validate a stored aggregate request-revocation list."""
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


def build_plugin_execution_request_lifecycle_record(
    request: PluginExecutionRequest,
    *,
    created_by: str,
    storage_manifest: PluginExecutionRequestStorageManifest | None = None,
    revocation_list: PluginExecutionRequestRevocationList | None = None,
) -> PluginExecutionRequestLifecycleRecord:
    """Build a deterministic operator lifecycle status record."""
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
        if not isinstance(revocations, (list, tuple)):
            raise ValueError("revocation list revocations must be a sequence")
        for revocation in revocations:
            if not isinstance(revocation, dict):
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
    """Build a deterministic batch summary for operator lifecycle review."""
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
    """Build a deterministic operator policy report for lifecycle batches."""
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


def validate_plugin_execution_request_storage_manifest(
    request: PluginExecutionRequest,
    manifest: PluginExecutionRequestStorageManifest,
) -> PluginExecutionRequestStorageManifest:
    """Validate that a storage manifest still matches its request envelope."""
    request_hash = str(request.audit_record.get("request_hash", ""))
    if manifest.request_hash != request_hash:
        raise ValueError("storage manifest request hash mismatch")
    if manifest.plan_hash != request.plan_hash:
        raise ValueError("storage manifest plan hash mismatch")
    if manifest.approval_hash != request.approval_hash:
        raise ValueError("storage manifest approval hash mismatch")
    if manifest.target_hash != request.target_hash:
        raise ValueError("storage manifest target hash mismatch")
    _validate_storage_backend_uri(manifest.storage_backend, manifest.storage_uri)
    validate_plugin_execution_request(
        request,
        revoked_request_hashes=manifest.revoked_request_hashes,
    )
    expected = build_plugin_execution_request_storage_manifest(
        request,
        storage_uri=manifest.storage_uri,
        storage_backend=manifest.storage_backend,
        retention_policy=manifest.retention_policy,
        created_by=manifest.created_by,
        revoked_request_hashes=manifest.revoked_request_hashes,
    )
    if manifest.audit_record != expected.audit_record:
        raise ValueError("storage manifest audit record mismatch")
    return manifest


def build_plugin_execution_request_storage_bundle(
    request: PluginExecutionRequest,
    manifest: PluginExecutionRequestStorageManifest,
) -> dict[str, object]:
    """Build a deterministic local persistence bundle for a request envelope."""
    validate_plugin_execution_request_storage_manifest(request, manifest)
    bundle: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_storage_bundle_v1",
        "version": "1.0.0",
        "request": dict(request.audit_record),
        "storage_manifest": dict(manifest.audit_record),
    }
    bundle["bundle_hash"] = _record_hash(bundle)
    return bundle


def build_plugin_execution_request_storage_adapter_manifest(
    request: PluginExecutionRequest,
    manifest: PluginExecutionRequestStorageManifest,
    *,
    write_performed: bool = False,
) -> PluginExecutionRequestStorageAdapterManifest:
    """Build a deterministic handoff manifest for local or external stores."""
    validate_plugin_execution_request_storage_manifest(request, manifest)
    bundle = build_plugin_execution_request_storage_bundle(request, manifest)
    storage_scheme = _validate_storage_backend_uri(
        manifest.storage_backend,
        manifest.storage_uri,
    )
    adapter_mode = (
        "local_file_atomic_write"
        if manifest.storage_backend == "local_file"
        else "deployment_owned_external_write"
    )
    if manifest.storage_backend in _NON_LOCAL_STORAGE_BACKENDS and write_performed:
        raise PermissionError("non-local storage adapters must not write implicitly")
    audit_record: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_storage_adapter_v1",
        "version": "1.0.0",
        "request_hash": manifest.request_hash,
        "storage_manifest_hash": manifest.manifest_hash,
        "storage_backend": manifest.storage_backend,
        "storage_uri": manifest.storage_uri,
        "storage_scheme": storage_scheme,
        "adapter_mode": adapter_mode,
        "bundle_hash": str(bundle["bundle_hash"]),
        "write_performed": write_performed,
        "created_by": manifest.created_by,
    }
    audit_record["adapter_hash"] = _record_hash(audit_record)
    return PluginExecutionRequestStorageAdapterManifest(
        schema="scpn_plugin_execution_request_storage_adapter_v1",
        version="1.0.0",
        request_hash=manifest.request_hash,
        storage_manifest_hash=manifest.manifest_hash,
        storage_backend=manifest.storage_backend,
        storage_uri=manifest.storage_uri,
        storage_scheme=storage_scheme,
        adapter_mode=adapter_mode,
        bundle_hash=str(bundle["bundle_hash"]),
        write_performed=write_performed,
        created_by=manifest.created_by,
        adapter_hash=str(audit_record["adapter_hash"]),
        audit_record=audit_record,
    )


def validate_plugin_execution_request_storage_adapter_manifest(
    adapter: PluginExecutionRequestStorageAdapterManifest,
) -> PluginExecutionRequestStorageAdapterManifest:
    """Validate a stored plugin request storage-adapter handoff manifest."""
    record = adapter.audit_record
    if record.get("schema") != "scpn_plugin_execution_request_storage_adapter_v1":
        raise ValueError("storage adapter schema mismatch")
    if record.get("version") != "1.0.0":
        raise ValueError("storage adapter version must be 1.0.0")
    expected_hash = record.get("adapter_hash")
    if not isinstance(expected_hash, str):
        raise ValueError("storage adapter is missing adapter_hash")
    _validate_sha256(expected_hash, "storage adapter hash")
    payload = dict(record)
    payload.pop("adapter_hash", None)
    if _record_hash(payload) != expected_hash:
        raise ValueError("storage adapter hash mismatch")
    for field_name, expected in (
        ("request_hash", adapter.request_hash),
        ("storage_manifest_hash", adapter.storage_manifest_hash),
        ("storage_backend", adapter.storage_backend),
        ("storage_uri", adapter.storage_uri),
        ("storage_scheme", adapter.storage_scheme),
        ("adapter_mode", adapter.adapter_mode),
        ("bundle_hash", adapter.bundle_hash),
        ("write_performed", adapter.write_performed),
        ("created_by", adapter.created_by),
    ):
        if record.get(field_name) != expected:
            raise ValueError(f"storage adapter {field_name} field mismatch")
    return adapter


def validate_plugin_execution_request_storage_bundle(
    bundle: dict[str, object],
) -> dict[str, object]:
    """Validate an on-disk plugin execution request storage bundle."""
    if bundle.get("schema") != "scpn_plugin_execution_request_storage_bundle_v1":
        raise ValueError(
            "storage bundle schema must be "
            "scpn_plugin_execution_request_storage_bundle_v1"
        )
    if bundle.get("version") != "1.0.0":
        raise ValueError("storage bundle version must be 1.0.0")
    bundle_hash = bundle.get("bundle_hash")
    if not isinstance(bundle_hash, str):
        raise ValueError("storage bundle is missing bundle_hash")
    _validate_sha256(bundle_hash, "storage bundle hash")
    expected_bundle = dict(bundle)
    expected_bundle.pop("bundle_hash", None)
    if _record_hash(expected_bundle) != bundle_hash:
        raise ValueError("storage bundle hash mismatch")

    request_record = bundle.get("request")
    manifest_record = bundle.get("storage_manifest")
    if not isinstance(request_record, dict):
        raise ValueError("storage bundle request must be an object")
    if not isinstance(manifest_record, dict):
        raise ValueError("storage bundle manifest must be an object")
    _validate_request_audit_record(request_record)
    _validate_storage_manifest_audit_record(manifest_record)
    for field in ("request_hash", "plan_hash", "approval_hash", "target_hash"):
        if request_record.get(field) != manifest_record.get(field):
            raise ValueError(f"storage bundle {field} mismatch")
    return bundle


def write_plugin_execution_request_storage_bundle(
    request: PluginExecutionRequest,
    manifest: PluginExecutionRequestStorageManifest,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Atomically persist a validated local-file execution request bundle."""
    if manifest.storage_backend != "local_file":
        raise ValueError("only local_file request storage bundles can be written")
    bundle = build_plugin_execution_request_storage_bundle(request, manifest)
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"request storage bundle already exists: {output_path}")
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(str(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    temporary_path = output_path.with_name(
        f".{output_path.name}.{bundle['bundle_hash']}.tmp"
    )
    temporary_path.write_text(payload, encoding="utf-8")
    temporary_path.replace(output_path)
    return bundle


def _request_revocation_hash(
    revoked_request_hashes: tuple[str, ...],
) -> str:
    normalised_revocations = tuple(
        sorted(revoked_hash.lower() for revoked_hash in revoked_request_hashes)
    )
    for revoked_hash in normalised_revocations:
        _validate_sha256(revoked_hash, "revoked request hash")
    return _record_hash({"revoked_request_hashes": list(normalised_revocations)})


def _validate_request_audit_record(record: dict[str, object]) -> None:
    if record.get("schema") != "scpn_plugin_runtime_execution_request_v1":
        raise ValueError("storage bundle request schema mismatch")
    request_hash = record.get("request_hash")
    if not isinstance(request_hash, str):
        raise ValueError("storage bundle request is missing request_hash")
    _validate_sha256(request_hash, "storage bundle request hash")
    expected = dict(record)
    expected.pop("request_hash", None)
    if _record_hash(expected) != request_hash:
        raise ValueError("storage bundle request hash mismatch")


def _validate_storage_manifest_audit_record(record: dict[str, object]) -> None:
    if record.get("schema") != "scpn_plugin_execution_request_storage_manifest_v1":
        raise ValueError("storage bundle manifest schema mismatch")
    storage_backend = record.get("storage_backend")
    storage_uri = record.get("storage_uri")
    if not isinstance(storage_backend, str):
        raise ValueError("storage bundle manifest storage_backend must be a string")
    if not isinstance(storage_uri, str):
        raise ValueError("storage bundle manifest storage_uri must be a string")
    _validate_storage_backend_uri(storage_backend, storage_uri)
    manifest_hash = record.get("manifest_hash")
    if not isinstance(manifest_hash, str):
        raise ValueError("storage bundle manifest is missing manifest_hash")
    _validate_sha256(manifest_hash, "storage bundle manifest hash")
    expected = dict(record)
    expected.pop("manifest_hash", None)
    if _record_hash(expected) != manifest_hash:
        raise ValueError("storage bundle manifest hash mismatch")
    revoked_hashes = record.get("revoked_request_hashes")
    if not isinstance(revoked_hashes, list):
        raise ValueError("storage bundle revoked_request_hashes must be a list")
    if not all(isinstance(item, str) for item in revoked_hashes):
        raise ValueError("storage bundle revoked request hashes must be strings")
    if _request_revocation_hash(tuple(revoked_hashes)) != record.get("revocation_hash"):
        raise ValueError("storage bundle revocation hash mismatch")


def _validate_storage_backend_uri(storage_backend: str, storage_uri: str) -> str:
    if storage_backend not in _STORAGE_BACKEND_SCHEMES:
        allowed = ", ".join(sorted(_STORAGE_BACKEND_SCHEMES))
        raise ValueError(
            f"unsupported storage backend {storage_backend!r}; use {allowed}"
        )
    parsed = urlparse(storage_uri)
    scheme = parsed.scheme.lower()
    allowed_schemes = _STORAGE_BACKEND_SCHEMES[storage_backend]
    if scheme not in allowed_schemes:
        expected = " or ".join(repr(item or "path") for item in allowed_schemes)
        raise ValueError(
            f"storage backend {storage_backend!r} requires URI scheme {expected}"
        )
    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise ValueError("storage URI must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("storage URI must not contain query or fragment components")
    if storage_backend == "local_file":
        if scheme == "file" and not parsed.path:
            raise ValueError("local_file storage URI requires a file path")
        if scheme == "" and not storage_uri:
            raise ValueError("local_file storage URI requires a file path")
        return scheme or "path"
    if not parsed.netloc:
        raise ValueError(f"storage backend {storage_backend!r} requires an authority")
    if not parsed.path or parsed.path == "/":
        raise ValueError(f"storage backend {storage_backend!r} requires an object path")
    return scheme


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


def load_plugin_capability(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
    *,
    policy: PluginRuntimeLoadPolicy | None = None,
) -> LoadedPluginCapability:
    """Resolve a declared plugin capability under an explicit runtime policy.

    Runtime loading is Python-owned and disabled by default. Callers must pass a
    policy with ``loading_permitted=True`` after their deployment boundary has
    approved the manifest. The target must be declared by the manifest, match the
    requested capability kind/name, and resolve inside the plugin package unless
    the policy explicitly relaxes that check.
    """
    validate_plugin_manifest(manifest)
    if policy is None:
        policy = _DEFAULT_RUNTIME_LOAD_POLICY
    _require_identifier(name, "capability name")
    if kind not in _VALID_KINDS:
        raise ValueError(f"unsupported plugin capability kind: {kind}")
    if not policy.loading_permitted:
        raise PermissionError("plugin runtime loading is disabled by policy")
    if kind not in policy.allowed_kinds:
        raise ValueError(f"{kind} capability is not permitted by runtime load policy")

    capability = _select_capability(manifest, kind, name)
    module_name, attribute_path = _parse_target(capability.target)
    if policy.require_package_target and not _target_within_package(
        module_name,
        manifest.package,
    ):
        raise ValueError(
            f"capability target {capability.target!r} is outside plugin package "
            f"{manifest.package!r}"
        )

    module = importlib.import_module(module_name)
    target_object = _resolve_attribute_path(module, attribute_path, capability.target)
    if not callable(target_object):
        raise TypeError(f"capability target {capability.target!r} must be callable")

    audit_record = _runtime_load_audit_record(
        manifest=manifest,
        capability=capability,
        policy=policy,
        module_name=module_name,
        callable_target=True,
    )
    return LoadedPluginCapability(
        manifest=manifest,
        capability=capability,
        target_object=target_object,
        audit_record=audit_record,
    )


def build_plugin_execution_plan(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    policy: PluginRuntimeExecutionPolicy | None = None,
) -> PluginExecutionPlan:
    """Build a deterministic runtime invocation plan without executing it."""
    if policy is None:
        policy = PluginRuntimeExecutionPolicy()
    if not policy.execution_permitted:
        raise PermissionError("plugin runtime execution is disabled by policy")
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")
    for key in kwargs:
        _require_identifier(key, "plugin execution keyword")

    validate_plugin_manifest(manifest)
    _require_identifier(name, "capability name")
    if kind not in _VALID_KINDS:
        raise ValueError(f"unsupported plugin capability kind: {kind}")

    capability = _select_capability(manifest, kind, name)
    load_policy = policy.to_load_policy()
    if kind not in load_policy.allowed_kinds:
        raise ValueError(f"{kind} capability is not permitted by runtime load policy")

    module_name, _attribute_path = _parse_target(capability.target)
    if load_policy.require_package_target and not _target_within_package(
        module_name,
        manifest.package,
    ):
        raise ValueError(
            f"capability target {capability.target!r} is outside plugin package "
            f"{manifest.package!r}"
        )

    target_hash = _preimport_target_hash(
        manifest=manifest,
        capability=capability,
        policy=policy,
    )
    _assert_execution_target_hash_approved(
        manifest=manifest,
        capability=capability,
        policy=policy,
    )

    keyword_names = tuple(sorted(kwargs))
    audit_record = _runtime_execution_plan_audit_record(
        manifest=manifest,
        capability=capability,
        policy=policy,
        target_hash=target_hash,
        argument_count=len(args),
        keyword_names=keyword_names,
    )
    return PluginExecutionPlan(
        manifest=manifest,
        capability=capability,
        argument_count=len(args),
        keyword_names=keyword_names,
        target_hash=target_hash,
        plan_hash=str(audit_record["plan_hash"]),
        audit_record=audit_record,
    )


def build_plugin_execution_approval(
    plan: PluginExecutionPlan,
    *,
    operator_identity: str,
    approval_reference: str,
    approval_reason: str,
) -> PluginExecutionApproval:
    """Build a deterministic operator approval artefact for an execution plan."""
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
    """Build a deterministic, non-importing request artefact for execution."""
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


def execute_plugin_capability(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    policy: PluginRuntimeExecutionPolicy | None = None,
) -> ExecutedPluginCapability:
    """Invoke a declared plugin capability under an explicit execution policy.

    Execution is denied before any target import unless both loading and
    execution are explicitly permitted. Audit metadata records the invocation
    shape without serialising argument values, so secrets and large payloads are
    not copied into the audit record.
    """
    if kwargs is None:
        kwargs = {}
    if policy is None:
        policy = PluginRuntimeExecutionPolicy()
    plan = build_plugin_execution_plan(
        manifest,
        kind,
        name,
        args=args,
        kwargs=kwargs,
        policy=policy,
    )
    loaded = load_plugin_capability(
        manifest,
        kind,
        name,
        policy=policy.to_load_policy(),
    )
    target = loaded.target_object
    if not callable(target):
        raise TypeError("loaded plugin target must be callable")
    result = target(*args, **kwargs)
    audit_record = _runtime_execute_audit_record(
        loaded=loaded,
        policy=policy,
        args=args,
        kwargs=kwargs,
        result=result,
        plan_hash=plan.plan_hash,
    )
    return ExecutedPluginCapability(
        loaded=loaded,
        result=result,
        audit_record=audit_record,
    )


def execute_plugin_execution_request(
    manifest: PluginManifest,
    request: PluginExecutionRequest,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
    revoked_request_hashes: tuple[str, ...] = (),
) -> ExecutedPluginCapability:
    """Invoke a plugin only when the approved request matches this call shape.

    The request-bound path validates manifest identity, invocation shape, plan
    hash, and target hash before importing the plugin module. Argument values
    remain outside audit records; only positional count and keyword names
    participate in the plan hash.
    """
    if kwargs is None:
        kwargs = {}
    if not isinstance(args, tuple):
        raise TypeError("args must be a tuple")
    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary")
    if request.schema != "scpn_plugin_runtime_execution_request_v1":
        raise ValueError(
            "request schema must be scpn_plugin_runtime_execution_request_v1"
        )
    validate_plugin_execution_request(
        request,
        revoked_request_hashes=revoked_request_hashes,
    )
    if manifest.name != request.plugin:
        raise ValueError("request plugin does not match manifest")

    policy = request.to_execution_policy()
    plan = build_plugin_execution_plan(
        manifest,
        request.kind,
        request.name,
        args=args,
        kwargs=kwargs,
        policy=policy,
    )
    if plan.plan_hash != request.plan_hash:
        raise PermissionError("execution request plan hash mismatch")
    if plan.target_hash != request.target_hash:
        raise PermissionError("execution request target hash mismatch")

    executed = execute_plugin_capability(
        manifest,
        request.kind,
        request.name,
        args=args,
        kwargs=kwargs,
        policy=policy,
    )
    audit_record = {
        **executed.audit_record,
        "request_hash": request.audit_record["request_hash"],
        "approval_hash": request.approval_hash,
        "operator_identity": request.operator_identity,
        "approval_reference": request.approval_reference,
    }
    audit_record["execution_hash"] = _record_hash(audit_record)
    return ExecutedPluginCapability(
        loaded=executed.loaded,
        result=executed.result,
        audit_record=audit_record,
    )


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


def _select_capability(
    manifest: PluginManifest,
    kind: PluginKind,
    name: str,
) -> PluginCapability:
    matches = tuple(
        capability
        for capability in manifest.capabilities
        if capability.kind == kind and capability.name == name
    )
    if not matches:
        raise LookupError(f"plugin capability {kind}:{name} is not declared")
    if len(matches) > 1:
        raise ValueError(f"plugin capability {kind}:{name} is declared more than once")
    return matches[0]


def _parse_target(target: str) -> tuple[str, str]:
    if ":" not in target:
        raise ValueError("capability target must use 'module:attribute' syntax")
    module_name, attribute_path = target.split(":", maxsplit=1)
    _require_non_empty(module_name, "capability target module")
    _require_non_empty(attribute_path, "capability target attribute")
    return module_name, attribute_path


def _target_within_package(module_name: str, package: str) -> bool:
    return module_name == package or module_name.startswith(f"{package}.")


def _resolve_attribute_path(
    module: object,
    attribute_path: str,
    target: str,
) -> object:
    current = module
    for attribute in attribute_path.split("."):
        _require_identifier(attribute, "capability target attribute")
        try:
            current = getattr(current, attribute)
        except AttributeError as exc:
            raise AttributeError(
                f"capability target {target!r} is not importable"
            ) from exc
    return current


def _runtime_load_audit_record(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeLoadPolicy,
    module_name: str,
    callable_target: bool,
) -> dict[str, object]:
    record = {
        "schema": "scpn_plugin_runtime_load_v1",
        "spo_version": __version__,
        "plugin": manifest.name,
        "plugin_version": manifest.version,
        "package": manifest.package,
        "kind": capability.kind,
        "name": capability.name,
        "target": capability.target,
        "module": module_name,
        "version": capability.version,
        "channels": list(capability.channels),
        "knobs": list(capability.knobs),
        "loading_permitted": policy.loading_permitted,
        "load_policy": "python_owned_explicit",
        "require_package_target": policy.require_package_target,
        "allowed_kinds": list(policy.allowed_kinds),
        "callable": callable_target,
    }
    record["target_hash"] = _record_hash(record)
    record["load_hash"] = _record_hash(record)
    return record


def _runtime_execute_audit_record(
    *,
    loaded: LoadedPluginCapability,
    policy: PluginRuntimeExecutionPolicy,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    result: object,
    plan_hash: str | None = None,
) -> dict[str, object]:
    record = {
        "schema": "scpn_plugin_runtime_execute_v1",
        "load_hash": loaded.audit_record["load_hash"],
        "target_hash": loaded.audit_record["target_hash"],
        "plugin": loaded.manifest.name,
        "plugin_version": loaded.manifest.version,
        "package": loaded.manifest.package,
        "kind": loaded.capability.kind,
        "name": loaded.capability.name,
        "target": loaded.capability.target,
        "loading_permitted": policy.loading_permitted,
        "execution_permitted": policy.execution_permitted,
        "target_hash_approved": _execution_target_hash_approved(
            str(loaded.audit_record["target_hash"]),
            policy,
        ),
        "approved_target_hashes": list(policy.approved_target_hashes),
        "load_policy": "python_owned_explicit",
        "argument_count": len(args),
        "keyword_names": sorted(kwargs),
        "result_type": type(result).__name__,
    }
    if plan_hash is not None:
        record["plan_hash"] = plan_hash
    record["execution_hash"] = _record_hash(record)
    return record


def _runtime_execution_plan_audit_record(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeExecutionPolicy,
    target_hash: str,
    argument_count: int,
    keyword_names: tuple[str, ...],
) -> dict[str, object]:
    record = {
        "schema": "scpn_plugin_runtime_execution_plan_v1",
        "schema_version": "1.0.0",
        "spo_version": __version__,
        "plugin": manifest.name,
        "plugin_version": manifest.version,
        "package": manifest.package,
        "kind": capability.kind,
        "name": capability.name,
        "target": capability.target,
        "target_hash": target_hash,
        "load_hash": target_hash,
        "argument_count": argument_count,
        "keyword_names": list(keyword_names),
        "execution_permitted": policy.execution_permitted,
        "loading_permitted": policy.loading_permitted,
        "require_package_target": policy.require_package_target,
        "allowed_kinds": list(policy.allowed_kinds),
        "require_target_hash_approval": policy.require_target_hash_approval,
        "approved_target_hashes": list(policy.approved_target_hashes),
        "target_hash_approved": _execution_target_hash_approved(
            target_hash=target_hash,
            policy=policy,
        ),
    }
    record["plan_hash"] = _record_hash(record)
    return record


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


def _assert_execution_target_hash_approved(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeExecutionPolicy,
) -> None:
    if not policy.require_target_hash_approval:
        return
    expected_hash = _preimport_target_hash(
        manifest=manifest,
        capability=capability,
        policy=policy,
    )
    if expected_hash not in policy.approved_target_hashes:
        raise PermissionError(
            f"plugin runtime target hash {expected_hash} is not approved"
        )


def _preimport_target_hash(
    *,
    manifest: PluginManifest,
    capability: PluginCapability,
    policy: PluginRuntimeExecutionPolicy,
) -> str:
    module_name, _attribute_path = _parse_target(capability.target)
    load_policy = policy.to_load_policy()
    record = _runtime_load_audit_record(
        manifest=manifest,
        capability=capability,
        policy=load_policy,
        module_name=module_name,
        callable_target=True,
    )
    return str(record["target_hash"])


def _execution_target_hash_approved(
    target_hash: str,
    policy: PluginRuntimeExecutionPolicy,
) -> bool:
    if not policy.require_target_hash_approval:
        return False
    return target_hash in policy.approved_target_hashes


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


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be a SHA-256 hex digest") from exc


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


_DEFAULT_RUNTIME_LOAD_POLICY = PluginRuntimeLoadPolicy()
