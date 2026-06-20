# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin execution request storage manifests

"""Storage manifests, adapter manifests, and execution-request bundle persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

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


def build_plugin_execution_request_storage_manifest(
    request: PluginExecutionRequest,
    *,
    storage_uri: str,
    storage_backend: str,
    retention_policy: str,
    created_by: str,
    revoked_request_hashes: tuple[str, ...] = (),
) -> PluginExecutionRequestStorageManifest:
    """Build a deterministic storage manifest for an execution request.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    storage_uri : str
        Storage URI.
    storage_backend : str
        Storage backend identifier.
    retention_policy : str
        Retention policy label.
    created_by : str
        Identifier of the creating actor.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.

    Returns
    -------
    PluginExecutionRequestStorageManifest
        A deterministic storage manifest for an execution request.
    """
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


def validate_plugin_execution_request_storage_manifest(
    request: PluginExecutionRequest,
    manifest: PluginExecutionRequestStorageManifest,
) -> PluginExecutionRequestStorageManifest:
    """Validate that a storage manifest still matches its request envelope.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    manifest : PluginExecutionRequestStorageManifest
        The manifest object.

    Returns
    -------
    PluginExecutionRequestStorageManifest
        That a storage manifest still matches its request envelope.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
    """Build a deterministic local persistence bundle for a request envelope.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    manifest : PluginExecutionRequestStorageManifest
        The manifest object.

    Returns
    -------
    dict[str, object]
        A deterministic local persistence bundle for a request envelope.
    """
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
    """Build a deterministic handoff manifest for local or external stores.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    manifest : PluginExecutionRequestStorageManifest
        The manifest object.
    write_performed : bool
        Whether a write was performed.

    Returns
    -------
    PluginExecutionRequestStorageAdapterManifest
        A deterministic handoff manifest for local or external stores.

    Raises
    ------
    PermissionError
        If the operation is not permitted by policy.
    """
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
    """Validate a stored plugin request storage-adapter handoff manifest.

    Parameters
    ----------
    adapter : PluginExecutionRequestStorageAdapterManifest
        The storage-adapter manifest.

    Returns
    -------
    PluginExecutionRequestStorageAdapterManifest
        A stored plugin request storage-adapter handoff manifest.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
    """Validate an on-disk plugin execution request storage bundle.

    Parameters
    ----------
    bundle : dict[str, object]
        The on-disk storage bundle.

    Returns
    -------
    dict[str, object]
        An on-disk plugin execution request storage bundle.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
    """Atomically persist a validated local-file execution request bundle.

    Parameters
    ----------
    request : PluginExecutionRequest
        The plugin execution request.
    manifest : PluginExecutionRequestStorageManifest
        The manifest object.
    path : str | Path
        Filesystem path to the target file.
    overwrite : bool
        Whether to overwrite an existing artefact.

    Returns
    -------
    dict[str, object]
        A validated local-file execution request bundle.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    FileExistsError
        If the target already exists and overwrite is false.
    IsADirectoryError
        If the target path is a directory.
    """
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
