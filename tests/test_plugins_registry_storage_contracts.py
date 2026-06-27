# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plugin registry storage contract tests

"""Focused storage-manifest and bundle validation contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scpn_phase_orchestrator.plugins.registry import (
    PluginCapability,
    PluginExecutionRequest,
    PluginExecutionRequestStorageAdapterManifest,
    PluginExecutionRequestStorageManifest,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    _record_hash,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request,
    build_plugin_execution_request_storage_adapter_manifest,
    build_plugin_execution_request_storage_bundle,
    build_plugin_execution_request_storage_manifest,
    validate_plugin_execution_request_storage_adapter_manifest,
    validate_plugin_execution_request_storage_bundle,
    validate_plugin_execution_request_storage_manifest,
    write_plugin_execution_request_storage_bundle,
)

_HEX = "a" * 64


def _manifest() -> PluginManifest:
    """Build a minimal compatible plugin manifest."""
    return PluginManifest(
        name="grid_pack",
        version="0.1.0",
        package="grid_pack",
        capabilities=(
            PluginCapability(
                kind="actuator",
                name="breaker",
                target="grid_pack.actuators:BreakerMapper",
                knobs=("K",),
            ),
        ),
        min_spo_version="0.1.0",
    )


def _request() -> PluginExecutionRequest:
    """Build a real approved plugin execution request."""
    draft_plan = build_plugin_execution_plan(
        _manifest(),
        "actuator",
        "breaker",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
        ),
    )
    plan = build_plugin_execution_plan(
        _manifest(),
        "actuator",
        "breaker",
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
            require_target_hash_approval=True,
            approved_target_hashes=(draft_plan.target_hash,),
        ),
    )
    approval = build_plugin_execution_approval(
        plan,
        operator_identity="operator_alpha",
        approval_reference="REQ-2026-06-27-STORAGE",
        approval_reason="operator approved",
    )
    return build_plugin_execution_request(plan, approval)


def _storage_manifest(
    *,
    storage_uri: str = "file:///var/lib/spo/plugin-requests/grid_pack.json",
    storage_backend: str = "local_file",
    revoked_request_hashes: tuple[str, ...] = (),
) -> PluginExecutionRequestStorageManifest:
    """Build a real storage manifest for the request fixture."""
    return build_plugin_execution_request_storage_manifest(
        _request(),
        storage_uri=storage_uri,
        storage_backend=storage_backend,
        retention_policy="retain_until_revoked",
        created_by="deployment_gate",
        revoked_request_hashes=revoked_request_hashes,
    )


def _storage_adapter() -> PluginExecutionRequestStorageAdapterManifest:
    """Build a real storage adapter manifest."""
    request = _request()
    manifest = build_plugin_execution_request_storage_manifest(
        request,
        storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
        storage_backend="local_file",
        retention_policy="retain_until_revoked",
        created_by="deployment_gate",
    )
    return build_plugin_execution_request_storage_adapter_manifest(request, manifest)


def _storage_bundle() -> dict[str, object]:
    """Build a real storage bundle."""
    request = _request()
    manifest = build_plugin_execution_request_storage_manifest(
        request,
        storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
        storage_backend="local_file",
        retention_policy="retain_until_revoked",
        created_by="deployment_gate",
    )
    return build_plugin_execution_request_storage_bundle(request, manifest)


def _rehashed_bundle(bundle: dict[str, object]) -> dict[str, object]:
    """Return ``bundle`` with a recomputed outer bundle hash."""
    core = dict(bundle)
    core.pop("bundle_hash", None)
    return {**core, "bundle_hash": _record_hash(core)}


def _rehashed_manifest_record(record: dict[str, object]) -> dict[str, object]:
    """Return a storage-manifest audit record with recomputed manifest hash."""
    core = dict(record)
    core.pop("manifest_hash", None)
    return {**core, "manifest_hash": _record_hash(core)}


class TestStorageRecordConstructionGuards:
    """Construction-time validation guards for storage dataclasses and builders."""

    def test_storage_manifest_rejects_missing_audit_record(self) -> None:
        """Storage manifests require an audit record."""
        manifest = _storage_manifest()

        with pytest.raises(ValueError, match="audit_record must be provided"):
            replace(manifest, audit_record=None)

    def test_storage_adapter_rejects_missing_audit_record(self) -> None:
        """Storage adapter manifests require an audit record."""
        adapter = _storage_adapter()

        with pytest.raises(ValueError, match="audit_record must be provided"):
            replace(adapter, audit_record=None)

    def test_storage_manifest_normalises_valid_revocation_hashes(self) -> None:
        """Storage manifest construction validates and sorts revocation hashes."""
        manifest = _storage_manifest(
            revoked_request_hashes=("B" * 64, "0" * 64),
        )

        assert manifest.revoked_request_hashes == ("0" * 64, "b" * 64)
        assert len(manifest.revocation_hash) == 64

    @pytest.mark.parametrize(
        ("field_name", "diagnostic"),
        [
            ("request_hash", "request hash mismatch"),
            ("plan_hash", "plan hash mismatch"),
            ("approval_hash", "approval hash mismatch"),
            ("target_hash", "target hash mismatch"),
        ],
    )
    def test_storage_manifest_validation_rejects_envelope_mismatch(
        self,
        field_name: str,
        diagnostic: str,
    ) -> None:
        """Storage manifest validation rejects request-envelope mismatches."""
        request = _request()
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="file:///var/lib/spo/plugin-requests/grid_pack.json",
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )
        tampered = replace(manifest, **{field_name: "0" * 64})

        with pytest.raises(ValueError, match=diagnostic):
            validate_plugin_execution_request_storage_manifest(request, tampered)


class TestStorageAdapterValidationGuards:
    """Stored adapter audit-record validation guards."""

    @pytest.mark.parametrize(
        ("record_update", "diagnostic"),
        [
            ({"schema": "wrong"}, "storage adapter schema mismatch"),
            ({"version": "2.0.0"}, "version must be 1.0.0"),
            ({"adapter_hash": 123}, "missing adapter_hash"),
        ],
    )
    def test_adapter_validation_rejects_malformed_audit_record(
        self,
        record_update: dict[str, object],
        diagnostic: str,
    ) -> None:
        """Adapter validation rejects malformed stored audit records."""
        adapter = _storage_adapter()
        tampered = replace(
            adapter,
            audit_record={**adapter.audit_record, **record_update},
        )

        with pytest.raises(ValueError, match=diagnostic):
            validate_plugin_execution_request_storage_adapter_manifest(tampered)

    def test_adapter_validation_rejects_hash_mismatch(self) -> None:
        """Adapter validation rejects audit records with stale hashes."""
        adapter = _storage_adapter()
        tampered = replace(
            adapter,
            audit_record={**adapter.audit_record, "created_by": "other_gate"},
        )

        with pytest.raises(ValueError, match="storage adapter hash mismatch"):
            validate_plugin_execution_request_storage_adapter_manifest(tampered)

    def test_adapter_validation_rejects_field_mismatch_after_rehash(self) -> None:
        """Adapter validation compares rehashed audit fields to dataclass fields."""
        adapter = _storage_adapter()
        record = {**adapter.audit_record, "created_by": "other_gate"}
        record_without_hash = dict(record)
        record_without_hash.pop("adapter_hash", None)
        record["adapter_hash"] = _record_hash(record_without_hash)
        tampered = replace(adapter, audit_record=record)

        with pytest.raises(ValueError, match="created_by field mismatch"):
            validate_plugin_execution_request_storage_adapter_manifest(tampered)


class TestStorageBundleValidationGuards:
    """On-disk storage-bundle validation guards."""

    @pytest.mark.parametrize(
        ("bundle_update", "diagnostic"),
        [
            ({"schema": "wrong"}, "storage bundle schema must be"),
            ({"bundle_hash": 123}, "missing bundle_hash"),
            ({"storage_manifest": "not-a-dict"}, "manifest must be an object"),
        ],
    )
    def test_bundle_validation_rejects_malformed_outer_record(
        self,
        bundle_update: dict[str, object],
        diagnostic: str,
    ) -> None:
        """Bundle validation rejects malformed outer storage-bundle records."""
        bundle = _storage_bundle()
        tampered = {**bundle, **bundle_update}
        if "bundle_hash" not in bundle_update:
            tampered = _rehashed_bundle(tampered)

        with pytest.raises(ValueError, match=diagnostic):
            validate_plugin_execution_request_storage_bundle(tampered)

    def test_bundle_validation_rejects_request_manifest_hash_mismatch(self) -> None:
        """Bundle validation rejects request and manifest envelope disagreement."""
        bundle = _storage_bundle()
        manifest_record = dict(bundle["storage_manifest"])  # type: ignore[arg-type]
        manifest_record["plan_hash"] = "0" * 64
        manifest_record = _rehashed_manifest_record(manifest_record)
        tampered = _rehashed_bundle({**bundle, "storage_manifest": manifest_record})

        with pytest.raises(ValueError, match="storage bundle plan_hash mismatch"):
            validate_plugin_execution_request_storage_bundle(tampered)

    def test_bundle_validation_rejects_manifest_storage_uri_not_string(self) -> None:
        """Bundle validation rejects non-string manifest storage URIs."""
        bundle = _storage_bundle()
        manifest_record = {**dict(bundle["storage_manifest"]), "storage_uri": 5}
        tampered = _rehashed_bundle({**bundle, "storage_manifest": manifest_record})

        with pytest.raises(ValueError, match="storage_uri must be a string"):
            validate_plugin_execution_request_storage_bundle(tampered)

    def test_bundle_validation_rejects_invalid_revoked_hash_inside_manifest(
        self,
    ) -> None:
        """Bundle validation validates revoked request hashes after manifest hashing."""
        bundle = _storage_bundle()
        manifest_record = {
            **dict(bundle["storage_manifest"]),
            "revoked_request_hashes": ["bad"],
        }
        manifest_record = _rehashed_manifest_record(manifest_record)
        tampered = _rehashed_bundle({**bundle, "storage_manifest": manifest_record})

        with pytest.raises(ValueError, match="revoked request hash"):
            validate_plugin_execution_request_storage_bundle(tampered)


class TestStoragePersistenceAndUriGuards:
    """Persistence and storage URI policy guards."""

    def test_write_bundle_rejects_non_local_manifest(self, tmp_path: Path) -> None:
        """Only local-file storage bundles can be written by the registry."""
        request = _request()
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri="s3://spo-prod/plugin-requests/grid_pack.json",
            storage_backend="s3_object",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )

        with pytest.raises(ValueError, match="only local_file"):
            write_plugin_execution_request_storage_bundle(
                request,
                manifest,
                tmp_path / "request.json",
            )

    def test_write_bundle_rejects_directory_target_when_overwriting(
        self,
        tmp_path: Path,
    ) -> None:
        """Atomic bundle writes reject directory targets."""
        request = _request()
        manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri=(tmp_path / "request.json").as_uri(),
            storage_backend="local_file",
            retention_policy="retain_until_revoked",
            created_by="deployment_gate",
        )

        with pytest.raises(IsADirectoryError):
            write_plugin_execution_request_storage_bundle(
                request,
                manifest,
                tmp_path,
                overwrite=True,
            )

    @pytest.mark.parametrize(
        ("storage_backend", "storage_uri", "diagnostic"),
        [
            ("ftp_object", "ftp://bucket/key", "unsupported storage backend"),
            ("s3_object", "s3://user:pass@bucket/key", "must not contain credentials"),
            ("s3_object", "s3://bucket/key?version=1", "query or fragment"),
            ("local_file", "file://", "requires a file path"),
            ("s3_object", "s3:/key", "requires an authority"),
            ("s3_object", "s3://bucket/", "requires an object path"),
        ],
    )
    def test_storage_manifest_rejects_invalid_backend_uri_policy(
        self,
        storage_backend: str,
        storage_uri: str,
        diagnostic: str,
    ) -> None:
        """Storage manifests validate backend-specific URI policy."""
        with pytest.raises(ValueError, match=diagnostic):
            _storage_manifest(
                storage_uri=storage_uri,
                storage_backend=storage_backend,
            )
