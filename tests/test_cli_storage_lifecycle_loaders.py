# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI storage/lifecycle payload loader guards

from __future__ import annotations

from typing import Any

import click
import pytest

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_lifecycle_from_payload,
    _load_lifecycle_summary_from_payload,
    _load_storage_manifest_from_payload,
)

_HEX = "a" * 64


def _storage_manifest(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_storage_manifest_v1",
        "request_hash": _HEX,
        "plan_hash": _HEX,
        "approval_hash": _HEX,
        "target_hash": _HEX,
        "revocation_hash": _HEX,
        "manifest_hash": _HEX,
        "plugin": "grid_pack",
        "kind": "monitor",
        "name": "frequency_drift",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "storage_uri": "file:///tmp/store",
        "storage_backend": "filesystem",
        "retention_policy": "30d",
        "created_by": "operator",
        "version": "1.0.0",
        "revoked_request_hashes": [],
    }
    payload.update(overrides)
    return payload


class TestStorageManifestLoader:
    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="storage manifest schema"):
            _load_storage_manifest_from_payload(_storage_manifest(schema="x"))

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(click.ClickException, match="storage_uri must be non-empty"):
            _load_storage_manifest_from_payload(_storage_manifest(storage_uri=""))

    def test_rejects_unsupported_kind(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported kind"):
            _load_storage_manifest_from_payload(_storage_manifest(kind="wizard"))

    def test_rejects_non_string_revoked_request_hashes(self) -> None:
        with pytest.raises(
            click.ClickException, match="revoked_request_hashes must be a string list"
        ):
            _load_storage_manifest_from_payload(
                _storage_manifest(revoked_request_hashes=[1])
            )


def _lifecycle(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_lifecycle_v1",
        "request_hash": _HEX,
        "lifecycle_hash": _HEX,
        "status": "approved",
        "plugin": "grid_pack",
        "kind": "monitor",
        "name": "frequency_drift",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "created_by": "operator",
        "version": "1.0.0",
        "revoked": False,
    }
    payload.update(overrides)
    return payload


class TestLifecycleLoader:
    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="lifecycle schema mismatch"):
            _load_lifecycle_from_payload(_lifecycle(schema="x"))

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(click.ClickException, match="created_by must be non-empty"):
            _load_lifecycle_from_payload(_lifecycle(created_by=""))

    def test_rejects_unsupported_status(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported status"):
            _load_lifecycle_from_payload(_lifecycle(status="pending"))

    def test_rejects_unsupported_kind(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported kind"):
            _load_lifecycle_from_payload(_lifecycle(kind="wizard"))

    def test_rejects_non_boolean_revoked(self) -> None:
        with pytest.raises(click.ClickException, match="revoked must be boolean"):
            _load_lifecycle_from_payload(_lifecycle(revoked="yes"))


def _summary(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_lifecycle_summary_v1",
        "summary_hash": _HEX,
        "request_count": 1,
        "status_counts": {"approved": 1},
        "lifecycle_hashes": [_HEX],
        "approved_request_hashes": [_HEX],
        "stored_request_hashes": [],
        "revoked_request_hashes": [],
        "storage_missing_request_hashes": [],
        "renewal_required_request_hashes": [],
        "created_by": "operator",
        "version": "1.0.0",
    }
    payload.update(overrides)
    return payload


class TestLifecycleSummaryLoader:
    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="lifecycle summary schema"):
            _load_lifecycle_summary_from_payload(_summary(schema="x"))

    def test_rejects_empty_version(self) -> None:
        with pytest.raises(click.ClickException, match="summary version must be"):
            _load_lifecycle_summary_from_payload(_summary(version=""))

    def test_rejects_empty_created_by(self) -> None:
        with pytest.raises(click.ClickException, match="summary created_by must be"):
            _load_lifecycle_summary_from_payload(_summary(created_by=""))

    def test_rejects_non_positive_request_count(self) -> None:
        with pytest.raises(
            click.ClickException, match="request_count must be a positive"
        ):
            _load_lifecycle_summary_from_payload(_summary(request_count=0))

    def test_rejects_malformed_status_counts(self) -> None:
        with pytest.raises(click.ClickException, match="status_counts is malformed"):
            _load_lifecycle_summary_from_payload(
                _summary(status_counts={"approved": "x"})
            )

    def test_rejects_non_string_hash_list(self) -> None:
        with pytest.raises(
            click.ClickException, match="lifecycle_hashes must be a string list"
        ):
            _load_lifecycle_summary_from_payload(_summary(lifecycle_hashes=[1]))
