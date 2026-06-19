# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI approval/request/revocation payload guards

from __future__ import annotations

from typing import Any

import click
import pytest

from scpn_phase_orchestrator.runtime.cli import (
    _load_approval_from_payload,
    _load_request_from_payload,
    _load_revocation_from_payload,
    _load_revocation_list_from_payload,
)

_HEX = "a" * 64


def _approval(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_approval_v1",
        "plan_hash": _HEX,
        "target_hash": _HEX,
        "approval_hash": _HEX,
        "plugin": "grid_pack",
        "kind": "monitor",
        "name": "frequency_drift",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "approval_reason": "approved for review",
        "approved": True,
        "execution_permitted": True,
        "version": "1.0.0",
    }
    payload.update(overrides)
    return payload


def _request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_runtime_execution_request_v1",
        "plan_hash": _HEX,
        "target_hash": _HEX,
        "approval_hash": _HEX,
        "plugin": "grid_pack",
        "kind": "monitor",
        "name": "frequency_drift",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "version": "1.0.0",
        "loading_permitted": True,
        "execution_permitted": True,
        "require_target_hash_approval": False,
        "require_package_target": True,
        "approved_target_hashes": [],
        "allowed_kinds": ["monitor"],
    }
    payload.update(overrides)
    return payload


def _revocation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_revocation_v1",
        "request_hash": _HEX,
        "plan_hash": _HEX,
        "approval_hash": _HEX,
        "target_hash": _HEX,
        "revocation_hash": _HEX,
        "plugin": "grid_pack",
        "kind": "monitor",
        "name": "frequency_drift",
        "operator_identity": "operator",
        "approval_reference": "REF-1",
        "revoked_by": "operator",
        "revocation_reference": "RREF-1",
        "revocation_reason": "superseded",
        "revoked": True,
        "version": "1.0.0",
    }
    payload.update(overrides)
    return payload


class TestApprovalLoader:
    def test_valid_round_trips(self) -> None:
        assert _load_approval_from_payload(_approval()).approved is True

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="approval schema mismatch"):
            _load_approval_from_payload(_approval(schema="x"))

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(click.ClickException, match="operator_identity must be"):
            _load_approval_from_payload(_approval(operator_identity=""))

    def test_rejects_non_boolean_approved(self) -> None:
        with pytest.raises(click.ClickException, match="approved must be a boolean"):
            _load_approval_from_payload(_approval(approved="yes"))

    def test_rejects_non_boolean_execution_permitted(self) -> None:
        with pytest.raises(click.ClickException, match="execution_permitted must be"):
            _load_approval_from_payload(_approval(execution_permitted="yes"))

    def test_rejects_unsupported_kind(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported kind"):
            _load_approval_from_payload(_approval(kind="wizard"))


class TestRequestLoader:
    def test_valid_round_trips(self) -> None:
        assert _load_request_from_payload(_request()).execution_permitted is True

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="request schema mismatch"):
            _load_request_from_payload(_request(schema="x"))

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(click.ClickException, match="plugin must be"):
            _load_request_from_payload(_request(plugin=""))

    def test_rejects_unsupported_kind(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported kind"):
            _load_request_from_payload(_request(kind="wizard"))

    def test_rejects_non_boolean_flag(self) -> None:
        with pytest.raises(click.ClickException, match="loading_permitted must be"):
            _load_request_from_payload(_request(loading_permitted="yes"))

    def test_rejects_non_string_approved_target_hashes(self) -> None:
        with pytest.raises(click.ClickException, match="approved_target_hashes must"):
            _load_request_from_payload(_request(approved_target_hashes=[1]))

    def test_rejects_invalid_allowed_kinds(self) -> None:
        with pytest.raises(click.ClickException, match="allowed_kinds must be"):
            _load_request_from_payload(_request(allowed_kinds=["wizard"]))


class TestRevocationLoader:
    def test_valid_round_trips(self) -> None:
        assert _load_revocation_from_payload(_revocation()).revoked is True

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="revocation schema mismatch"):
            _load_revocation_from_payload(_revocation(schema="x"))

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(click.ClickException, match="revoked_by must be"):
            _load_revocation_from_payload(_revocation(revoked_by=""))

    def test_rejects_unsupported_kind(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported kind"):
            _load_revocation_from_payload(_revocation(kind="wizard"))

    def test_rejects_non_revoked(self) -> None:
        with pytest.raises(click.ClickException, match="revoked must be true"):
            _load_revocation_from_payload(_revocation(revoked=False))


def _revocation_list(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_revocation_list_v1",
        "request_hashes": [_HEX],
        "revocation_hashes": [_HEX],
        "revocation_count": 1,
        "created_by": "operator",
        "revocation_list_hash": _HEX,
        "version": "1.0.0",
    }
    payload.update(overrides)
    return payload


class TestRevocationListLoader:
    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(
            click.ClickException, match="revocation list schema mismatch"
        ):
            _load_revocation_list_from_payload(_revocation_list(schema="x"))

    def test_rejects_empty_version(self) -> None:
        with pytest.raises(click.ClickException, match="version must be non-empty"):
            _load_revocation_list_from_payload(_revocation_list(version=""))

    def test_rejects_empty_created_by(self) -> None:
        with pytest.raises(click.ClickException, match="created_by must be non-empty"):
            _load_revocation_list_from_payload(_revocation_list(created_by=""))

    def test_rejects_non_positive_count(self) -> None:
        with pytest.raises(click.ClickException, match="revocation_count must be"):
            _load_revocation_list_from_payload(_revocation_list(revocation_count=0))

    def test_rejects_non_string_request_hashes(self) -> None:
        with pytest.raises(click.ClickException, match="request_hashes must be"):
            _load_revocation_list_from_payload(_revocation_list(request_hashes=[1]))

    def test_rejects_non_string_revocation_hashes(self) -> None:
        with pytest.raises(click.ClickException, match="revocation_hashes must be"):
            _load_revocation_list_from_payload(_revocation_list(revocation_hashes=[1]))
