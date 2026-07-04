# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI storage-adapter and remediation loader guards

from __future__ import annotations

from typing import Any

import click
import pytest

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_lifecycle_multistore_drilldown_payload,
    _load_lifecycle_policy_report_payload,
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_plan_payload,
    _load_storage_adapter_from_payload,
)

_HEX = "a" * 64


def _adapter(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_storage_adapter_v1",
        "request_hash": _HEX,
        "storage_manifest_hash": _HEX,
        "bundle_hash": _HEX,
        "adapter_hash": _HEX,
        "storage_backend": "filesystem",
        "storage_uri": "file:///tmp/store",
        "storage_scheme": "file",
        "adapter_mode": "external",
        "write_performed": True,
        "created_by": "operator",
        "version": "1.0.0",
    }
    payload.update(overrides)
    return payload


class TestStorageAdapterLoader:
    def test_valid_round_trips_write_performed_flag(self) -> None:
        payload = _adapter(write_performed=False)

        adapter = _load_storage_adapter_from_payload(payload)

        assert adapter.storage_backend == "filesystem"
        assert adapter.write_performed is False
        assert adapter.audit_record is payload

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="storage adapter schema"):
            _load_storage_adapter_from_payload(_adapter(schema="x"))

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(
            click.ClickException, match="storage_scheme must be non-empty"
        ):
            _load_storage_adapter_from_payload(_adapter(storage_scheme=""))

    def test_rejects_non_boolean_write_performed(self) -> None:
        with pytest.raises(
            click.ClickException, match="write_performed must be boolean"
        ):
            _load_storage_adapter_from_payload(_adapter(write_performed="yes"))


def _policy(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_lifecycle_policy_v1",
        "summary_hash": _HEX,
        "policy_hash": _HEX,
        "request_count": 1,
        "renewal_required_request_hashes": [_HEX],
        "missing_adapter_request_hashes": [],
        "external_write_followup_request_hashes": [],
    }
    payload.update(overrides)
    return payload


class TestPolicyReportLoader:
    def test_valid_round_trips(self) -> None:
        assert _load_lifecycle_policy_report_payload(_policy())["request_count"] == 1

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="lifecycle policy schema"):
            _load_lifecycle_policy_report_payload(_policy(schema="x"))

    def test_rejects_non_positive_request_count(self) -> None:
        with pytest.raises(
            click.ClickException, match="request_count must be a positive"
        ):
            _load_lifecycle_policy_report_payload(_policy(request_count=0))

    def test_rejects_non_string_hash_list(self) -> None:
        with pytest.raises(
            click.ClickException,
            match="renewal_required_request_hashes must be a string list",
        ):
            _load_lifecycle_policy_report_payload(
                _policy(renewal_required_request_hashes=[1])
            )


def _store(**overrides: Any) -> dict[str, Any]:
    store: dict[str, Any] = {
        "store_hash": _HEX,
        "policy_hash": _HEX,
        "summary_hash": _HEX,
        "renewal_required_request_hashes": [],
        "storage_missing_request_hashes": [],
        "missing_adapter_request_hashes": [],
        "external_write_followup_request_hashes": [],
    }
    store.update(overrides)
    return store


def _drilldown(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1",
        "drilldown_hash": _HEX,
        "policy_count": 1,
        "stores": [_store()],
        "global_flagged_request_hashes": [],
    }
    payload.update(overrides)
    return payload


class TestMultistoreDrilldownLoader:
    def test_valid_round_trips(self) -> None:
        assert (
            _load_lifecycle_multistore_drilldown_payload(_drilldown())["policy_count"]
            == 1
        )

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="multi-store drilldown schema"):
            _load_lifecycle_multistore_drilldown_payload(_drilldown(schema="x"))

    def test_rejects_non_positive_policy_count(self) -> None:
        with pytest.raises(click.ClickException, match="policy_count must be positive"):
            _load_lifecycle_multistore_drilldown_payload(_drilldown(policy_count=0))

    def test_rejects_empty_stores(self) -> None:
        with pytest.raises(click.ClickException, match="stores must be non-empty list"):
            _load_lifecycle_multistore_drilldown_payload(_drilldown(stores=[]))

    def test_rejects_non_object_store(self) -> None:
        with pytest.raises(click.ClickException, match="store record must be object"):
            _load_lifecycle_multistore_drilldown_payload(_drilldown(stores=["x"]))

    def test_rejects_non_string_store_hash_list(self) -> None:
        with pytest.raises(
            click.ClickException,
            match="storage_missing_request_hashes must be a string list",
        ):
            _load_lifecycle_multistore_drilldown_payload(
                _drilldown(stores=[_store(storage_missing_request_hashes=[1])])
            )

    def test_rejects_non_string_global_flagged_hashes(self) -> None:
        with pytest.raises(
            click.ClickException,
            match="global_flagged_request_hashes must be a string list",
        ):
            _load_lifecycle_multistore_drilldown_payload(
                _drilldown(global_flagged_request_hashes=[1])
            )

    def test_rejects_malformed_global_flagged_hashes(self) -> None:
        with pytest.raises(
            click.ClickException,
            match="global_flagged_request_hash 'short' is not a valid",
        ):
            _load_lifecycle_multistore_drilldown_payload(
                _drilldown(global_flagged_request_hashes=["short"])
            )

    def test_rejects_malformed_store_hash_list_items(self) -> None:
        with pytest.raises(
            click.ClickException,
            match="external_write_followup_request_hashes 'short' is not a valid",
        ):
            _load_lifecycle_multistore_drilldown_payload(
                _drilldown(
                    stores=[_store(external_write_followup_request_hashes=["short"])]
                )
            )


def _action(**overrides: Any) -> dict[str, Any]:
    action: dict[str, Any] = {
        "action_hash": _HEX,
        "request_hash": _HEX,
        "store_hash": _HEX,
        "policy_hash": _HEX,
        "summary_hash": _HEX,
        "action_type": "renew_approval",
        "priority": 1,
    }
    action.update(overrides)
    return action


def _remediation_plan(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "scpn_plugin_execution_request_lifecycle_remediation_plan_v1",
        "plan_hash": _HEX,
        "drilldown_hash": _HEX,
        "action_count": 1,
        "actions": [_action()],
    }
    payload.update(overrides)
    return payload


class TestRemediationPlanLoader:
    def test_valid_round_trips(self) -> None:
        assert (
            _load_lifecycle_remediation_plan_payload(_remediation_plan())[
                "action_count"
            ]
            == 1
        )

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="remediation plan schema"):
            _load_lifecycle_remediation_plan_payload(_remediation_plan(schema="x"))

    def test_rejects_negative_action_count(self) -> None:
        with pytest.raises(
            click.ClickException, match="action_count must be non-negative"
        ):
            _load_lifecycle_remediation_plan_payload(_remediation_plan(action_count=-1))

    def test_rejects_count_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="does not match actions"):
            _load_lifecycle_remediation_plan_payload(_remediation_plan(action_count=2))

    def test_rejects_non_object_action(self) -> None:
        with pytest.raises(click.ClickException, match="action must be an object"):
            _load_lifecycle_remediation_plan_payload(
                _remediation_plan(actions=["x"], action_count=1)
            )

    def test_rejects_unsupported_action_type(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported action_type"):
            _load_lifecycle_remediation_plan_payload(
                _remediation_plan(actions=[_action(action_type="teleport")])
            )

    def test_rejects_non_positive_priority(self) -> None:
        with pytest.raises(click.ClickException, match="priority must be a positive"):
            _load_lifecycle_remediation_plan_payload(
                _remediation_plan(actions=[_action(priority=0)])
            )


def _action_status(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
        ),
        "status_hash": _HEX,
        "action_hash": _HEX,
        "plan_hash": _HEX,
        "state": "pending",
    }
    payload.update(overrides)
    return payload


class TestRemediationActionStatusLoader:
    def test_valid_round_trips(self) -> None:
        result = _load_lifecycle_remediation_action_status_payload(_action_status())
        assert result["state"] == "pending"

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="action status schema"):
            _load_lifecycle_remediation_action_status_payload(
                _action_status(schema="x")
            )

    def test_rejects_unsupported_state(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported state"):
            _load_lifecycle_remediation_action_status_payload(
                _action_status(state="frozen")
            )
