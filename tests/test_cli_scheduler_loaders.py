# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI remediation scheduler/handoff loader guards

from __future__ import annotations

from typing import Any

import click
import pytest

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_lifecycle_remediation_deployment_handoff_payload,
    _load_lifecycle_remediation_execution_dashboard_payload,
    _load_lifecycle_remediation_scheduler_acknowledgement_payload,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _load_lifecycle_remediation_scheduler_queue_payload,
    _load_lifecycle_remediation_scheduler_telemetry_payload,
)

_HEX = "a" * 64
_PREFIX = "scpn_plugin_execution_request_lifecycle_remediation"


def _dashboard(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_execution_dashboard_v1",
        "execution_hash": _HEX,
        "plan_hash": _HEX,
        "action_count": 1,
        "rows": [
            {
                "action_hash": _HEX,
                "request_hash": _HEX,
                "state": "pending",
                "action_type": "renew_approval",
                "priority": 1,
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestExecutionDashboardLoader:
    def test_valid_round_trips(self) -> None:
        result = _load_lifecycle_remediation_execution_dashboard_payload(_dashboard())
        assert result["action_count"] == 1

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="execution dashboard schema"):
            _load_lifecycle_remediation_execution_dashboard_payload(
                _dashboard(schema="x")
            )

    def test_rejects_count_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="does not match rows"):
            _load_lifecycle_remediation_execution_dashboard_payload(
                _dashboard(action_count=2)
            )

    def test_rejects_unsupported_state(self) -> None:
        rows = [{**_dashboard()["rows"][0], "state": "frozen"}]
        with pytest.raises(click.ClickException, match="unsupported state"):
            _load_lifecycle_remediation_execution_dashboard_payload(
                _dashboard(rows=rows)
            )

    def test_rejects_non_positive_priority(self) -> None:
        rows = [{**_dashboard()["rows"][0], "priority": 0}]
        with pytest.raises(click.ClickException, match="priority must be a positive"):
            _load_lifecycle_remediation_execution_dashboard_payload(
                _dashboard(rows=rows)
            )


def _handoff(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_deployment_handoff_v1",
        "handoff_hash": _HEX,
        "plan_hash": _HEX,
        "execution_hash": _HEX,
        "unresolved_action_count": 1,
        "handoff_actions": [
            {
                "handoff_action_hash": _HEX,
                "action_hash": _HEX,
                "request_hash": _HEX,
                "action_type": "renew_approval",
                "priority": 1,
                "deployment_command_template": "deploy --review",
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestDeploymentHandoffLoader:
    def test_valid_round_trips(self) -> None:
        result = _load_lifecycle_remediation_deployment_handoff_payload(_handoff())
        assert result["unresolved_action_count"] == 1

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="deployment handoff schema"):
            _load_lifecycle_remediation_deployment_handoff_payload(_handoff(schema="x"))

    def test_rejects_count_mismatch(self) -> None:
        with pytest.raises(
            click.ClickException, match="does not match handoff_actions"
        ):
            _load_lifecycle_remediation_deployment_handoff_payload(
                _handoff(unresolved_action_count=3)
            )

    def test_rejects_empty_command_template(self) -> None:
        actions = [
            {**_handoff()["handoff_actions"][0], "deployment_command_template": ""}
        ]
        with pytest.raises(
            click.ClickException, match="deployment_command_template must be non-empty"
        ):
            _load_lifecycle_remediation_deployment_handoff_payload(
                _handoff(handoff_actions=actions)
            )


def _queue(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_scheduler_queue_v1",
        "scheduler_hash": _HEX,
        "plan_hash": _HEX,
        "execution_hash": _HEX,
        "handoff_hash": _HEX,
        "queue_entry_count": 1,
        "queue_entries": [
            {
                "entry_hash": _HEX,
                "handoff_action_hash": _HEX,
                "action_hash": _HEX,
                "request_hash": _HEX,
                "action_type": "renew_approval",
                "priority": 1,
                "schedule_epoch": 0,
                "scheduler_command_template": "schedule --review",
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestSchedulerQueueLoader:
    def test_valid_round_trips(self) -> None:
        assert (
            _load_lifecycle_remediation_scheduler_queue_payload(_queue())[
                "queue_entry_count"
            ]
            == 1
        )

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="scheduler queue schema"):
            _load_lifecycle_remediation_scheduler_queue_payload(_queue(schema="x"))

    def test_rejects_negative_schedule_epoch(self) -> None:
        entries = [{**_queue()["queue_entries"][0], "schedule_epoch": -1}]
        with pytest.raises(click.ClickException, match="schedule_epoch must be"):
            _load_lifecycle_remediation_scheduler_queue_payload(
                _queue(queue_entries=entries)
            )


def _telemetry(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_scheduler_telemetry_v1",
        "telemetry_hash": _HEX,
        "plan_hash": _HEX,
        "execution_hash": _HEX,
        "handoff_hash": _HEX,
        "scheduler_hash": _HEX,
        "queue_entry_count": 0,
        "rows": [],
    }
    payload.update(overrides)
    return payload


class TestSchedulerTelemetryLoader:
    def test_valid_round_trips(self) -> None:
        assert (
            _load_lifecycle_remediation_scheduler_telemetry_payload(_telemetry())[
                "queue_entry_count"
            ]
            == 0
        )

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="scheduler telemetry schema"):
            _load_lifecycle_remediation_scheduler_telemetry_payload(
                _telemetry(schema="x")
            )

    def test_rejects_count_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="does not match rows"):
            _load_lifecycle_remediation_scheduler_telemetry_payload(
                _telemetry(queue_entry_count=2)
            )


def _adapter_handoff(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_scheduler_adapter_handoff_v1",
        "adapter_handoff_hash": _HEX,
        "telemetry_hash": _HEX,
        "plan_hash": _HEX,
        "execution_hash": _HEX,
        "entry_count": 1,
        "entries": [
            {
                "adapter_entry_hash": _HEX,
                "entry_hash": _HEX,
                "action_hash": _HEX,
                "request_hash": _HEX,
            }
        ],
    }
    payload.update(overrides)
    return payload


class TestSchedulerAdapterHandoffLoader:
    def test_valid_round_trips(self) -> None:
        assert (
            _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
                _adapter_handoff()
            )["entry_count"]
            == 1
        )

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="adapter handoff schema"):
            _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
                _adapter_handoff(schema="x")
            )

    def test_rejects_count_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="does not match entries"):
            _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
                _adapter_handoff(entry_count=2)
            )

    def test_rejects_non_object_entry(self) -> None:
        with pytest.raises(click.ClickException, match="entry must be"):
            _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
                _adapter_handoff(entries=["x"])
            )


def _acknowledgement(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_scheduler_acknowledgement_v1",
        "acknowledgement_hash": _HEX,
        "adapter_handoff_hash": _HEX,
        "telemetry_hash": _HEX,
        "plan_hash": _HEX,
        "execution_hash": _HEX,
        "adapter_entry_hash": _HEX,
        "entry_hash": _HEX,
        "action_hash": _HEX,
        "request_hash": _HEX,
        "state": "in_progress",
        "acknowledged_by": "operator",
        "external_reference": "EXT-1",
    }
    payload.update(overrides)
    return payload


class TestSchedulerAcknowledgementLoader:
    def test_valid_round_trips(self) -> None:
        assert (
            _load_lifecycle_remediation_scheduler_acknowledgement_payload(
                _acknowledgement()
            )["state"]
            == "in_progress"
        )

    def test_rejects_schema_mismatch(self) -> None:
        with pytest.raises(click.ClickException, match="acknowledgement schema"):
            _load_lifecycle_remediation_scheduler_acknowledgement_payload(
                _acknowledgement(schema="x")
            )

    def test_rejects_unsupported_state(self) -> None:
        with pytest.raises(click.ClickException, match="unsupported state"):
            _load_lifecycle_remediation_scheduler_acknowledgement_payload(
                _acknowledgement(state="pending")
            )

    def test_rejects_empty_string_field(self) -> None:
        with pytest.raises(click.ClickException, match="acknowledged_by must be"):
            _load_lifecycle_remediation_scheduler_acknowledgement_payload(
                _acknowledgement(acknowledged_by="")
            )
