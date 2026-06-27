# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scheduler contract guards

"""Public CLI contract tests for scheduler validation and replay fail-closed paths."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeVar, cast

import pytest
from click.testing import CliRunner, Result

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli._payloads import _record_hash

Payload = dict[str, object]
_F = TypeVar("_F", bound=Callable[..., object])

_PREFIX = "scpn_plugin_execution_request_lifecycle_remediation"
_PLAN_HASH = "1" * 64
_EXECUTION_HASH = "2" * 64
_HANDOFF_HASH = "3" * 64
_SCHEDULER_HASH = "4" * 64
_TELEMETRY_HASH = "5" * 64
_ADAPTER_HANDOFF_HASH = "6" * 64
_ENTRY_HASH = "7" * 64
_HANDOFF_ACTION_HASH = "8" * 64
_ACTION_HASH = "9" * 64
_REQUEST_HASH = "a" * 64
_ADAPTER_ENTRY_HASH = "b" * 64
_ACKNOWLEDGEMENT_HASH = "c" * 64
_REPLAY_HASH = "d" * 64


class _PytestFixtureDecorator(Protocol):
    """Typed facade for pytest fixture decorators."""

    def __call__(self, func: _F) -> _F:
        """Return a fixture function while preserving its type."""


class _PytestParametrizeDecorator(Protocol):
    """Typed facade for pytest parametrize decorators."""

    def __call__(
        self,
        argnames: str | Sequence[str],
        argvalues: object,
    ) -> Callable[[_F], _F]:
        """Return a parametrized test decorator preserving the test type."""


_pytest_fixture = cast(_PytestFixtureDecorator, pytest.fixture)
_pytest_parametrize = cast(_PytestParametrizeDecorator, pytest.mark.parametrize)


@_pytest_fixture
def runner() -> CliRunner:
    """Return a Click runner for public scheduler command invocations."""
    return CliRunner()


def _write_payload(path: Path, payload: Mapping[str, object]) -> Path:
    """Write ``payload`` to ``path`` as deterministic UTF-8 JSON."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _invoke(runner: CliRunner, args: Sequence[str]) -> Result:
    """Invoke ``spo plugins`` with scheduler command arguments."""
    return runner.invoke(main, ["plugins", *args])


def _assert_fails(result: Result, expected_message: str) -> None:
    """Assert that a scheduler command failed with the expected message."""
    assert result.exit_code == 1, result.output
    assert expected_message in result.output


def _with_replaced_options(
    args: Sequence[str],
    option_args: Sequence[str],
) -> list[str]:
    """Return command ``args`` with each option in ``option_args`` replaced."""
    replaced = list(args)
    for option in option_args[::2]:
        option_index = replaced.index(option)
        del replaced[option_index : option_index + 2]
    replaced.extend(option_args)
    return replaced


def _deployment_handoff_payload() -> Payload:
    """Return a valid remediation deployment handoff payload."""
    action: Payload = {
        "handoff_action_hash": _HANDOFF_ACTION_HASH,
        "action_hash": _ACTION_HASH,
        "request_hash": _REQUEST_HASH,
        "action_type": "renew_approval",
        "priority": 1,
        "deployment_command_template": "spo plugins approve-execution-plan PLAN_JSON",
    }
    payload: Payload = {
        "schema": f"{_PREFIX}_deployment_handoff_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "handoff_actions": [action],
        "unresolved_action_count": 1,
        "created_by": "deployment_gate",
    }
    payload["handoff_hash"] = _record_hash(payload)
    return payload


def _scheduler_queue_payload() -> Payload:
    """Return a valid scheduler queue payload."""
    entry: Payload = {
        "entry_hash": _ENTRY_HASH,
        "handoff_action_hash": _HANDOFF_ACTION_HASH,
        "action_hash": _ACTION_HASH,
        "request_hash": _REQUEST_HASH,
        "action_type": "renew_approval",
        "priority": 1,
        "schedule_epoch": 1_700_000_000,
        "scheduler_command_template": "spo plugins approve-execution-plan PLAN_JSON",
    }
    payload: Payload = {
        "schema": f"{_PREFIX}_scheduler_queue_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "handoff_hash": _HANDOFF_HASH,
        "window_start_epoch": 1_700_000_000,
        "window_duration_seconds": 3600,
        "queue_entry_count": 1,
        "queue_entries": [entry],
        "created_by": "deployment_scheduler",
    }
    payload["scheduler_hash"] = _record_hash(payload)
    return payload


def _scheduler_telemetry_payload(
    *,
    telemetry_hash: str = _TELEMETRY_HASH,
    row_state: str = "pending",
) -> Payload:
    """Return a scheduler telemetry payload with one row."""
    row: Payload = {
        "entry_hash": _ENTRY_HASH,
        "handoff_action_hash": _HANDOFF_ACTION_HASH,
        "action_hash": _ACTION_HASH,
        "request_hash": _REQUEST_HASH,
        "action_type": "renew_approval",
        "priority": 1,
        "schedule_epoch": 1_700_000_000,
        "state": row_state,
        "overdue": True,
        "status_hash": None,
        "updated_by": None,
        "note": "",
    }
    payload: Payload = {
        "schema": f"{_PREFIX}_scheduler_telemetry_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "handoff_hash": _HANDOFF_HASH,
        "scheduler_hash": _SCHEDULER_HASH,
        "as_of_epoch": 1_700_000_100,
        "queue_entry_count": 1,
        "state_counts": {
            "pending": 1 if row_state == "pending" else 0,
            "in_progress": 0,
            "completed": 0,
            "blocked": 0,
            "overdue": 1,
        },
        "overdue_action_hashes": [_ACTION_HASH],
        "rows": [row],
        "created_by": "deployment_scheduler",
        "telemetry_hash": telemetry_hash,
    }
    return payload


def _adapter_handoff_payload() -> Payload:
    """Return a valid scheduler adapter handoff payload."""
    entry: Payload = {
        "adapter_entry_hash": _ADAPTER_ENTRY_HASH,
        "entry_hash": _ENTRY_HASH,
        "handoff_action_hash": _HANDOFF_ACTION_HASH,
        "action_hash": _ACTION_HASH,
        "request_hash": _REQUEST_HASH,
        "action_type": "renew_approval",
        "priority": 1,
        "schedule_epoch": 1_700_000_000,
        "overdue": True,
        "adapter_target": {
            "adapter_name": "airflow",
            "adapter_endpoint": "airflow://cluster-a",
        },
        "acknowledgement_command_template": (
            "spo plugins lifecycle-remediation-scheduler-acknowledgement "
            "ADAPTER_HANDOFF_JSON ENTRY_HASH --state STATE --acknowledged-by "
            "OPERATOR --external-reference REF"
        ),
    }
    payload: Payload = {
        "schema": f"{_PREFIX}_scheduler_adapter_handoff_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "telemetry_hash": _TELEMETRY_HASH,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "entry_count": 1,
        "entries": [entry],
        "created_by": "deployment_scheduler",
        "adapter_handoff_hash": _ADAPTER_HANDOFF_HASH,
    }
    return payload


def _acknowledgement_payload(
    *,
    adapter_entry_hash: str = _ADAPTER_ENTRY_HASH,
    acknowledgement_hash: str = _ACKNOWLEDGEMENT_HASH,
) -> Payload:
    """Return a scheduler acknowledgement payload."""
    return {
        "schema": f"{_PREFIX}_scheduler_acknowledgement_v1",
        "version": "1.0.0",
        "adapter_handoff_hash": _ADAPTER_HANDOFF_HASH,
        "telemetry_hash": _TELEMETRY_HASH,
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "adapter_entry_hash": adapter_entry_hash,
        "entry_hash": _ENTRY_HASH,
        "action_hash": _ACTION_HASH,
        "request_hash": _REQUEST_HASH,
        "state": "completed",
        "acknowledged_by": "airflow_worker",
        "external_reference": "airflow-run-1",
        "note": "",
        "acknowledgement_hash": acknowledgement_hash,
    }


def _replay_payload(
    *,
    telemetry_hash: str = _TELEMETRY_HASH,
    rows: list[Payload] | None = None,
    schema: str | None = None,
) -> Payload:
    """Return a scheduler acknowledgement replay payload."""
    replay_rows = rows if rows is not None else []
    return {
        "schema": schema or f"{_PREFIX}_scheduler_acknowledgement_replay_v1",
        "version": "1.0.0",
        "adapter_handoff_hash": _ADAPTER_HANDOFF_HASH,
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "telemetry_hash": telemetry_hash,
        "acknowledgement_count": len(replay_rows),
        "state_counts": {"in_progress": 0, "completed": 0, "blocked": 0},
        "rows": replay_rows,
        "created_by": "deployment_scheduler",
        "replay_hash": _REPLAY_HASH,
    }


def _replay_row(
    *, state: str = "completed", action_hash: str = _ACTION_HASH
) -> Payload:
    """Return one replay row for dashboard aggregation tests."""
    return {
        "acknowledgement_hash": _ACKNOWLEDGEMENT_HASH,
        "adapter_entry_hash": _ADAPTER_ENTRY_HASH,
        "entry_hash": _ENTRY_HASH,
        "action_hash": action_hash,
        "request_hash": _REQUEST_HASH,
        "state": state,
        "external_reference": "airflow-run-1",
        "acknowledged_by": "airflow_worker",
        "note": "",
        "replay_row_hash": _record_hash(
            {
                "acknowledgement_hash": _ACKNOWLEDGEMENT_HASH,
                "adapter_entry_hash": _ADAPTER_ENTRY_HASH,
                "entry_hash": _ENTRY_HASH,
                "action_hash": action_hash,
                "request_hash": _REQUEST_HASH,
                "state": state,
                "external_reference": "airflow-run-1",
                "acknowledged_by": "airflow_worker",
                "note": "",
            }
        ),
    }


@_pytest_parametrize(
    ("option_args", "expected_message"),
    [
        (["--created-by", ""], "created_by must be non-empty"),
        (
            ["--window-start-epoch", "-1"],
            "window_start_epoch must be non-negative",
        ),
        (
            ["--window-duration-seconds", "0"],
            "window_duration_seconds must be positive",
        ),
    ],
)
def test_scheduler_queue_rejects_invalid_command_options(
    runner: CliRunner,
    tmp_path: Path,
    option_args: list[str],
    expected_message: str,
) -> None:
    """Queue command rejects invalid scheduler metadata before loading output."""
    handoff_path = _write_payload(
        tmp_path / "handoff.json",
        _deployment_handoff_payload(),
    )
    args = [
        "lifecycle-remediation-scheduler-queue",
        str(handoff_path),
        "--window-start-epoch",
        "1700000000",
        "--window-duration-seconds",
        "3600",
        "--created-by",
        "deployment_scheduler",
    ]

    result = _invoke(runner, _with_replaced_options(args, option_args))

    _assert_fails(result, expected_message)


@_pytest_parametrize(
    ("option_args", "expected_message"),
    [
        (["--created-by", ""], "created_by must be non-empty"),
        (["--as-of-epoch", "-1"], "as_of_epoch must be non-negative"),
    ],
)
def test_scheduler_telemetry_rejects_invalid_command_options(
    runner: CliRunner,
    tmp_path: Path,
    option_args: list[str],
    expected_message: str,
) -> None:
    """Telemetry command rejects invalid scheduler metadata before loading output."""
    queue_path = _write_payload(tmp_path / "queue.json", _scheduler_queue_payload())
    args = [
        "lifecycle-remediation-scheduler-telemetry",
        str(queue_path),
        "--as-of-epoch",
        "1700000100",
        "--created-by",
        "deployment_scheduler",
    ]

    result = _invoke(runner, _with_replaced_options(args, option_args))

    _assert_fails(result, expected_message)


@_pytest_parametrize(
    ("option_args", "expected_message"),
    [
        (["--created-by", ""], "created_by must be non-empty"),
        (["--adapter-name", ""], "adapter_name must be non-empty"),
        (["--adapter-endpoint", ""], "adapter_endpoint must be non-empty"),
    ],
)
def test_scheduler_adapter_handoff_rejects_invalid_command_options(
    runner: CliRunner,
    tmp_path: Path,
    option_args: list[str],
    expected_message: str,
) -> None:
    """Adapter handoff command rejects invalid scheduler metadata."""
    telemetry_path = _write_payload(
        tmp_path / "telemetry.json",
        _scheduler_telemetry_payload(),
    )
    args = [
        "lifecycle-remediation-scheduler-adapter-handoff",
        str(telemetry_path),
        "--adapter-name",
        "airflow",
        "--adapter-endpoint",
        "airflow://cluster-a",
        "--created-by",
        "deployment_scheduler",
    ]

    result = _invoke(runner, _with_replaced_options(args, option_args))

    _assert_fails(result, expected_message)


@_pytest_parametrize(
    ("option_args", "expected_message"),
    [
        (["--acknowledged-by", ""], "acknowledged_by must be non-empty"),
        (["--external-reference", ""], "external_reference must be non-empty"),
    ],
)
def test_scheduler_acknowledgement_rejects_invalid_command_options(
    runner: CliRunner,
    tmp_path: Path,
    option_args: list[str],
    expected_message: str,
) -> None:
    """Acknowledgement command rejects invalid operator metadata."""
    adapter_path = _write_payload(
        tmp_path / "adapter.json",
        _adapter_handoff_payload(),
    )
    args = [
        "lifecycle-remediation-scheduler-acknowledgement",
        str(adapter_path),
        _ADAPTER_ENTRY_HASH,
        "--state",
        "completed",
        "--acknowledged-by",
        "airflow_worker",
        "--external-reference",
        "airflow-run-1",
    ]

    result = _invoke(runner, _with_replaced_options(args, option_args))

    _assert_fails(result, expected_message)


def test_scheduler_acknowledgement_replay_rejects_empty_creator(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Acknowledgement replay command rejects empty creator identity."""
    adapter_path = _write_payload(
        tmp_path / "adapter.json",
        _adapter_handoff_payload(),
    )

    result = _invoke(
        runner,
        [
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_path),
            "--created-by",
            "",
        ],
    )

    _assert_fails(result, "created_by must be non-empty")


def test_scheduler_acknowledgement_replay_rejects_duplicate_acknowledgements(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Replay command rejects duplicate acknowledgements for one adapter entry."""
    adapter_path = _write_payload(
        tmp_path / "adapter.json",
        _adapter_handoff_payload(),
    )
    ack_a_path = _write_payload(
        tmp_path / "ack-a.json",
        _acknowledgement_payload(acknowledgement_hash=_ACKNOWLEDGEMENT_HASH),
    )
    ack_b_path = _write_payload(
        tmp_path / "ack-b.json",
        _acknowledgement_payload(acknowledgement_hash="e" * 64),
    )

    result = _invoke(
        runner,
        [
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_path),
            str(ack_a_path),
            str(ack_b_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )

    _assert_fails(result, "duplicate adapter_entry_hash acknowledgement")


def test_scheduler_acknowledgement_replay_rejects_ack_missing_from_handoff(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    """Replay command rejects acknowledgements for entries outside the handoff."""
    adapter_path = _write_payload(
        tmp_path / "adapter.json",
        _adapter_handoff_payload(),
    )
    unknown_ack_path = _write_payload(
        tmp_path / "ack-unknown.json",
        _acknowledgement_payload(adapter_entry_hash="f" * 64),
    )

    result = _invoke(
        runner,
        [
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_path),
            str(unknown_ack_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )

    _assert_fails(result, "acknowledgement adapter_entry_hash missing from handoff")


@_pytest_parametrize(
    ("case_name", "expected_message"),
    [
        ("empty_creator", "created_by must be non-empty"),
        ("bad_schema", "unexpected acknowledgement replay schema"),
        ("hash_mismatch", "telemetry_hash mismatch"),
        ("bad_replay_state", "unsupported replay state"),
        ("duplicate_action", "duplicate replay action_hash"),
        ("bad_effective_state", "unsupported effective state"),
    ],
)
def test_scheduler_execution_dashboard_rejects_corrupt_replay_contracts(
    runner: CliRunner,
    tmp_path: Path,
    case_name: str,
    expected_message: str,
) -> None:
    """Dashboard command rejects corrupt replay and effective-state contracts."""
    telemetry_payload = _scheduler_telemetry_payload(
        row_state="deferred" if case_name == "bad_effective_state" else "pending"
    )
    replay_payload = _replay_payload()
    created_by = "deployment_scheduler"
    if case_name == "empty_creator":
        created_by = ""
    elif case_name == "bad_schema":
        replay_payload = _replay_payload(schema="not-a-scheduler-replay")
    elif case_name == "hash_mismatch":
        replay_payload = _replay_payload(telemetry_hash="e" * 64)
    elif case_name == "bad_replay_state":
        replay_payload = _replay_payload(rows=[_replay_row(state="cancelled")])
    elif case_name == "duplicate_action":
        replay_payload = _replay_payload(rows=[_replay_row(), _replay_row()])

    telemetry_path = _write_payload(
        tmp_path / "telemetry.json",
        telemetry_payload,
    )
    replay_path = _write_payload(tmp_path / "replay.json", replay_payload)

    result = _invoke(
        runner,
        [
            "lifecycle-remediation-scheduler-execution-dashboard",
            str(telemetry_path),
            str(replay_path),
            "--created-by",
            created_by,
        ],
    )

    _assert_fails(result, expected_message)
