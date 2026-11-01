# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scheduler control command contracts


"""Control-plan and runbook CLI contracts for scheduler control."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from scpn_phase_orchestrator.runtime.cli.plugins._group import plugins_group
from tests.scheduler_control_fixtures import (
    _REGISTERED_MODULE,
    _adapter_entry,
    _adapter_handoff_payload,
    _assert_fails,
    _control_action,
    _control_plan_payload,
    _dashboard_payload,
    _dashboard_row,
    _hex,
    _invoke_json,
    _write_json,
)

_R_CALL = TypeVar("_R_CALL")


def _call(fn: Callable[..., _R_CALL], *args: object, **kwargs: object) -> _R_CALL:
    """Call ``fn`` with arguments mypy cannot check: invalid or dynamic kwargs."""
    return fn(*args, **kwargs)


def _as_list(value: object) -> list[Any]:
    """Return ``value`` as list, asserting its runtime type."""
    assert isinstance(value, list)
    return value


def _as_str(value: object) -> str:
    """Return ``value`` as str, asserting its runtime type."""
    assert isinstance(value, str)
    return value


def test_scheduler_control_plugin_module_registers_commands() -> None:
    """Assert the scheduler-control module remains imported for registration."""
    assert _REGISTERED_MODULE.__name__.endswith("scheduler_control")
    assert "lifecycle-remediation-scheduler-control-plan" in plugins_group.commands


def test_control_plan_maps_all_dashboard_states(tmp_path: Path) -> None:
    """Control planning maps every dashboard state to the documented action."""
    dashboard_path = _write_json(tmp_path, "dashboard.json", _dashboard_payload())

    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-control-plan",
            str(dashboard_path),
            "--created-by",
            "operator_console",
        ]
    )

    assert payload["control_counts"] == {
        "dispatch": 1,
        "monitor": 1,
        "expedite": 1,
        "escalate": 1,
        "no_op": 1,
    }
    actions = payload["control_actions"]
    assert isinstance(actions, list)
    assert sorted(action["control_action"] for action in actions) == [
        "dispatch",
        "escalate",
        "expedite",
        "monitor",
        "no_op",
    ]
    assert len(_as_str(payload["control_plan_hash"])) == 64


@pytest.mark.parametrize(
    ("payload_factory", "args", "expected"),
    [
        (lambda: _dashboard_payload(), ["--created-by", ""], "created_by must be"),
        (
            lambda: _dashboard_payload(schema="wrong"),
            ["--created-by", "operator_console"],
            "unexpected scheduler execution dashboard schema",
        ),
        (
            lambda: _dashboard_payload(rows="wrong"),
            ["--created-by", "operator_console"],
            "rows must be a list",
        ),
        (
            lambda: _dashboard_payload(rows=["wrong"]),
            ["--created-by", "operator_console"],
            "row must be object",
        ),
        (
            lambda: _dashboard_payload(
                rows=[_dashboard_row(marker="a", effective_state="unknown")]
            ),
            ["--created-by", "operator_console"],
            "unsupported effective_state",
        ),
    ],
)
def test_control_plan_rejects_invalid_dashboard_contracts(
    tmp_path: Path,
    payload_factory: Callable[[], dict[str, Any]],
    args: list[str],
    expected: str,
) -> None:
    """Control planning rejects malformed dashboard inputs before dispatch."""
    dashboard_path = _write_json(tmp_path, "dashboard.json", payload_factory())

    _assert_fails(
        [
            "lifecycle-remediation-scheduler-control-plan",
            str(dashboard_path),
            *args,
        ],
        expected,
    )


def test_runbook_groups_all_actions_and_optional_adapter_entries(
    tmp_path: Path,
) -> None:
    """Runbook creation groups actions and tolerates missing adapter rows."""
    control_path = _write_json(tmp_path, "control.json", _control_plan_payload())
    adapter_path = _write_json(tmp_path, "adapter.json", _adapter_handoff_payload())

    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-runbook",
            str(control_path),
            str(adapter_path),
            "--created-by",
            "operator_console",
        ]
    )

    groups = payload["groups"]
    assert isinstance(groups, list)
    assert [group["control_action"] for group in groups] == [
        "escalate",
        "expedite",
        "dispatch",
        "monitor",
        "no_op",
    ]
    dispatch_step = _as_list(groups[2]["steps"])[0]
    monitor_step = _as_list(groups[3]["steps"])[0]
    assert dispatch_step["adapter_name"] == "airflow"
    assert monitor_step["adapter_name"] is None
    assert len(_as_str(payload["runbook_hash"])) == 64


@pytest.mark.parametrize(
    ("control_payload", "adapter_payload", "args", "expected"),
    [
        (
            _control_plan_payload(),
            _adapter_handoff_payload(),
            ["--created-by", ""],
            "created_by must be",
        ),
        (
            _control_plan_payload(schema="wrong"),
            _adapter_handoff_payload(),
            ["--created-by", "operator_console"],
            "unexpected scheduler control plan schema",
        ),
        (
            _control_plan_payload(plan_hash=_hex("x")),
            _adapter_handoff_payload(),
            ["--created-by", "operator_console"],
            "plan_hash mismatch",
        ),
        (
            _control_plan_payload(control_actions="wrong"),
            _adapter_handoff_payload(),
            ["--created-by", "operator_console"],
            "control_actions must be a list",
        ),
        (
            _call(_control_plan_payload, actions=["wrong"]),
            _adapter_handoff_payload(),
            ["--created-by", "operator_console"],
            "control action must be object",
        ),
        (
            _control_plan_payload(
                actions=[
                    _control_action(
                        action_hash=_hex("a"),
                        request_hash=_hex("b"),
                        control_action="unknown",
                    )
                ]
            ),
            _adapter_handoff_payload(),
            ["--created-by", "operator_console"],
            "unsupported control_action",
        ),
        (
            _control_plan_payload(),
            _adapter_handoff_payload(
                entries=[
                    _adapter_entry(
                        action_hash=_hex("a"),
                        request_hash=_hex("b"),
                        marker="k",
                    ),
                    _adapter_entry(
                        action_hash=_hex("a"),
                        request_hash=_hex("b"),
                        marker="m",
                    ),
                ]
            ),
            ["--created-by", "operator_console"],
            "duplicate action_hash in adapter handoff",
        ),
    ],
)
def test_runbook_rejects_invalid_control_plan_and_handoff_contracts(
    tmp_path: Path,
    control_payload: dict[str, Any],
    adapter_payload: dict[str, Any],
    args: list[str],
    expected: str,
) -> None:
    """Runbook creation rejects inconsistent control-plan and adapter inputs."""
    control_path = _write_json(tmp_path, "control.json", control_payload)
    adapter_path = _write_json(tmp_path, "adapter.json", adapter_payload)

    _assert_fails(
        [
            "lifecycle-remediation-scheduler-runbook",
            str(control_path),
            str(adapter_path),
            *args,
        ],
        expected,
    )
