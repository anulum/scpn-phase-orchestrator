# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scheduler control command contracts


"""Automation-profile and capture CLI contracts for scheduler control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tests.scheduler_control_fixtures import (
    _adapter_handoff_payload,
    _assert_fails,
    _automation_profile_payload,
    _automation_rule,
    _hex,
    _invoke_json,
    _runbook_group,
    _runbook_payload,
    _runbook_step,
    _write_json,
)


def test_automation_profile_maps_all_runbook_controls(tmp_path: Path) -> None:
    """Automation profile creation maps runbook controls to target states."""
    runbook_path = _write_json(tmp_path, "runbook.json", _runbook_payload())

    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-automation-profile",
            str(runbook_path),
            "--profile-name",
            "airflow-default",
            "--profile-version",
            "1.0.0",
            "--created-by",
            "operator_console",
        ]
    )

    rules = cast(list[dict[str, Any]], payload["automation_rules"])
    by_action = {cast(str, rule["control_action"]): rule for rule in rules}
    assert by_action["dispatch"]["automation_mode"] == "auto"
    assert by_action["monitor"]["target_state"] == "in_progress"
    assert by_action["expedite"]["automation_mode"] == "auto"
    assert by_action["escalate"]["target_state"] == "blocked"
    assert by_action["no_op"]["target_state"] == "completed"
    assert len(cast(str, payload["automation_profile_hash"])) == 64


@pytest.mark.parametrize(
    ("payload", "args", "expected"),
    [
        (
            _runbook_payload(),
            ["--profile-name", "x", "--profile-version", "1", "--created-by", ""],
            "created_by must be",
        ),
        (
            _runbook_payload(),
            [
                "--profile-name",
                "",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "profile_name must be",
        ),
        (
            _runbook_payload(),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "",
                "--created-by",
                "operator_console",
            ],
            "profile_version must be",
        ),
        (
            _runbook_payload(schema="wrong"),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "unexpected scheduler runbook schema",
        ),
        (
            _runbook_payload(groups="wrong"),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "groups must be a list",
        ),
        (
            _runbook_payload(groups=[cast(dict[str, Any], "wrong")]),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "group must be object",
        ),
        (
            _runbook_payload(groups=[_runbook_group("unknown", [])]),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "unsupported control_action",
        ),
        (
            _runbook_payload(groups=[{"control_action": "dispatch", "steps": "wrong"}]),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "steps must be list",
        ),
        (
            _runbook_payload(
                groups=[_runbook_group("dispatch", [cast(dict[str, Any], "wrong")])]
            ),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "step must be object",
        ),
        (
            _runbook_payload(
                groups=[
                    _runbook_group(
                        "dispatch",
                        [
                            {
                                **_runbook_step(
                                    action_hash=_hex("a"),
                                    request_hash=_hex("b"),
                                    control_action="dispatch",
                                    priority=1,
                                ),
                                "action_type": "",
                            }
                        ],
                    )
                ]
            ),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "action_type must be non-empty",
        ),
        (
            _runbook_payload(
                groups=[
                    _runbook_group(
                        "dispatch",
                        [
                            {
                                **_runbook_step(
                                    action_hash=_hex("a"),
                                    request_hash=_hex("b"),
                                    control_action="dispatch",
                                    priority=1,
                                ),
                                "priority": 0,
                            }
                        ],
                    )
                ]
            ),
            [
                "--profile-name",
                "x",
                "--profile-version",
                "1",
                "--created-by",
                "operator_console",
            ],
            "priority must be positive",
        ),
    ],
)
def test_automation_profile_rejects_invalid_runbook_contracts(
    tmp_path: Path,
    payload: dict[str, Any],
    args: list[str],
    expected: str,
) -> None:
    """Automation profile creation rejects malformed runbook inputs."""
    runbook_path = _write_json(tmp_path, "runbook.json", payload)

    _assert_fails(
        [
            "lifecycle-remediation-scheduler-automation-profile",
            str(runbook_path),
            *args,
        ],
        expected,
    )


def test_acknowledgement_capture_binds_profile_rule_to_adapter_entry(
    tmp_path: Path,
) -> None:
    """Acknowledgement capture binds a profile rule to a matching adapter entry."""
    profile_path = _write_json(
        tmp_path,
        "profile.json",
        _automation_profile_payload(
            rules=[
                _automation_rule(
                    action_hash=_hex("a"),
                    request_hash=_hex("b"),
                    control_action="dispatch",
                    automation_mode="auto",
                    target_state="in_progress",
                )
            ]
        ),
    )
    adapter_path = _write_json(tmp_path, "adapter.json", _adapter_handoff_payload())

    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-acknowledgement-capture",
            str(profile_path),
            str(adapter_path),
            _hex("a"),
            "--external-reference",
            "airflow-run-1",
            "--acknowledged-by",
            "operator_console",
            "--captured-state",
            "in_progress",
            "--note",
            "captured",
        ]
    )

    assert payload["action_hash"] == _hex("a")
    assert payload["adapter_entry_hash"] == _hex("k")
    assert len(cast(str, payload["capture_hash"])) == 64


@pytest.mark.parametrize(
    ("profile_payload", "adapter_payload", "args", "expected"),
    [
        (
            _automation_profile_payload(),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "in_progress",
            ],
            "external_reference must be",
        ),
        (
            _automation_profile_payload(),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "",
                "--captured-state",
                "in_progress",
            ],
            "acknowledged_by must be",
        ),
        (
            _automation_profile_payload(schema="wrong"),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "in_progress",
            ],
            "unexpected automation profile schema",
        ),
        (
            _automation_profile_payload(automation_rules="wrong"),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "in_progress",
            ],
            "automation_rules must be list",
        ),
        (
            _automation_profile_payload(),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "in_progress",
            ],
            "action_hash not present in automation profile",
        ),
        (
            _automation_profile_payload(),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "completed",
            ],
            "captured_state does not match auto target_state",
        ),
        (
            _automation_profile_payload(plan_hash=_hex("x")),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "in_progress",
            ],
            "plan_hash mismatch",
        ),
        (
            _automation_profile_payload(
                rules=[
                    _automation_rule(
                        action_hash=_hex("c"),
                        request_hash=_hex("d"),
                        control_action="monitor",
                        automation_mode="auto",
                        target_state="in_progress",
                    )
                ]
            ),
            _adapter_handoff_payload(),
            [
                "--external-reference",
                "run",
                "--acknowledged-by",
                "operator",
                "--captured-state",
                "in_progress",
            ],
            "action_hash not present in adapter handoff",
        ),
    ],
)
def test_acknowledgement_capture_rejects_invalid_profile_and_adapter_contracts(
    tmp_path: Path,
    profile_payload: dict[str, Any],
    adapter_payload: dict[str, Any],
    args: list[str],
    expected: str,
) -> None:
    """Acknowledgement capture rejects unbound or inconsistent artifacts."""
    profile_path = _write_json(tmp_path, "profile.json", profile_payload)
    adapter_path = _write_json(tmp_path, "adapter.json", adapter_payload)
    action_hash = _hex("a")
    if expected == "action_hash not present in adapter handoff":
        action_hash = _hex("c")
    elif expected == "action_hash not present in automation profile":
        action_hash = _hex("z")

    _assert_fails(
        [
            "lifecycle-remediation-scheduler-acknowledgement-capture",
            str(profile_path),
            str(adapter_path),
            action_hash,
            *args,
        ],
        expected,
    )
