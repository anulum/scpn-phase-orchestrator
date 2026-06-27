# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scheduler control command contracts


"""Retry-profile and orchestration CLI contracts for scheduler control."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tests.scheduler_control_fixtures import (
    _assert_fails,
    _automation_profile_payload,
    _automation_rule,
    _capture_payload,
    _hex,
    _invoke_json,
    _retry_profile_payload,
    _retry_rule,
    _write_json,
)


def test_retry_profile_maps_retry_enabled_only_for_auto_dispatch_and_expedite(
    tmp_path: Path,
) -> None:
    """Retry profile creation enables retry only for auto dispatch and expedite."""
    profile_path = _write_json(tmp_path, "profile.json", _automation_profile_payload())

    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-retry-profile",
            str(profile_path),
            "--max-attempts",
            "4",
            "--base-delay-seconds",
            "20",
            "--backoff-multiplier",
            "2.0",
            "--created-by",
            "operator_console",
        ]
    )

    rules = cast(list[dict[str, Any]], payload["retry_rules"])
    policies = {
        cast(str, rule["control_action"]): cast(str, rule["policy_mode"])
        for rule in rules
    }
    assert policies == {
        "dispatch": "retry_enabled",
        "expedite": "retry_enabled",
        "monitor": "retry_disabled",
        "escalate": "retry_disabled",
        "no_op": "retry_disabled",
    }
    assert len(cast(str, payload["retry_profile_hash"])) == 64


@pytest.mark.parametrize(
    ("payload", "args", "expected"),
    [
        (
            _automation_profile_payload(),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "",
            ],
            "created_by must be",
        ),
        (
            _automation_profile_payload(),
            [
                "--max-attempts",
                "0",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "max_attempts must be positive",
        ),
        (
            _automation_profile_payload(),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "0",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "base_delay_seconds must be positive",
        ),
        (
            _automation_profile_payload(),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "0.5",
                "--created-by",
                "operator",
            ],
            "backoff_multiplier must be",
        ),
        (
            _automation_profile_payload(schema="wrong"),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "unexpected automation profile schema",
        ),
        (
            _automation_profile_payload(automation_rules="wrong"),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "automation_rules must be list",
        ),
        (
            _automation_profile_payload(rules=[cast(dict[str, Any], "wrong")]),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "rule must be object",
        ),
        (
            _automation_profile_payload(
                rules=[
                    {
                        **_automation_rule(
                            action_hash=_hex("a"),
                            request_hash=_hex("b"),
                            control_action="dispatch",
                            automation_mode="unsupported",
                            target_state="in_progress",
                        )
                    }
                ]
            ),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "unsupported automation_mode",
        ),
        (
            _automation_profile_payload(
                rules=[
                    {
                        **_automation_rule(
                            action_hash=_hex("a"),
                            request_hash=_hex("b"),
                            control_action="unknown",
                            automation_mode="auto",
                            target_state="in_progress",
                        )
                    }
                ]
            ),
            [
                "--max-attempts",
                "1",
                "--base-delay-seconds",
                "1",
                "--backoff-multiplier",
                "1.0",
                "--created-by",
                "operator",
            ],
            "unsupported control_action",
        ),
    ],
)
def test_retry_profile_rejects_invalid_automation_contracts(
    tmp_path: Path,
    payload: dict[str, Any],
    args: list[str],
    expected: str,
) -> None:
    """Retry profile creation rejects invalid automation profile inputs."""
    profile_path = _write_json(tmp_path, "profile.json", payload)

    _assert_fails(
        [
            "lifecycle-remediation-scheduler-retry-profile",
            str(profile_path),
            *args,
        ],
        expected,
    )


def test_retry_orchestration_emits_entries_for_retry_enabled_unfinished_captures(
    tmp_path: Path,
) -> None:
    """Retry orchestration emits entries only for retry-enabled unfinished captures."""
    retry_path = _write_json(tmp_path, "retry.json", _retry_profile_payload())
    retry_enabled_blocked = _write_json(
        tmp_path,
        "blocked.json",
        _capture_payload(action_hash=_hex("a"), request_hash=_hex("b")),
    )
    retry_enabled_completed = _write_json(
        tmp_path,
        "completed.json",
        _capture_payload(
            action_hash=_hex("e"),
            request_hash=_hex("f"),
            captured_state="completed",
        ),
    )
    retry_disabled_blocked = _write_json(
        tmp_path,
        "disabled.json",
        _capture_payload(
            action_hash=_hex("c"),
            request_hash=_hex("d"),
            captured_state="blocked",
        ),
    )

    payload = _invoke_json(
        [
            "lifecycle-remediation-scheduler-retry-orchestration",
            str(retry_path),
            str(retry_enabled_blocked),
            str(retry_enabled_completed),
            str(retry_disabled_blocked),
            "--created-by",
            "operator_console",
        ]
    )

    entries = cast(list[dict[str, Any]], payload["retry_entries"])
    assert payload["retry_entry_count"] == 1
    assert entries[0]["action_hash"] == _hex("a")
    assert entries[0]["next_delay_seconds"] == 20
    assert len(cast(str, payload["retry_orchestration_hash"])) == 64


@pytest.mark.parametrize(
    ("retry_payload", "captures", "args", "expected"),
    [
        (
            _retry_profile_payload(),
            [_capture_payload()],
            ["--created-by", ""],
            "created_by must be",
        ),
        (
            _retry_profile_payload(schema="wrong"),
            [_capture_payload()],
            ["--created-by", "operator"],
            "unexpected retry profile schema",
        ),
        (
            _retry_profile_payload(retry_rules="wrong"),
            [_capture_payload()],
            ["--created-by", "operator"],
            "retry_rules must be list",
        ),
        (
            _retry_profile_payload(rules=[cast(dict[str, Any], "wrong")]),
            [_capture_payload()],
            ["--created-by", "operator"],
            "rule must be object",
        ),
        (
            _retry_profile_payload(
                rules=[
                    _retry_rule(
                        action_hash=_hex("a"),
                        request_hash=_hex("b"),
                        control_action="dispatch",
                        policy_mode="retry_enabled",
                    ),
                    _retry_rule(
                        action_hash=_hex("a"),
                        request_hash=_hex("b"),
                        control_action="dispatch",
                        policy_mode="retry_enabled",
                    ),
                ]
            ),
            [_capture_payload()],
            ["--created-by", "operator"],
            "duplicate rule action_hash",
        ),
        (
            _retry_profile_payload(),
            [_capture_payload(schema="wrong")],
            ["--created-by", "operator"],
            "unexpected acknowledgement capture schema",
        ),
        (
            _retry_profile_payload(),
            [_capture_payload(), _capture_payload()],
            ["--created-by", "operator"],
            "duplicate capture action_hash",
        ),
        (
            _retry_profile_payload(),
            [_capture_payload(plan_hash=_hex("x"))],
            ["--created-by", "operator"],
            "plan_hash mismatch",
        ),
        (
            _retry_profile_payload(),
            [_capture_payload(action_hash=_hex("z"), request_hash=_hex("y"))],
            ["--created-by", "operator"],
            "capture action_hash missing from retry profile",
        ),
    ],
)
def test_retry_orchestration_rejects_invalid_profile_and_capture_contracts(
    tmp_path: Path,
    retry_payload: dict[str, Any],
    captures: list[dict[str, Any]],
    args: list[str],
    expected: str,
) -> None:
    """Retry orchestration rejects inconsistent retry and capture artifacts."""
    retry_path = _write_json(tmp_path, "retry.json", retry_payload)
    capture_paths = [
        _write_json(tmp_path, f"capture-{index}.json", capture)
        for index, capture in enumerate(captures)
    ]

    _assert_fails(
        [
            "lifecycle-remediation-scheduler-retry-orchestration",
            str(retry_path),
            *(str(path) for path in capture_paths),
            *args,
        ],
        expected,
    )
