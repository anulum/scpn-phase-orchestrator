# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scheduler control command contracts

"""Shared fixtures for scheduler-control public CLI contract tests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from click.testing import CliRunner, Result

from scpn_phase_orchestrator.runtime.cli.plugins import scheduler_control
from scpn_phase_orchestrator.runtime.cli.plugins._group import plugins_group

_REGISTERED_MODULE = scheduler_control
_PREFIX = "scpn_plugin_execution_request_lifecycle_remediation_scheduler"
_PLAN_HASH = "1" * 64
_EXECUTION_HASH = "2" * 64
_DASHBOARD_HASH = "3" * 64
_RUNBOOK_HASH = "4" * 64
_AUTOMATION_PROFILE_HASH = "5" * 64
_ADAPTER_HANDOFF_HASH = "6" * 64
_TELEMETRY_HASH = "7" * 64


def _hex(value: str) -> str:
    """Return a deterministic valid SHA-256 fixture hash for ``value``."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _record_hash(payload: dict[str, Any]) -> str:
    """Return the command-compatible deterministic record hash for ``payload``."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _with_hash(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Return ``payload`` with ``field_name`` populated from its current content."""
    hashed = dict(payload)
    hashed[field_name] = _record_hash(hashed)
    return hashed


def _write_json(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    """Write ``payload`` as stable JSON under ``tmp_path`` and return the path."""
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _invoke(args: Sequence[str]) -> Result:
    """Invoke the plugins Click group with ``args``."""
    return CliRunner().invoke(plugins_group, list(args))


def _invoke_json(args: Sequence[str]) -> dict[str, Any]:
    """Invoke a plugins subcommand and return its JSON output."""
    result = _invoke(args)
    assert result.exit_code == 0, result.output
    return cast(dict[str, Any], json.loads(result.output))


def _assert_fails(args: Sequence[str], expected: str) -> None:
    """Assert that a plugins subcommand rejects its inputs with ``expected`` text."""
    result = _invoke(args)
    assert result.exit_code == 1
    assert expected in result.output


def _dashboard_row(
    *,
    marker: str,
    effective_state: str,
    overdue: bool = False,
    priority: int = 1,
) -> dict[str, Any]:
    """Build a scheduler execution dashboard row."""
    return {
        "action_hash": _hex(marker),
        "request_hash": _hex(chr(ord(marker) + 1)),
        "action_type": "persist_request",
        "priority": priority,
        "effective_state": effective_state,
        "overdue": overdue,
    }


def _dashboard_payload(**overrides: Any) -> dict[str, Any]:
    """Build a scheduler execution dashboard artifact."""
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_execution_dashboard_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "dashboard_hash": _DASHBOARD_HASH,
        "rows": [
            _dashboard_row(marker="a", effective_state="pending", priority=3),
            _dashboard_row(marker="c", effective_state="in_progress", priority=4),
            _dashboard_row(
                marker="e",
                effective_state="pending",
                overdue=True,
                priority=1,
            ),
            _dashboard_row(marker="g", effective_state="blocked", priority=2),
            _dashboard_row(marker="i", effective_state="completed", priority=5),
        ],
    }
    payload.update(overrides)
    return payload


def _control_action(
    *,
    action_hash: str,
    request_hash: str,
    control_action: str,
    priority: int = 1,
) -> dict[str, Any]:
    """Build a scheduler control-plan action."""
    return {
        "action_hash": action_hash,
        "request_hash": request_hash,
        "action_type": "persist_request",
        "priority": priority,
        "effective_state": "pending",
        "overdue": False,
        "control_action": control_action,
        "reason": f"{control_action}_reason",
    }


def _control_plan_payload(
    *,
    actions: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a scheduler control-plan artifact."""
    control_actions = actions if actions is not None else _base_control_actions()
    counts = dict.fromkeys(("dispatch", "monitor", "expedite", "escalate", "no_op"), 0)
    for action in control_actions:
        if isinstance(action, dict):
            control_action = action.get("control_action")
            if isinstance(control_action, str) and control_action in counts:
                counts[control_action] += 1
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_control_plan_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "dashboard_hash": _DASHBOARD_HASH,
        "control_action_count": len(control_actions),
        "control_counts": counts,
        "control_actions": control_actions,
        "created_by": "operator_console",
    }
    payload.update(overrides)
    return _with_hash(payload, "control_plan_hash")


def _base_control_actions() -> list[dict[str, Any]]:
    """Build one control action for every scheduler-control action type."""
    return [
        _control_action(
            action_hash=_hex("a"),
            request_hash=_hex("b"),
            control_action="dispatch",
            priority=3,
        ),
        _control_action(
            action_hash=_hex("c"),
            request_hash=_hex("d"),
            control_action="monitor",
            priority=4,
        ),
        _control_action(
            action_hash=_hex("e"),
            request_hash=_hex("f"),
            control_action="expedite",
            priority=1,
        ),
        _control_action(
            action_hash=_hex("g"),
            request_hash=_hex("h"),
            control_action="escalate",
            priority=2,
        ),
        _control_action(
            action_hash=_hex("i"),
            request_hash=_hex("j"),
            control_action="no_op",
            priority=5,
        ),
    ]


def _adapter_entry(
    *,
    action_hash: str,
    request_hash: str,
    marker: str,
) -> dict[str, Any]:
    """Build a scheduler adapter-handoff entry."""
    return {
        "adapter_entry_hash": _hex(marker),
        "entry_hash": _hex(chr(ord(marker) + 1)),
        "action_hash": action_hash,
        "request_hash": request_hash,
        "adapter_target": {
            "adapter_name": "airflow",
            "adapter_endpoint": "airflow://cluster-a",
        },
        "acknowledgement_command_template": "ack --action ACTION_HASH",
    }


def _adapter_handoff_payload(
    *,
    entries: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a scheduler adapter-handoff artifact."""
    adapter_entries = (
        entries
        if entries is not None
        else [
            _adapter_entry(
                action_hash=_hex("a"),
                request_hash=_hex("b"),
                marker="k",
            ),
            _adapter_entry(
                action_hash=_hex("e"),
                request_hash=_hex("f"),
                marker="m",
            ),
        ]
    )
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_adapter_handoff_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "telemetry_hash": _TELEMETRY_HASH,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "entry_count": len(adapter_entries),
        "entries": adapter_entries,
        "created_by": "deployment_scheduler",
    }
    payload.update(overrides)
    return _with_hash(payload, "adapter_handoff_hash")


def _runbook_step(
    *,
    action_hash: str,
    request_hash: str,
    control_action: str,
    priority: int,
) -> dict[str, Any]:
    """Build a scheduler runbook step."""
    step: dict[str, Any] = {
        "action_hash": action_hash,
        "request_hash": request_hash,
        "control_action": control_action,
        "reason": f"{control_action}_reason",
        "priority": priority,
        "action_type": "persist_request",
        "adapter_entry_hash": _ADAPTER_HANDOFF_HASH,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "acknowledgement_command_template": "ack --action ACTION_HASH",
    }
    step["runbook_step_hash"] = _record_hash(step)
    return step


def _runbook_group(
    control_action: str,
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a scheduler runbook group."""
    return {
        "control_action": control_action,
        "step_count": len(steps),
        "steps": steps,
    }


def _runbook_payload(
    *,
    groups: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a scheduler runbook artifact."""
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_runbook_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "control_plan_hash": _hex("p"),
        "adapter_handoff_hash": _ADAPTER_HANDOFF_HASH,
        "group_count": 5,
        "groups": groups if groups is not None else _base_runbook_groups(),
        "created_by": "operator_console",
    }
    payload["group_count"] = len(cast(list[dict[str, Any]], payload["groups"]))
    payload.update(overrides)
    return _with_hash(payload, "runbook_hash")


def _base_runbook_groups() -> list[dict[str, Any]]:
    """Build runbook groups covering all scheduler-control actions."""
    return [
        _runbook_group(
            "escalate",
            [
                _runbook_step(
                    action_hash=_hex("g"),
                    request_hash=_hex("h"),
                    control_action="escalate",
                    priority=2,
                )
            ],
        ),
        _runbook_group(
            "expedite",
            [
                _runbook_step(
                    action_hash=_hex("e"),
                    request_hash=_hex("f"),
                    control_action="expedite",
                    priority=1,
                )
            ],
        ),
        _runbook_group(
            "dispatch",
            [
                _runbook_step(
                    action_hash=_hex("a"),
                    request_hash=_hex("b"),
                    control_action="dispatch",
                    priority=3,
                )
            ],
        ),
        _runbook_group(
            "monitor",
            [
                _runbook_step(
                    action_hash=_hex("c"),
                    request_hash=_hex("d"),
                    control_action="monitor",
                    priority=4,
                )
            ],
        ),
        _runbook_group(
            "no_op",
            [
                _runbook_step(
                    action_hash=_hex("i"),
                    request_hash=_hex("j"),
                    control_action="no_op",
                    priority=5,
                )
            ],
        ),
    ]


def _automation_rule(
    *,
    action_hash: str,
    request_hash: str,
    control_action: str,
    automation_mode: str,
    target_state: str,
    priority: int = 1,
) -> dict[str, Any]:
    """Build a scheduler automation-profile rule."""
    rule: dict[str, Any] = {
        "control_action": control_action,
        "action_hash": action_hash,
        "request_hash": request_hash,
        "action_type": "persist_request",
        "priority": priority,
        "automation_mode": automation_mode,
        "target_state": target_state,
        "capture_command_template": "capture --action ACTION_HASH",
    }
    rule["automation_rule_hash"] = _record_hash(rule)
    return rule


def _automation_profile_payload(
    *,
    rules: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a scheduler automation-profile artifact."""
    automation_rules = rules if rules is not None else _base_automation_rules()
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_automation_profile_v1",
        "version": "1.0.0",
        "profile_name": "airflow-default",
        "profile_version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "runbook_hash": _RUNBOOK_HASH,
        "automation_rule_count": len(automation_rules),
        "automation_rules": automation_rules,
        "created_by": "operator_console",
    }
    payload.update(overrides)
    return _with_hash(payload, "automation_profile_hash")


def _base_automation_rules() -> list[dict[str, Any]]:
    """Build automation rules covering auto, manual, enabled, and disabled paths."""
    return [
        _automation_rule(
            action_hash=_hex("a"),
            request_hash=_hex("b"),
            control_action="dispatch",
            automation_mode="auto",
            target_state="in_progress",
            priority=1,
        ),
        _automation_rule(
            action_hash=_hex("e"),
            request_hash=_hex("f"),
            control_action="expedite",
            automation_mode="auto",
            target_state="in_progress",
            priority=2,
        ),
        _automation_rule(
            action_hash=_hex("c"),
            request_hash=_hex("d"),
            control_action="monitor",
            automation_mode="auto",
            target_state="in_progress",
            priority=3,
        ),
        _automation_rule(
            action_hash=_hex("g"),
            request_hash=_hex("h"),
            control_action="escalate",
            automation_mode="manual",
            target_state="blocked",
            priority=4,
        ),
        _automation_rule(
            action_hash=_hex("i"),
            request_hash=_hex("j"),
            control_action="no_op",
            automation_mode="manual",
            target_state="completed",
            priority=5,
        ),
    ]


def _capture_payload(
    *,
    action_hash: str = "",
    request_hash: str = "",
    captured_state: str = "blocked",
    target_state: str = "in_progress",
    automation_mode: str = "auto",
    plan_hash: str = _PLAN_HASH,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a scheduler acknowledgement-capture artifact."""
    resolved_action_hash = action_hash or _hex("a")
    resolved_request_hash = request_hash or _hex("b")
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_acknowledgement_capture_v1",
        "version": "1.0.0",
        "automation_profile_hash": _AUTOMATION_PROFILE_HASH,
        "adapter_handoff_hash": _ADAPTER_HANDOFF_HASH,
        "plan_hash": plan_hash,
        "execution_hash": _EXECUTION_HASH,
        "action_hash": resolved_action_hash,
        "request_hash": resolved_request_hash,
        "adapter_entry_hash": _hex("k"),
        "captured_state": captured_state,
        "target_state": target_state,
        "automation_mode": automation_mode,
        "external_reference": "airflow-run-1",
        "acknowledged_by": "operator_console",
        "note": "",
    }
    payload.update(overrides)
    return _with_hash(payload, "capture_hash")


def _retry_rule(
    *,
    action_hash: str,
    request_hash: str,
    control_action: str,
    policy_mode: str,
) -> dict[str, Any]:
    """Build a scheduler retry-profile rule."""
    enabled = policy_mode == "retry_enabled"
    rule: dict[str, Any] = {
        "action_hash": action_hash,
        "request_hash": request_hash,
        "automation_mode": "auto" if enabled else "manual",
        "control_action": control_action,
        "target_state": "in_progress",
        "policy_mode": policy_mode,
        "max_attempts": 4 if enabled else 0,
        "base_delay_seconds": 20 if enabled else 0,
        "backoff_multiplier": 2.0 if enabled else 1.0,
    }
    rule["retry_rule_hash"] = _record_hash(rule)
    return rule


def _retry_profile_payload(
    *,
    rules: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a scheduler retry-profile artifact."""
    retry_rules = (
        rules
        if rules is not None
        else [
            _retry_rule(
                action_hash=_hex("a"),
                request_hash=_hex("b"),
                control_action="dispatch",
                policy_mode="retry_enabled",
            ),
            _retry_rule(
                action_hash=_hex("c"),
                request_hash=_hex("d"),
                control_action="monitor",
                policy_mode="retry_disabled",
            ),
            _retry_rule(
                action_hash=_hex("e"),
                request_hash=_hex("f"),
                control_action="expedite",
                policy_mode="retry_enabled",
            ),
        ]
    )
    payload: dict[str, Any] = {
        "schema": f"{_PREFIX}_retry_profile_v1",
        "version": "1.0.0",
        "plan_hash": _PLAN_HASH,
        "execution_hash": _EXECUTION_HASH,
        "automation_profile_hash": _AUTOMATION_PROFILE_HASH,
        "retry_rule_count": len(retry_rules),
        "retry_rules": retry_rules,
        "created_by": "operator_console",
    }
    payload.update(overrides)
    return _with_hash(payload, "retry_profile_hash")
