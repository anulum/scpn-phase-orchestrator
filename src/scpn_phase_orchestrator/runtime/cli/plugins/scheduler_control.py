# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin remediation scheduler control and retry commands

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _record_hash,
    _require_sha256,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group,
)


@plugins_group.command("lifecycle-remediation-scheduler-control-plan")
@click.argument(
    "scheduler_execution_dashboard_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator component creating interactive control plan artifact.",
)
def plugins_lifecycle_remediation_scheduler_control_plan(
    scheduler_execution_dashboard_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic interactive control actions from scheduler dashboard.

    Parameters
    ----------
    scheduler_execution_dashboard_json : Path
        Path to the scheduler execution dashboard JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler control plan schema mismatch: "
            "created_by must be non-empty"
        )
    dashboard = _load_json_file(
        scheduler_execution_dashboard_json,
        artifact="remediation scheduler execution dashboard",
    )
    if dashboard.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_execution_dashboard_v1"
    ):
        raise click.ClickException(
            "remediation scheduler control plan schema mismatch: "
            "unexpected scheduler execution dashboard schema"
        )
    dashboard_hash = _require_sha256(dashboard.get("dashboard_hash"), "dashboard_hash")
    rows = dashboard.get("rows")
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation scheduler control plan schema mismatch: rows must be a list"
        )
    control_actions: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException(
                "remediation scheduler control plan schema mismatch: row must be object"
            )
        action_hash = _require_sha256(row.get("action_hash"), "action_hash")
        effective_state = row.get("effective_state")
        if effective_state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation scheduler control plan schema mismatch: "
                "unsupported effective_state"
            )
        overdue = bool(row.get("overdue", False))
        if effective_state == "completed":
            action = "no_op"
            reason = "already_completed"
        elif effective_state == "blocked":
            action = "escalate"
            reason = "blocked_requires_operator_intervention"
        elif overdue:
            action = "expedite"
            reason = "overdue_action_requires_priority_bump"
        elif effective_state == "in_progress":
            action = "monitor"
            reason = "execution_in_progress_track_progress"
        else:
            action = "dispatch"
            reason = "ready_for_dispatch"
        control_row: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": _require_sha256(row.get("request_hash"), "request_hash"),
            "action_type": row.get("action_type"),
            "priority": row.get("priority"),
            "effective_state": effective_state,
            "overdue": overdue,
            "control_action": action,
            "reason": reason,
            "operator_command_template": (
                "spo plugins lifecycle-remediation-action-status PLAN_JSON ACTION_HASH "
                "--state STATE --updated-by OPERATOR --note NOTE"
            ),
        }
        control_row["control_row_hash"] = _record_hash(control_row)
        control_actions.append(control_row)
    control_counts: dict[str, int] = {
        "dispatch": 0,
        "monitor": 0,
        "expedite": 0,
        "escalate": 0,
        "no_op": 0,
    }
    for item in control_actions:
        control_counts[cast(str, item["control_action"])] += 1
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_control_plan_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(dashboard.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            dashboard.get("execution_hash"), "execution_hash"
        ),
        "dashboard_hash": dashboard_hash,
        "control_action_count": len(control_actions),
        "control_counts": control_counts,
        "control_actions": sorted(
            control_actions,
            key=lambda item: (
                cast(int, item["priority"]),
                cast(str, item["control_action"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    payload["control_plan_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-runbook")
@click.argument(
    "scheduler_control_plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "scheduler_adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator component creating scheduler runbook artifact.",
)
def plugins_lifecycle_remediation_scheduler_runbook(
    scheduler_control_plan_json: Path,
    scheduler_adapter_handoff_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic operator runbook grouped by control action and adapter.

    Parameters
    ----------
    scheduler_control_plan_json : Path
        Path to the scheduler control plan JSON file.
    scheduler_adapter_handoff_json : Path
        Path to the scheduler adapter handoff JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: "
            "created_by must be non-empty"
        )
    control_plan = _load_json_file(
        scheduler_control_plan_json,
        artifact="remediation scheduler control plan",
    )
    if control_plan.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_control_plan_v1"
    ):
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: "
            "unexpected scheduler control plan schema"
        )
    _require_sha256(control_plan.get("control_plan_hash"), "control_plan_hash")
    adapter_handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            scheduler_adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    plan_hash = _require_sha256(control_plan.get("plan_hash"), "plan_hash")
    adapter_plan_hash = _require_sha256(adapter_handoff.get("plan_hash"), "plan_hash")
    if plan_hash != adapter_plan_hash:
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: plan_hash mismatch"
        )
    control_actions = control_plan.get("control_actions")
    if not isinstance(control_actions, list):
        raise click.ClickException(
            "remediation scheduler runbook schema mismatch: "
            "control_actions must be a list"
        )
    adapter_entries = cast(list[dict[str, object]], adapter_handoff["entries"])
    adapter_by_action_hash: dict[str, dict[str, object]] = {}
    for entry in adapter_entries:
        action_hash = _require_sha256(entry.get("action_hash"), "action_hash")
        if action_hash in adapter_by_action_hash:
            raise click.ClickException(
                "remediation scheduler runbook schema mismatch: "
                "duplicate action_hash in adapter handoff"
            )
        adapter_by_action_hash[action_hash] = entry
    groups: dict[str, list[dict[str, object]]] = {
        "dispatch": [],
        "monitor": [],
        "expedite": [],
        "escalate": [],
        "no_op": [],
    }
    for action in control_actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation scheduler runbook schema mismatch: control action must be "
                "object"
            )
        action_hash = _require_sha256(action.get("action_hash"), "action_hash")
        control_action = action.get("control_action")
        if control_action not in groups:
            raise click.ClickException(
                "remediation scheduler runbook schema mismatch: unsupported "
                "control_action"
            )
        adapter_entry = adapter_by_action_hash.get(action_hash)
        runbook_step: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": _require_sha256(action.get("request_hash"), "request_hash"),
            "control_action": control_action,
            "reason": action.get("reason"),
            "priority": action.get("priority"),
            "action_type": action.get("action_type"),
            "adapter_entry_hash": (
                adapter_entry.get("adapter_entry_hash")
                if adapter_entry is not None
                else None
            ),
            "adapter_name": (
                cast(dict[str, object], adapter_entry["adapter_target"]).get(
                    "adapter_name"
                )
                if adapter_entry is not None
                else None
            ),
            "adapter_endpoint": (
                cast(dict[str, object], adapter_entry["adapter_target"]).get(
                    "adapter_endpoint"
                )
                if adapter_entry is not None
                else None
            ),
            "acknowledgement_command_template": (
                adapter_entry.get("acknowledgement_command_template")
                if adapter_entry is not None
                else None
            ),
        }
        runbook_step["runbook_step_hash"] = _record_hash(runbook_step)
        groups[cast(str, control_action)].append(runbook_step)
    ordered_groups: list[dict[str, object]] = []
    for name in ("escalate", "expedite", "dispatch", "monitor", "no_op"):
        items = sorted(
            groups[name],
            key=lambda item: (
                cast(int, item["priority"]),
                cast(str, item["action_hash"]),
            ),
        )
        ordered_groups.append(
            {
                "control_action": name,
                "step_count": len(items),
                "steps": items,
            }
        )
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_runbook_v1"
        ),
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "execution_hash": _require_sha256(
            control_plan.get("execution_hash"), "execution_hash"
        ),
        "control_plan_hash": _require_sha256(
            control_plan.get("control_plan_hash"),
            "control_plan_hash",
        ),
        "adapter_handoff_hash": _require_sha256(
            adapter_handoff.get("adapter_handoff_hash"),
            "adapter_handoff_hash",
        ),
        "group_count": len(ordered_groups),
        "groups": ordered_groups,
        "created_by": created_by,
    }
    payload["runbook_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-automation-profile")
@click.argument(
    "scheduler_runbook_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--profile-name",
    required=True,
    help="Automation profile name.",
)
@click.option(
    "--profile-version",
    required=True,
    help="Automation profile semantic version.",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating automation profile artifact.",
)
def plugins_lifecycle_remediation_scheduler_automation_profile(
    scheduler_runbook_json: Path,
    profile_name: str,
    profile_version: str,
    created_by: str,
) -> None:
    """Emit deterministic adapter automation profile from scheduler runbook.

    Parameters
    ----------
    scheduler_runbook_json : Path
        Path to the scheduler runbook JSON file.
    profile_name : str
        Name of the automation profile.
    profile_version : str
        Version of the automation profile.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "created_by must be non-empty"
        )
    if not profile_name:
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "profile_name must be non-empty"
        )
    if not profile_version:
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "profile_version must be non-empty"
        )
    runbook = _load_json_file(
        scheduler_runbook_json,
        artifact="remediation scheduler runbook",
    )
    if (
        runbook.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_runbook_v1"
    ):
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "unexpected scheduler runbook schema"
        )
    _require_sha256(runbook.get("runbook_hash"), "runbook_hash")
    groups = runbook.get("groups")
    if not isinstance(groups, list):
        raise click.ClickException(
            "remediation scheduler automation profile schema mismatch: "
            "groups must be a list"
        )
    automation_rules: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise click.ClickException(
                "remediation scheduler automation profile schema mismatch: "
                "group must be object"
            )
        control_action = group.get("control_action")
        if control_action not in {
            "dispatch",
            "monitor",
            "expedite",
            "escalate",
            "no_op",
        }:
            raise click.ClickException(
                "remediation scheduler automation profile schema mismatch: "
                "unsupported control_action"
            )
        steps = group.get("steps")
        if not isinstance(steps, list):
            raise click.ClickException(
                "remediation scheduler automation profile schema mismatch: "
                "steps must be list"
            )
        for step in steps:
            if not isinstance(step, dict):
                raise click.ClickException(
                    "remediation scheduler automation profile schema mismatch: "
                    "step must be object"
                )
            action_hash = _require_sha256(step.get("action_hash"), "action_hash")
            request_hash = _require_sha256(step.get("request_hash"), "request_hash")
            action_type = step.get("action_type")
            priority = step.get("priority")
            if not isinstance(action_type, str) or not action_type:
                raise click.ClickException(
                    "remediation scheduler automation profile schema mismatch: "
                    "action_type must be non-empty string"
                )
            if not isinstance(priority, int) or priority < 1:
                raise click.ClickException(
                    "remediation scheduler automation profile schema mismatch: "
                    "priority must be positive integer"
                )
            automation_mode = (
                "manual" if control_action in {"escalate", "no_op"} else "auto"
            )
            target_state = {
                "dispatch": "in_progress",
                "monitor": "in_progress",
                "expedite": "in_progress",
                "escalate": "blocked",
                "no_op": "completed",
            }[cast(str, control_action)]
            rule: dict[str, object] = {
                "control_action": control_action,
                "action_hash": action_hash,
                "request_hash": request_hash,
                "action_type": action_type,
                "priority": priority,
                "automation_mode": automation_mode,
                "target_state": target_state,
                "capture_command_template": (
                    "spo plugins "
                    "lifecycle-remediation-scheduler-acknowledgement-capture "
                    "AUTOMATION_PROFILE_JSON ADAPTER_HANDOFF_JSON ACTION_HASH "
                    "--external-reference REF --acknowledged-by OPERATOR "
                    "--captured-state STATE --note NOTE"
                ),
            }
            rule["automation_rule_hash"] = _record_hash(rule)
            automation_rules.append(rule)
    profile_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_automation_profile_v1"
        ),
        "version": "1.0.0",
        "profile_name": profile_name,
        "profile_version": profile_version,
        "plan_hash": _require_sha256(runbook.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            runbook.get("execution_hash"), "execution_hash"
        ),
        "runbook_hash": _require_sha256(runbook.get("runbook_hash"), "runbook_hash"),
        "automation_rule_count": len(automation_rules),
        "automation_rules": sorted(
            automation_rules,
            key=lambda item: (
                cast(int, item["priority"]),
                cast(str, item["control_action"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    profile_payload["automation_profile_hash"] = _record_hash(profile_payload)
    click.echo(json.dumps(profile_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-acknowledgement-capture")
@click.argument(
    "automation_profile_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("action_hash")
@click.option(
    "--external-reference",
    required=True,
    help="External scheduler run identifier.",
)
@click.option(
    "--acknowledged-by",
    required=True,
    help="Operator or adapter acknowledging execution.",
)
@click.option(
    "--captured-state",
    required=True,
    type=click.Choice(["in_progress", "completed", "blocked"]),
    help="Captured execution state.",
)
@click.option(
    "--note",
    default="",
    show_default=True,
    help="Optional capture note.",
)
def plugins_lifecycle_remediation_scheduler_acknowledgement_capture(
    automation_profile_json: Path,
    adapter_handoff_json: Path,
    action_hash: str,
    external_reference: str,
    acknowledged_by: str,
    captured_state: str,
    note: str,
) -> None:
    """Capture acknowledgement using automation profile and adapter handoff.

    Parameters
    ----------
    automation_profile_json : Path
        Path to the automation profile JSON file.
    adapter_handoff_json : Path
        Path to the adapter handoff JSON file.
    action_hash : str
        Hash of the remediation action.
    external_reference : str
        External system reference.
    acknowledged_by : str
        Identifier of the acknowledged actor.
    captured_state : str
        Captured acknowledgement state.
    note : str
        Free-form note recorded with the record.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not external_reference:
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "external_reference must be non-empty"
        )
    if not acknowledged_by:
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "acknowledged_by must be non-empty"
        )
    profile = _load_json_file(
        automation_profile_json,
        artifact="remediation scheduler automation profile",
    )
    if profile.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "unexpected automation profile schema"
        )
    _require_sha256(profile.get("automation_profile_hash"), "automation_profile_hash")
    normalized_action_hash = _require_sha256(action_hash, "action_hash")
    rules = profile.get("automation_rules")
    if not isinstance(rules, list):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "automation_rules must be list"
        )
    rule = next(
        (
            item
            for item in rules
            if isinstance(item, dict)
            and item.get("action_hash") == normalized_action_hash
        ),
        None,
    )
    if not isinstance(rule, dict):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "action_hash not present in automation profile"
        )
    target_state = cast(str, rule.get("target_state"))
    if (
        target_state != captured_state
        and cast(str, rule.get("automation_mode")) == "auto"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "captured_state does not match auto target_state"
        )
    adapter_handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    if _require_sha256(profile.get("plan_hash"), "plan_hash") != _require_sha256(
        adapter_handoff.get("plan_hash"), "plan_hash"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "plan_hash mismatch between automation profile and adapter handoff"
        )
    entries = cast(list[dict[str, object]], adapter_handoff["entries"])
    matched_entry = next(
        (
            entry
            for entry in entries
            if _require_sha256(entry.get("action_hash"), "action_hash")
            == normalized_action_hash
        ),
        None,
    )
    if matched_entry is None:
        raise click.ClickException(
            "remediation scheduler acknowledgement capture schema mismatch: "
            "action_hash not present in adapter handoff"
        )
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_acknowledgement_capture_v1"
        ),
        "version": "1.0.0",
        "automation_profile_hash": _require_sha256(
            profile.get("automation_profile_hash"),
            "automation_profile_hash",
        ),
        "adapter_handoff_hash": _require_sha256(
            adapter_handoff.get("adapter_handoff_hash"),
            "adapter_handoff_hash",
        ),
        "plan_hash": _require_sha256(profile.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            profile.get("execution_hash"), "execution_hash"
        ),
        "action_hash": normalized_action_hash,
        "request_hash": _require_sha256(rule.get("request_hash"), "request_hash"),
        "adapter_entry_hash": _require_sha256(
            matched_entry.get("adapter_entry_hash"),
            "adapter_entry_hash",
        ),
        "captured_state": captured_state,
        "target_state": target_state,
        "automation_mode": rule.get("automation_mode"),
        "external_reference": external_reference,
        "acknowledged_by": acknowledged_by,
        "note": note,
    }
    payload["capture_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-retry-profile")
@click.argument(
    "automation_profile_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--max-attempts",
    default=3,
    show_default=True,
    type=int,
    help="Maximum retry attempts for eligible automated actions.",
)
@click.option(
    "--base-delay-seconds",
    default=30,
    show_default=True,
    type=int,
    help="Base delay in seconds before first retry.",
)
@click.option(
    "--backoff-multiplier",
    default=2.0,
    show_default=True,
    type=float,
    help="Retry backoff multiplier.",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating retry profile artifact.",
)
def plugins_lifecycle_remediation_scheduler_retry_profile(
    automation_profile_json: Path,
    max_attempts: int,
    base_delay_seconds: int,
    backoff_multiplier: float,
    created_by: str,
) -> None:
    """Emit deterministic retry/backoff policy profile from automation profile.

    Parameters
    ----------
    automation_profile_json : Path
        Path to the automation profile JSON file.
    max_attempts : int
        Maximum number of retry attempts.
    base_delay_seconds : int
        Base retry delay in seconds.
    backoff_multiplier : float
        Exponential backoff multiplier.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "created_by must be non-empty"
        )
    if max_attempts < 1:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "max_attempts must be positive"
        )
    if base_delay_seconds < 1:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "base_delay_seconds must be positive"
        )
    if backoff_multiplier < 1.0:
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "backoff_multiplier must be >= 1.0"
        )
    profile = _load_json_file(
        automation_profile_json,
        artifact="remediation scheduler automation profile",
    )
    if profile.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
    ):
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "unexpected automation profile schema"
        )
    automation_profile_hash = _require_sha256(
        profile.get("automation_profile_hash"),
        "automation_profile_hash",
    )
    rules = profile.get("automation_rules")
    if not isinstance(rules, list):
        raise click.ClickException(
            "remediation scheduler retry profile schema mismatch: "
            "automation_rules must be list"
        )
    retry_rules: list[dict[str, object]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            raise click.ClickException(
                "remediation scheduler retry profile schema mismatch: "
                "rule must be object"
            )
        action_hash = _require_sha256(rule.get("action_hash"), "action_hash")
        request_hash = _require_sha256(rule.get("request_hash"), "request_hash")
        automation_mode = rule.get("automation_mode")
        control_action = rule.get("control_action")
        if automation_mode not in {"auto", "manual"}:
            raise click.ClickException(
                "remediation scheduler retry profile schema mismatch: unsupported "
                "automation_mode"
            )
        if control_action not in {
            "dispatch",
            "monitor",
            "expedite",
            "escalate",
            "no_op",
        }:
            raise click.ClickException(
                "remediation scheduler retry profile schema mismatch: unsupported "
                "control_action"
            )
        policy_mode = (
            "retry_enabled"
            if automation_mode == "auto" and control_action in {"dispatch", "expedite"}
            else "retry_disabled"
        )
        retry_rule: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": request_hash,
            "automation_mode": automation_mode,
            "control_action": control_action,
            "target_state": rule.get("target_state"),
            "policy_mode": policy_mode,
            "max_attempts": max_attempts if policy_mode == "retry_enabled" else 0,
            "base_delay_seconds": (
                base_delay_seconds if policy_mode == "retry_enabled" else 0
            ),
            "backoff_multiplier": (
                backoff_multiplier if policy_mode == "retry_enabled" else 1.0
            ),
        }
        retry_rule["retry_rule_hash"] = _record_hash(retry_rule)
        retry_rules.append(retry_rule)
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_retry_profile_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(profile.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            profile.get("execution_hash"), "execution_hash"
        ),
        "automation_profile_hash": automation_profile_hash,
        "retry_rule_count": len(retry_rules),
        "retry_rules": sorted(
            retry_rules,
            key=lambda item: (
                cast(str, item["policy_mode"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    payload["retry_profile_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-scheduler-retry-orchestration")
@click.argument(
    "retry_profile_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "acknowledgement_capture_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating retry orchestration artifact.",
)
def plugins_lifecycle_remediation_scheduler_retry_orchestration(
    retry_profile_json: Path,
    acknowledgement_capture_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit deterministic retry queue from captured acknowledgements and policy.

    Parameters
    ----------
    retry_profile_json : Path
        Path to the retry profile JSON file.
    acknowledgement_capture_json : tuple[Path, ...]
        Path to the acknowledgement capture JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler retry orchestration schema mismatch: "
            "created_by must be non-empty"
        )
    retry_profile = _load_json_file(
        retry_profile_json,
        artifact="remediation scheduler retry profile",
    )
    if retry_profile.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_retry_profile_v1"
    ):
        raise click.ClickException(
            "remediation scheduler retry orchestration schema mismatch: "
            "unexpected retry profile schema"
        )
    retry_profile_hash = _require_sha256(
        retry_profile.get("retry_profile_hash"),
        "retry_profile_hash",
    )
    retry_rules = retry_profile.get("retry_rules")
    if not isinstance(retry_rules, list):
        raise click.ClickException(
            "remediation scheduler retry orchestration schema mismatch: "
            "retry_rules must be list"
        )
    retry_rule_by_action_hash: dict[str, dict[str, object]] = {}
    for rule in retry_rules:
        if not isinstance(rule, dict):
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: "
                "rule must be object"
            )
        action_hash = _require_sha256(rule.get("action_hash"), "action_hash")
        if action_hash in retry_rule_by_action_hash:
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: duplicate "
                "rule action_hash"
            )
        retry_rule_by_action_hash[action_hash] = rule

    capture_by_action_hash: dict[str, dict[str, object]] = {}
    for path in acknowledgement_capture_json:
        capture = _load_json_file(
            path,
            artifact="remediation scheduler acknowledgement capture",
        )
        if capture.get("schema") != (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_capture_v1"
        ):
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: "
                "unexpected acknowledgement capture schema"
            )
        action_hash = _require_sha256(capture.get("action_hash"), "action_hash")
        if action_hash in capture_by_action_hash:
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: duplicate "
                "capture action_hash"
            )
        if _require_sha256(capture.get("plan_hash"), "plan_hash") != _require_sha256(
            retry_profile.get("plan_hash"), "plan_hash"
        ):
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: plan_hash "
                "mismatch"
            )
        capture_by_action_hash[action_hash] = capture

    retry_entries: list[dict[str, object]] = []
    for action_hash, capture in sorted(capture_by_action_hash.items()):
        rule = retry_rule_by_action_hash.get(action_hash)
        if rule is None:
            raise click.ClickException(
                "remediation scheduler retry orchestration schema mismatch: "
                "capture action_hash missing from retry profile"
            )
        state = capture.get("captured_state")
        if state == "completed":
            continue
        if cast(str, rule["policy_mode"]) != "retry_enabled":
            continue
        max_attempts = cast(int, rule["max_attempts"])
        base_delay_seconds = cast(int, rule["base_delay_seconds"])
        backoff_multiplier = cast(float, rule["backoff_multiplier"])
        attempt = 1
        next_delay_seconds = int(
            base_delay_seconds * (backoff_multiplier ** (attempt - 1))
        )
        entry: dict[str, object] = {
            "action_hash": action_hash,
            "request_hash": _require_sha256(
                capture.get("request_hash"), "request_hash"
            ),
            "capture_hash": _require_sha256(
                capture.get("capture_hash"), "capture_hash"
            ),
            "capture_state": state,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "next_delay_seconds": next_delay_seconds,
            "external_reference": capture.get("external_reference"),
            "retry_command_template": (
                "spo plugins lifecycle-remediation-scheduler-acknowledgement-capture "
                "AUTOMATION_PROFILE_JSON ADAPTER_HANDOFF_JSON ACTION_HASH "
                "--external-reference REF --acknowledged-by OPERATOR "
                "--captured-state STATE --note NOTE"
            ),
        }
        entry["retry_entry_hash"] = _record_hash(entry)
        retry_entries.append(entry)
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_retry_orchestration_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(retry_profile.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            retry_profile.get("execution_hash"), "execution_hash"
        ),
        "retry_profile_hash": retry_profile_hash,
        "retry_entry_count": len(retry_entries),
        "retry_entries": sorted(
            retry_entries,
            key=lambda item: (
                cast(int, item["next_delay_seconds"]),
                cast(str, item["action_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    payload["retry_orchestration_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
