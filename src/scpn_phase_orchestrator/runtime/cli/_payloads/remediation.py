# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI remediation payload loaders

"""Remediation plan, action, dashboard, and deployment hand-off loaders."""

from __future__ import annotations

import click

from ._shared import _require_sha256


def _load_lifecycle_remediation_plan_payload(
    plan_payload: dict[str, object],
) -> dict[str, object]:
    """Load a validated remediation plan from a payload, else raise."""
    if (
        plan_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
    ):
        raise click.ClickException(
            "remediation plan schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
        )
    _require_sha256(plan_payload.get("plan_hash"), "plan_hash")
    _require_sha256(plan_payload.get("drilldown_hash"), "drilldown_hash")
    action_count = plan_payload.get("action_count")
    actions = plan_payload.get("actions")
    if not isinstance(action_count, int) or action_count < 0:
        raise click.ClickException(
            "remediation plan schema mismatch: action_count must be non-negative"
        )
    if not isinstance(actions, list):
        raise click.ClickException(
            "remediation plan schema mismatch: actions must be a list"
        )
    if action_count != len(actions):
        raise click.ClickException(
            "remediation plan schema mismatch: action_count does not match actions"
        )
    for action in actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation plan schema mismatch: action must be an object"
            )
        _require_sha256(action.get("action_hash"), "action_hash")
        _require_sha256(action.get("request_hash"), "request_hash")
        _require_sha256(action.get("store_hash"), "store_hash")
        _require_sha256(action.get("policy_hash"), "policy_hash")
        _require_sha256(action.get("summary_hash"), "summary_hash")
        action_type = action.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation plan schema mismatch: unsupported action_type"
            )
        priority = action.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation plan schema mismatch: priority must be a positive integer"
            )
    return plan_payload


def _load_lifecycle_remediation_action_status_payload(
    status_payload: dict[str, object],
) -> dict[str, object]:
    """Load a validated remediation action-status from a payload, else raise."""
    if (
        status_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
    ):
        raise click.ClickException(
            "remediation action status schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
        )
    _require_sha256(status_payload.get("status_hash"), "status_hash")
    _require_sha256(status_payload.get("action_hash"), "action_hash")
    _require_sha256(status_payload.get("plan_hash"), "plan_hash")
    state = status_payload.get("state")
    if state not in {"pending", "in_progress", "completed", "blocked"}:
        raise click.ClickException(
            "remediation action status schema mismatch: unsupported state"
        )
    return status_payload


def _load_lifecycle_remediation_execution_dashboard_payload(
    dashboard_payload: dict[str, object],
) -> dict[str, object]:
    """Load a validated remediation execution dashboard from a payload, else raise."""
    if (
        dashboard_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
    ):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
        )
    _require_sha256(dashboard_payload.get("execution_hash"), "execution_hash")
    _require_sha256(dashboard_payload.get("plan_hash"), "plan_hash")
    action_count = dashboard_payload.get("action_count")
    rows = dashboard_payload.get("rows")
    if not isinstance(action_count, int) or action_count < 0:
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: "
            "action_count must be non-negative"
        )
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: rows must be a list"
        )
    if action_count != len(rows):
        raise click.ClickException(
            "remediation execution dashboard schema mismatch: "
            "action_count does not match rows"
        )
    for row in rows:
        if not isinstance(row, dict):
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: row must be object"
            )
        _require_sha256(row.get("action_hash"), "action_hash")
        _require_sha256(row.get("request_hash"), "request_hash")
        state = row.get("state")
        if state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: unsupported state"
            )
        action_type = row.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: "
                "unsupported action_type"
            )
        priority = row.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation execution dashboard schema mismatch: "
                "priority must be a positive integer"
            )
    return dashboard_payload


def _load_lifecycle_remediation_deployment_handoff_payload(
    handoff_payload: dict[str, object],
) -> dict[str, object]:
    """Load a validated remediation deployment handoff from a payload, else raise."""
    if (
        handoff_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
    ):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
        )
    _require_sha256(handoff_payload.get("handoff_hash"), "handoff_hash")
    _require_sha256(handoff_payload.get("plan_hash"), "plan_hash")
    _require_sha256(handoff_payload.get("execution_hash"), "execution_hash")
    unresolved_count = handoff_payload.get("unresolved_action_count")
    handoff_actions = handoff_payload.get("handoff_actions")
    if not isinstance(unresolved_count, int) or unresolved_count < 0:
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "unresolved_action_count must be non-negative"
        )
    if not isinstance(handoff_actions, list):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "handoff_actions must be a list"
        )
    if unresolved_count != len(handoff_actions):
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "unresolved_action_count does not match handoff_actions"
        )
    for action in handoff_actions:
        if not isinstance(action, dict):
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: action must be object"
            )
        _require_sha256(action.get("handoff_action_hash"), "handoff_action_hash")
        _require_sha256(action.get("action_hash"), "action_hash")
        _require_sha256(action.get("request_hash"), "request_hash")
        action_type = action.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "unsupported action_type"
            )
        priority = action.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "priority must be a positive integer"
            )
        template = action.get("deployment_command_template")
        if not isinstance(template, str) or not template:
            raise click.ClickException(
                "remediation deployment handoff schema mismatch: "
                "deployment_command_template must be non-empty"
            )
    return handoff_payload
