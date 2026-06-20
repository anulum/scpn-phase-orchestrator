# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI scheduler payload loaders

"""Scheduler queue, telemetry, adapter hand-off, and acknowledgement loaders."""

from __future__ import annotations

import click

from ._shared import _require_sha256


def _load_lifecycle_remediation_scheduler_queue_payload(
    queue_payload: dict[str, object],
) -> dict[str, object]:
    if (
        queue_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
    ):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
        )
    _require_sha256(queue_payload.get("scheduler_hash"), "scheduler_hash")
    _require_sha256(queue_payload.get("plan_hash"), "plan_hash")
    _require_sha256(queue_payload.get("execution_hash"), "execution_hash")
    _require_sha256(queue_payload.get("handoff_hash"), "handoff_hash")
    queue_entry_count = queue_payload.get("queue_entry_count")
    queue_entries = queue_payload.get("queue_entries")
    if not isinstance(queue_entry_count, int) or queue_entry_count < 0:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "queue_entry_count must be non-negative"
        )
    if not isinstance(queue_entries, list):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: queue_entries must be a list"
        )
    if queue_entry_count != len(queue_entries):
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "queue_entry_count does not match queue_entries"
        )
    for entry in queue_entries:
        if not isinstance(entry, dict):
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: entry must be object"
            )
        _require_sha256(entry.get("entry_hash"), "entry_hash")
        _require_sha256(entry.get("handoff_action_hash"), "handoff_action_hash")
        _require_sha256(entry.get("action_hash"), "action_hash")
        _require_sha256(entry.get("request_hash"), "request_hash")
        action_type = entry.get("action_type")
        if action_type not in {
            "renew_approval",
            "persist_request",
            "register_storage_adapter",
            "confirm_external_write",
        }:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: unsupported action_type"
            )
        priority = entry.get("priority")
        if not isinstance(priority, int) or priority < 1:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "priority must be a positive integer"
            )
        schedule_epoch = entry.get("schedule_epoch")
        if not isinstance(schedule_epoch, int) or schedule_epoch < 0:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "schedule_epoch must be non-negative integer"
            )
        template = entry.get("scheduler_command_template")
        if not isinstance(template, str) or not template:
            raise click.ClickException(
                "remediation scheduler queue schema mismatch: "
                "scheduler_command_template must be non-empty"
            )
    return queue_payload


def _load_lifecycle_remediation_scheduler_telemetry_payload(
    telemetry_payload: dict[str, object],
) -> dict[str, object]:
    if (
        telemetry_payload.get("schema")
        != "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
    ):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
        )
    _require_sha256(telemetry_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(telemetry_payload.get("plan_hash"), "plan_hash")
    _require_sha256(telemetry_payload.get("execution_hash"), "execution_hash")
    _require_sha256(telemetry_payload.get("handoff_hash"), "handoff_hash")
    _require_sha256(telemetry_payload.get("scheduler_hash"), "scheduler_hash")
    queue_entry_count = telemetry_payload.get("queue_entry_count")
    rows = telemetry_payload.get("rows")
    if not isinstance(queue_entry_count, int) or queue_entry_count < 0:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "queue_entry_count must be non-negative"
        )
    if not isinstance(rows, list):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: rows must be a list"
        )
    if queue_entry_count != len(rows):
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "queue_entry_count does not match rows"
        )
    return telemetry_payload


def _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
    handoff_payload: dict[str, object],
) -> dict[str, object]:
    if handoff_payload.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
    ):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        )
    _require_sha256(handoff_payload.get("adapter_handoff_hash"), "adapter_handoff_hash")
    _require_sha256(handoff_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(handoff_payload.get("plan_hash"), "plan_hash")
    _require_sha256(handoff_payload.get("execution_hash"), "execution_hash")
    entries = handoff_payload.get("entries")
    entry_count = handoff_payload.get("entry_count")
    if not isinstance(entry_count, int) or entry_count < 0:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "entry_count must be non-negative"
        )
    if not isinstance(entries, list):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: entries must be "
            "list"
        )
    if entry_count != len(entries):
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "entry_count does not match entries"
        )
    for entry in entries:
        if not isinstance(entry, dict):
            raise click.ClickException(
                "remediation scheduler adapter handoff schema mismatch: entry must be "
                "object"
            )
        _require_sha256(entry.get("adapter_entry_hash"), "adapter_entry_hash")
        _require_sha256(entry.get("entry_hash"), "entry_hash")
        _require_sha256(entry.get("action_hash"), "action_hash")
        _require_sha256(entry.get("request_hash"), "request_hash")
    return handoff_payload


def _load_lifecycle_remediation_scheduler_acknowledgement_payload(
    acknowledgement_payload: dict[str, object],
) -> dict[str, object]:
    if acknowledgement_payload.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_v1"
    ):
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: expected "
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_v1"
        )
    _require_sha256(
        acknowledgement_payload.get("acknowledgement_hash"),
        "acknowledgement_hash",
    )
    _require_sha256(
        acknowledgement_payload.get("adapter_handoff_hash"),
        "adapter_handoff_hash",
    )
    _require_sha256(acknowledgement_payload.get("telemetry_hash"), "telemetry_hash")
    _require_sha256(acknowledgement_payload.get("plan_hash"), "plan_hash")
    _require_sha256(acknowledgement_payload.get("execution_hash"), "execution_hash")
    _require_sha256(
        acknowledgement_payload.get("adapter_entry_hash"),
        "adapter_entry_hash",
    )
    _require_sha256(acknowledgement_payload.get("entry_hash"), "entry_hash")
    _require_sha256(acknowledgement_payload.get("action_hash"), "action_hash")
    _require_sha256(acknowledgement_payload.get("request_hash"), "request_hash")
    state = acknowledgement_payload.get("state")
    if state not in {"in_progress", "completed", "blocked"}:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: unsupported state"
        )
    for field_name in ("acknowledged_by", "external_reference"):
        value = acknowledgement_payload.get(field_name)
        if not isinstance(value, str) or not value:
            raise click.ClickException(
                "remediation scheduler acknowledgement schema mismatch: "
                f"{field_name} must be non-empty"
            )
    return acknowledgement_payload
