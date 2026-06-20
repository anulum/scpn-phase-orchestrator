# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin lifecycle remediation commands

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
    _load_lifecycle_multistore_drilldown_payload,
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_execution_dashboard_payload,
    _load_lifecycle_remediation_plan_payload,
    _record_hash,
    _require_sha256,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group,
)


@plugins_group.command("lifecycle-remediation-orchestration")
@click.argument(
    "drilldown_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the remediation plan.",
)
def plugins_lifecycle_remediation_orchestration(
    drilldown_json: Path,
    created_by: str,
) -> None:
    """Emit a deterministic, priority-ordered cross-store remediation plan.

    Parameters
    ----------
    drilldown_json : Path
        Path to the drilldown JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation orchestration schema mismatch: created_by must be non-empty"
        )
    drilldown = _load_lifecycle_multistore_drilldown_payload(
        _load_json_file(drilldown_json, artifact="multi-store drilldown")
    )
    stores = cast(list[dict[str, object]], drilldown["stores"])
    actions: list[dict[str, object]] = []
    priority_map: dict[str, int] = {
        "renew_approval": 1,
        "persist_request": 2,
        "register_storage_adapter": 3,
        "confirm_external_write": 4,
    }
    for store in stores:
        store_hash = _require_sha256(store.get("store_hash"), "store_hash")
        policy_hash = _require_sha256(store.get("policy_hash"), "policy_hash")
        summary_hash = _require_sha256(store.get("summary_hash"), "summary_hash")
        for action_type, source_field in (
            ("renew_approval", "renewal_required_request_hashes"),
            ("persist_request", "storage_missing_request_hashes"),
            ("register_storage_adapter", "missing_adapter_request_hashes"),
            ("confirm_external_write", "external_write_followup_request_hashes"),
        ):
            request_hashes = cast(list[str], store[source_field])
            for request_hash in request_hashes:
                action: dict[str, object] = {
                    "action_type": action_type,
                    "priority": priority_map[action_type],
                    "request_hash": request_hash,
                    "store_hash": store_hash,
                    "policy_hash": policy_hash,
                    "summary_hash": summary_hash,
                }
                action["action_hash"] = _record_hash(action)
                actions.append(action)
    actions.sort(
        key=lambda item: (
            cast(int, item["priority"]),
            str(item["request_hash"]),
            str(item["store_hash"]),
            str(item["action_type"]),
        )
    )
    orchestration_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_remediation_plan_v1",
        "version": "1.0.0",
        "drilldown_hash": _require_sha256(
            drilldown.get("drilldown_hash"),
            "drilldown_hash",
        ),
        "action_count": len(actions),
        "actions": actions,
        "created_by": created_by,
    }
    orchestration_payload["plan_hash"] = _record_hash(orchestration_payload)
    click.echo(json.dumps(orchestration_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-action-status")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("action_hash")
@click.option(
    "--state",
    type=click.Choice(["pending", "in_progress", "completed", "blocked"]),
    required=True,
    help="Execution state for the remediation action.",
)
@click.option(
    "--updated-by",
    required=True,
    help="Operator or deployment component updating the action state.",
)
@click.option(
    "--note",
    default="",
    show_default=False,
    help="Optional operator note for this state transition.",
)
def plugins_lifecycle_remediation_action_status(
    plan_json: Path,
    action_hash: str,
    state: str,
    updated_by: str,
    note: str,
) -> None:
    """Emit a deterministic remediation action status record.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    action_hash : str
        Hash of the remediation action.
    state : str
        State label for the record.
    updated_by : str
        Identifier of the updating actor.
    note : str
        Free-form note recorded with the record.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not updated_by:
        raise click.ClickException(
            "remediation action status schema mismatch: updated_by must be non-empty"
        )
    action_hash = _require_sha256(action_hash, "action_hash")
    plan = _load_lifecycle_remediation_plan_payload(
        _load_json_file(plan_json, artifact="remediation plan")
    )
    actions = cast(list[dict[str, object]], plan["actions"])
    selected: dict[str, object] | None = None
    for action in actions:
        if action["action_hash"] == action_hash:
            selected = action
            break
    if selected is None:
        raise click.ClickException("action_hash is not part of the remediation plan")
    status_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(plan["plan_hash"], "plan_hash"),
        "action_hash": action_hash,
        "request_hash": selected["request_hash"],
        "store_hash": selected["store_hash"],
        "action_type": selected["action_type"],
        "priority": selected["priority"],
        "state": state,
        "updated_by": updated_by,
        "note": note,
    }
    status_payload["status_hash"] = _record_hash(status_payload)
    click.echo(json.dumps(status_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-execution-dashboard")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "status_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the execution dashboard.",
)
def plugins_lifecycle_remediation_execution_dashboard(
    plan_json: Path,
    status_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic closed-loop dashboard for remediation execution.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    status_json : tuple[Path, ...]
        Path to the status JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation dashboard schema mismatch: created_by must be non-empty"
        )
    plan = _load_lifecycle_remediation_plan_payload(
        _load_json_file(plan_json, artifact="remediation plan")
    )
    statuses = tuple(
        _load_lifecycle_remediation_action_status_payload(
            _load_json_file(path, artifact="remediation action status")
        )
        for path in status_json
    )
    plan_hash = _require_sha256(plan["plan_hash"], "plan_hash")
    actions = cast(list[dict[str, object]], plan["actions"])
    action_by_hash = {
        _require_sha256(action["action_hash"], "action_hash"): action
        for action in actions
    }
    status_by_hash: dict[str, dict[str, object]] = {}
    for status_record in statuses:
        status_plan_hash = _require_sha256(status_record["plan_hash"], "plan_hash")
        if status_plan_hash != plan_hash:
            raise click.ClickException(
                "status plan_hash does not match remediation plan"
            )
        action_hash = _require_sha256(status_record["action_hash"], "action_hash")
        if action_hash not in action_by_hash:
            raise click.ClickException(
                "status action_hash is not part of remediation plan"
            )
        if action_hash in status_by_hash:
            raise click.ClickException(
                "duplicate remediation action status for action_hash"
            )
        status_by_hash[action_hash] = status_record
    state_counts = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
    }
    unresolved: list[str] = []
    resolved: list[str] = []
    execution_rows: list[dict[str, object]] = []
    for action in sorted(
        actions,
        key=lambda item: (
            cast(int, item["priority"]),
            str(item["request_hash"]),
            str(item["action_hash"]),
        ),
    ):
        action_hash = _require_sha256(action["action_hash"], "action_hash")
        status = status_by_hash.get(action_hash)
        if status is None:
            state = "pending"
            status_hash: str | None = None
            updated_by: str | None = None
            note = ""
        else:
            state = cast(str, status["state"])
            status_hash = _require_sha256(status["status_hash"], "status_hash")
            updated_by = str(status["updated_by"])
            note = str(status.get("note", ""))
        state_counts[state] += 1
        if state in {"completed"}:
            resolved.append(action_hash)
        else:
            unresolved.append(action_hash)
        execution_rows.append(
            {
                "action_hash": action_hash,
                "request_hash": action["request_hash"],
                "action_type": action["action_type"],
                "priority": action["priority"],
                "state": state,
                "status_hash": status_hash,
                "updated_by": updated_by,
                "note": note,
            }
        )
    dashboard_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
        ),
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "action_count": len(actions),
        "state_counts": state_counts,
        "resolved_action_hashes": sorted(resolved),
        "unresolved_action_hashes": sorted(unresolved),
        "rows": execution_rows,
        "created_by": created_by,
    }
    dashboard_payload["execution_hash"] = _record_hash(dashboard_payload)
    click.echo(json.dumps(dashboard_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-remediation-deployment-handoff")
@click.argument(
    "execution_dashboard_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the deployment handoff.",
)
def plugins_lifecycle_remediation_deployment_handoff(
    execution_dashboard_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic deployment handoff actions for unresolved remediation.

    Parameters
    ----------
    execution_dashboard_json : Path
        Path to the execution dashboard JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation deployment handoff schema mismatch: "
            "created_by must be non-empty"
        )
    dashboard = _load_lifecycle_remediation_execution_dashboard_payload(
        _load_json_file(
            execution_dashboard_json,
            artifact="remediation execution dashboard",
        )
    )
    plan_hash = _require_sha256(dashboard.get("plan_hash"), "plan_hash")
    rows = cast(list[dict[str, object]], dashboard["rows"])
    unresolved_rows = [
        row for row in rows if row["state"] in {"pending", "in_progress", "blocked"}
    ]
    command_templates = {
        "renew_approval": (
            "spo plugins approve-execution-plan PLAN_JSON "
            "--operator-id OPERATOR_ID --approval-reference REF "
            "--approval-reason REASON"
        ),
        "persist_request": (
            "spo plugins persist-execution-request REQUEST_JSON OUTPUT_JSON "
            "--storage-uri STORAGE_URI --created-by DEPLOYMENT_COMPONENT"
        ),
        "register_storage_adapter": (
            "spo plugins storage-adapter-manifest REQUEST_JSON "
            "--storage-uri STORAGE_URI --storage-backend BACKEND "
            "--created-by DEPLOYMENT_COMPONENT"
        ),
        "confirm_external_write": (
            "Record external storage/API write completion and emit "
            "spo plugins lifecycle-remediation-action-status PLAN_JSON ACTION_HASH "
            "--state completed --updated-by DEPLOYMENT_COMPONENT"
        ),
    }
    handoff_actions: list[dict[str, object]] = []
    for row in sorted(
        unresolved_rows,
        key=lambda item: (
            cast(int, item["priority"]),
            str(item["action_hash"]),
        ),
    ):
        action_type = cast(str, row["action_type"])
        handoff_action: dict[str, object] = {
            "action_hash": row["action_hash"],
            "request_hash": row["request_hash"],
            "action_type": action_type,
            "priority": row["priority"],
            "state": row["state"],
            "deployment_command_template": command_templates[action_type],
        }
        handoff_action["handoff_action_hash"] = _record_hash(handoff_action)
        handoff_actions.append(handoff_action)
    handoff_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": plan_hash,
        "execution_hash": _require_sha256(
            dashboard.get("execution_hash"), "execution_hash"
        ),
        "unresolved_action_count": len(unresolved_rows),
        "handoff_actions": handoff_actions,
        "created_by": created_by,
    }
    handoff_payload["handoff_hash"] = _record_hash(handoff_payload)
    click.echo(json.dumps(handoff_payload, indent=2, sort_keys=True))
