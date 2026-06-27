# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin scheduler queue and telemetry commands

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
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar, cast

import click

from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_deployment_handoff_payload,
    _load_lifecycle_remediation_scheduler_acknowledgement_payload,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _load_lifecycle_remediation_scheduler_queue_payload,
    _load_lifecycle_remediation_scheduler_telemetry_payload,
    _record_hash,
    _require_sha256,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group,
)

_F = TypeVar("_F", bound=Callable[..., object])


class _ClickCommandDecorator(Protocol):
    """Typed facade for Click command decorator factories."""

    def __call__(self, name: str) -> Callable[[_F], _F]:
        """Return a command decorator preserving the callback type."""


class _ClickDecoratorFactory(Protocol):
    """Typed facade for Click argument and option decorator factories."""

    def __call__(self, *args: object, **kwargs: object) -> Callable[[_F], _F]:
        """Return a parameter decorator preserving the callback type."""


_plugin_command = cast(_ClickCommandDecorator, plugins_group.command)
_click_argument = cast(_ClickDecoratorFactory, click.argument)
_click_option = cast(_ClickDecoratorFactory, click.option)


@_plugin_command("lifecycle-remediation-scheduler-queue")
@_click_argument(
    "handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_option(
    "--window-start-epoch",
    required=True,
    type=int,
    help="Scheduler window start as Unix epoch seconds (UTC).",
)
@_click_option(
    "--window-duration-seconds",
    default=3600,
    show_default=True,
    type=int,
    help="Scheduler execution window length in seconds.",
)
@_click_option(
    "--created-by",
    required=True,
    help="Scheduler component creating the queue payload.",
)
def plugins_lifecycle_remediation_scheduler_queue(
    handoff_json: Path,
    window_start_epoch: int,
    window_duration_seconds: int,
    created_by: str,
) -> None:
    """Emit a deterministic scheduler queue from remediation deployment handoff.

    Parameters
    ----------
    handoff_json : Path
        Path to the handoff JSON file.
    window_start_epoch : int
        Window start as a UNIX epoch.
    window_duration_seconds : int
        Window duration in seconds.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: created_by must be non-empty"
        )
    if window_start_epoch < 0:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "window_start_epoch must be non-negative"
        )
    if window_duration_seconds < 1:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: "
            "window_duration_seconds must be positive"
        )
    handoff = _load_lifecycle_remediation_deployment_handoff_payload(
        _load_json_file(handoff_json, artifact="remediation deployment handoff")
    )
    actions = cast(list[dict[str, object]], handoff["handoff_actions"])
    if len(actions) > window_duration_seconds:
        raise click.ClickException(
            "remediation scheduler queue schema mismatch: unresolved action count "
            "exceeds scheduler window duration"
        )
    queue_entries: list[dict[str, object]] = []
    for index, action in enumerate(
        sorted(
            actions,
            key=lambda item: (
                cast(int, item["priority"]),
                str(item["handoff_action_hash"]),
            ),
        )
    ):
        schedule_epoch = window_start_epoch + index
        entry: dict[str, object] = {
            "handoff_action_hash": action["handoff_action_hash"],
            "action_hash": action["action_hash"],
            "request_hash": action["request_hash"],
            "action_type": action["action_type"],
            "priority": action["priority"],
            "schedule_epoch": schedule_epoch,
            "scheduler_command_template": action["deployment_command_template"],
        }
        entry["entry_hash"] = _record_hash(entry)
        queue_entries.append(entry)
    queue_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            handoff.get("execution_hash"),
            "execution_hash",
        ),
        "handoff_hash": _require_sha256(handoff.get("handoff_hash"), "handoff_hash"),
        "window_start_epoch": window_start_epoch,
        "window_duration_seconds": window_duration_seconds,
        "queue_entry_count": len(queue_entries),
        "queue_entries": queue_entries,
        "created_by": created_by,
    }
    queue_payload["scheduler_hash"] = _record_hash(queue_payload)
    click.echo(json.dumps(queue_payload, indent=2, sort_keys=True))


@_plugin_command("lifecycle-remediation-scheduler-telemetry")
@_click_argument(
    "scheduler_queue_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_argument(
    "action_status_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_option(
    "--as-of-epoch",
    required=True,
    type=int,
    help="Telemetry snapshot epoch seconds (UTC).",
)
@_click_option(
    "--created-by",
    required=True,
    help="Scheduler component creating telemetry payload.",
)
def plugins_lifecycle_remediation_scheduler_telemetry(
    scheduler_queue_json: Path,
    action_status_json: tuple[Path, ...],
    as_of_epoch: int,
    created_by: str,
) -> None:
    """Emit deterministic operator telemetry for scheduler remediation queue.

    Parameters
    ----------
    scheduler_queue_json : Path
        Path to the scheduler queue JSON file.
    action_status_json : tuple[Path, ...]
        Path to the action status JSON file.
    as_of_epoch : int
        Reference time as a UNIX epoch.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "created_by must be non-empty"
        )
    if as_of_epoch < 0:
        raise click.ClickException(
            "remediation scheduler telemetry schema mismatch: "
            "as_of_epoch must be non-negative"
        )
    queue = _load_lifecycle_remediation_scheduler_queue_payload(
        _load_json_file(scheduler_queue_json, artifact="remediation scheduler queue")
    )
    status_by_action_hash: dict[str, dict[str, object]] = {}
    for path in action_status_json:
        status = _load_lifecycle_remediation_action_status_payload(
            _load_json_file(path, artifact="remediation action status")
        )
        action_hash = _require_sha256(status.get("action_hash"), "action_hash")
        if action_hash in status_by_action_hash:
            raise click.ClickException(
                "remediation scheduler telemetry schema mismatch: "
                "duplicate action status action_hash"
            )
        status_by_action_hash[action_hash] = {
            "state": status["state"],
            "status_hash": status["status_hash"],
            "updated_by": status.get("updated_by", ""),
            "note": status.get("note", ""),
        }

    queue_entries = cast(list[dict[str, object]], queue["queue_entries"])
    state_counts: dict[str, int] = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
        "overdue": 0,
    }
    rows: list[dict[str, object]] = []
    overdue_action_hashes: list[str] = []
    for entry in sorted(
        queue_entries,
        key=lambda item: (
            cast(int, item["schedule_epoch"]),
            cast(int, item["priority"]),
            str(item["action_hash"]),
        ),
    ):
        action_hash = _require_sha256(entry.get("action_hash"), "action_hash")
        schedule_epoch = cast(int, entry["schedule_epoch"])
        status_record = status_by_action_hash.get(action_hash)
        if status_record is None:
            state = "pending"
            status_hash: str | None = None
            updated_by: str | None = None
            note = ""
        else:
            state = cast(str, status_record["state"])
            status_hash = _require_sha256(status_record["status_hash"], "status_hash")
            updated_by = cast(str, status_record["updated_by"])
            note = cast(str, status_record["note"])
        overdue = state in {"pending", "in_progress", "blocked"} and (
            schedule_epoch < as_of_epoch
        )
        state_counts[state] += 1
        if overdue:
            state_counts["overdue"] += 1
            overdue_action_hashes.append(action_hash)
        row: dict[str, object] = {
            "entry_hash": entry["entry_hash"],
            "handoff_action_hash": entry["handoff_action_hash"],
            "action_hash": action_hash,
            "request_hash": entry["request_hash"],
            "action_type": entry["action_type"],
            "priority": entry["priority"],
            "schedule_epoch": schedule_epoch,
            "state": state,
            "overdue": overdue,
            "status_hash": status_hash,
            "updated_by": updated_by,
            "note": note,
        }
        rows.append(row)
    telemetry_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(queue.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            queue.get("execution_hash"), "execution_hash"
        ),
        "handoff_hash": _require_sha256(queue.get("handoff_hash"), "handoff_hash"),
        "scheduler_hash": _require_sha256(
            queue.get("scheduler_hash"), "scheduler_hash"
        ),
        "as_of_epoch": as_of_epoch,
        "queue_entry_count": len(queue_entries),
        "state_counts": state_counts,
        "overdue_action_hashes": sorted(overdue_action_hashes),
        "rows": rows,
        "created_by": created_by,
    }
    telemetry_payload["telemetry_hash"] = _record_hash(telemetry_payload)
    click.echo(json.dumps(telemetry_payload, indent=2, sort_keys=True))


@_plugin_command("lifecycle-remediation-scheduler-adapter-handoff")
@_click_argument(
    "scheduler_telemetry_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_option(
    "--adapter-name",
    required=True,
    help="External scheduler adapter name.",
)
@_click_option(
    "--adapter-endpoint",
    required=True,
    help="External scheduler adapter endpoint identifier.",
)
@_click_option(
    "--created-by",
    required=True,
    help="Component creating adapter handoff payload.",
)
def plugins_lifecycle_remediation_scheduler_adapter_handoff(
    scheduler_telemetry_json: Path,
    adapter_name: str,
    adapter_endpoint: str,
    created_by: str,
) -> None:
    """Emit deterministic external scheduler adapter handoff payload.

    Parameters
    ----------
    scheduler_telemetry_json : Path
        Path to the scheduler telemetry JSON file.
    adapter_name : str
        Name of the external adapter.
    adapter_endpoint : str
        Endpoint of the external adapter.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "created_by must be non-empty"
        )
    if not adapter_name:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "adapter_name must be non-empty"
        )
    if not adapter_endpoint:
        raise click.ClickException(
            "remediation scheduler adapter handoff schema mismatch: "
            "adapter_endpoint must be non-empty"
        )
    telemetry = _load_lifecycle_remediation_scheduler_telemetry_payload(
        _load_json_file(
            scheduler_telemetry_json,
            artifact="remediation scheduler telemetry",
        )
    )
    rows = cast(list[dict[str, object]], telemetry["rows"])
    active_rows = [
        row
        for row in rows
        if cast(str, row["state"]) in {"pending", "in_progress", "blocked"}
    ]
    entries: list[dict[str, object]] = []
    for row in sorted(
        active_rows,
        key=lambda item: (
            cast(int, item["priority"]),
            cast(int, item["schedule_epoch"]),
            str(item["action_hash"]),
        ),
    ):
        entry: dict[str, object] = {
            "entry_hash": row["entry_hash"],
            "handoff_action_hash": row["handoff_action_hash"],
            "action_hash": row["action_hash"],
            "request_hash": row["request_hash"],
            "action_type": row["action_type"],
            "priority": row["priority"],
            "schedule_epoch": row["schedule_epoch"],
            "overdue": row["overdue"],
            "adapter_target": {
                "adapter_name": adapter_name,
                "adapter_endpoint": adapter_endpoint,
            },
            "acknowledgement_command_template": (
                "spo plugins lifecycle-remediation-scheduler-acknowledgement "
                "ADAPTER_HANDOFF_JSON ENTRY_HASH --state STATE "
                "--acknowledged-by OPERATOR --external-reference REF"
            ),
        }
        entry["adapter_entry_hash"] = _record_hash(entry)
        entries.append(entry)
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_adapter_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(telemetry.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            telemetry.get("execution_hash"), "execution_hash"
        ),
        "telemetry_hash": _require_sha256(
            telemetry.get("telemetry_hash"), "telemetry_hash"
        ),
        "adapter_name": adapter_name,
        "adapter_endpoint": adapter_endpoint,
        "entry_count": len(entries),
        "entries": entries,
        "created_by": created_by,
    }
    payload["adapter_handoff_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@_plugin_command("lifecycle-remediation-scheduler-acknowledgement")
@_click_argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_argument("entry_hash")
@_click_option(
    "--state",
    required=True,
    type=click.Choice(["in_progress", "completed", "blocked"]),
    help="External scheduler execution state.",
)
@_click_option(
    "--acknowledged-by",
    required=True,
    help="Actor or component acknowledging execution.",
)
@_click_option(
    "--external-reference",
    required=True,
    help="External scheduler job/task reference.",
)
@_click_option(
    "--note",
    default="",
    show_default=True,
    help="Optional acknowledgement note.",
)
def plugins_lifecycle_remediation_scheduler_acknowledgement(
    adapter_handoff_json: Path,
    entry_hash: str,
    state: str,
    acknowledged_by: str,
    external_reference: str,
    note: str,
) -> None:
    """Emit deterministic acknowledgement artifact for adapter execution.

    Parameters
    ----------
    adapter_handoff_json : Path
        Path to the adapter handoff JSON file.
    entry_hash : str
        Hash of the handoff entry.
    state : str
        State label for the record.
    acknowledged_by : str
        Identifier of the acknowledged actor.
    external_reference : str
        External system reference.
    note : str
        Free-form note recorded with the record.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not acknowledged_by:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: "
            "acknowledged_by must be non-empty"
        )
    if not external_reference:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: "
            "external_reference must be non-empty"
        )
    handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    normalized_entry_hash = _require_sha256(entry_hash, "entry_hash")
    entries = cast(list[dict[str, object]], handoff["entries"])
    matched = next(
        (
            entry
            for entry in entries
            if entry["adapter_entry_hash"] == normalized_entry_hash
        ),
        None,
    )
    if matched is None:
        raise click.ClickException(
            "remediation scheduler acknowledgement schema mismatch: "
            "entry_hash not present in adapter handoff"
        )
    payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_acknowledgement_v1"
        ),
        "version": "1.0.0",
        "adapter_handoff_hash": _require_sha256(
            handoff.get("adapter_handoff_hash"), "adapter_handoff_hash"
        ),
        "telemetry_hash": _require_sha256(
            handoff.get("telemetry_hash"), "telemetry_hash"
        ),
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            handoff.get("execution_hash"), "execution_hash"
        ),
        "adapter_entry_hash": normalized_entry_hash,
        "entry_hash": matched["entry_hash"],
        "action_hash": matched["action_hash"],
        "request_hash": matched["request_hash"],
        "state": state,
        "acknowledged_by": acknowledged_by,
        "external_reference": external_reference,
        "note": note,
    }
    payload["acknowledgement_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@_plugin_command("lifecycle-remediation-scheduler-acknowledgement-replay")
@_click_argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_argument(
    "acknowledgement_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_option(
    "--created-by",
    required=True,
    help="Component creating acknowledgement replay manifest.",
)
def plugins_lifecycle_remediation_scheduler_acknowledgement_replay(
    adapter_handoff_json: Path,
    acknowledgement_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit deterministic replay manifest from scheduler acknowledgements.

    Parameters
    ----------
    adapter_handoff_json : Path
        Path to the adapter handoff JSON file.
    acknowledgement_json : tuple[Path, ...]
        Path to the acknowledgement JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler acknowledgement replay schema mismatch: "
            "created_by must be non-empty"
        )
    handoff = _load_lifecycle_remediation_scheduler_adapter_handoff_payload(
        _load_json_file(
            adapter_handoff_json,
            artifact="remediation scheduler adapter handoff",
        )
    )
    handoff_hash = _require_sha256(
        handoff.get("adapter_handoff_hash"),
        "adapter_handoff_hash",
    )
    entries = cast(list[dict[str, object]], handoff["entries"])
    entry_by_adapter_hash: dict[str, dict[str, object]] = {}
    for entry in entries:
        adapter_entry_hash = _require_sha256(
            entry.get("adapter_entry_hash"),
            "adapter_entry_hash",
        )
        entry_by_adapter_hash[adapter_entry_hash] = entry

    replay_rows: list[dict[str, object]] = []
    seen_adapter_entry_hashes: set[str] = set()
    for path in acknowledgement_json:
        payload = _load_lifecycle_remediation_scheduler_acknowledgement_payload(
            _load_json_file(path, artifact="remediation scheduler acknowledgement")
        )
        payload_handoff_hash = _require_sha256(
            payload.get("adapter_handoff_hash"),
            "adapter_handoff_hash",
        )
        if payload_handoff_hash != handoff_hash:
            raise click.ClickException(
                "remediation scheduler acknowledgement replay schema mismatch: "
                "adapter_handoff_hash mismatch"
            )
        adapter_entry_hash = _require_sha256(
            payload.get("adapter_entry_hash"),
            "adapter_entry_hash",
        )
        if adapter_entry_hash in seen_adapter_entry_hashes:
            raise click.ClickException(
                "remediation scheduler acknowledgement replay schema mismatch: "
                "duplicate adapter_entry_hash acknowledgement"
            )
        handoff_entry = entry_by_adapter_hash.get(adapter_entry_hash)
        if handoff_entry is None:
            raise click.ClickException(
                "remediation scheduler acknowledgement replay schema mismatch: "
                "acknowledgement adapter_entry_hash missing from handoff"
            )
        seen_adapter_entry_hashes.add(adapter_entry_hash)
        replay_row: dict[str, object] = {
            "acknowledgement_hash": _require_sha256(
                payload.get("acknowledgement_hash"),
                "acknowledgement_hash",
            ),
            "adapter_entry_hash": adapter_entry_hash,
            "entry_hash": handoff_entry["entry_hash"],
            "action_hash": handoff_entry["action_hash"],
            "request_hash": handoff_entry["request_hash"],
            "state": payload["state"],
            "external_reference": payload["external_reference"],
            "acknowledged_by": payload["acknowledged_by"],
            "note": payload.get("note", ""),
        }
        replay_row["replay_row_hash"] = _record_hash(replay_row)
        replay_rows.append(replay_row)

    state_counts: dict[str, int] = {"in_progress": 0, "completed": 0, "blocked": 0}
    for row in replay_rows:
        state_counts[cast(str, row["state"])] += 1

    replay_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_acknowledgement_replay_v1"
        ),
        "version": "1.0.0",
        "adapter_handoff_hash": handoff_hash,
        "plan_hash": _require_sha256(handoff.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            handoff.get("execution_hash"), "execution_hash"
        ),
        "telemetry_hash": _require_sha256(
            handoff.get("telemetry_hash"), "telemetry_hash"
        ),
        "acknowledgement_count": len(replay_rows),
        "state_counts": state_counts,
        "rows": sorted(
            replay_rows,
            key=lambda item: (
                cast(str, item["state"]),
                cast(str, item["adapter_entry_hash"]),
            ),
        ),
        "created_by": created_by,
    }
    replay_payload["replay_hash"] = _record_hash(replay_payload)
    click.echo(json.dumps(replay_payload, indent=2, sort_keys=True))


@_plugin_command("lifecycle-remediation-scheduler-execution-dashboard")
@_click_argument(
    "scheduler_telemetry_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_argument(
    "acknowledgement_replay_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@_click_option(
    "--created-by",
    required=True,
    help="Component creating external scheduler execution dashboard.",
)
def plugins_lifecycle_remediation_scheduler_execution_dashboard(
    scheduler_telemetry_json: Path,
    acknowledgement_replay_json: Path,
    created_by: str,
) -> None:
    """Emit deterministic live execution dashboard across scheduler adapters.

    Parameters
    ----------
    scheduler_telemetry_json : Path
        Path to the scheduler telemetry JSON file.
    acknowledgement_replay_json : Path
        Path to the acknowledgement replay JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "remediation scheduler execution dashboard schema mismatch: "
            "created_by must be non-empty"
        )
    telemetry = _load_lifecycle_remediation_scheduler_telemetry_payload(
        _load_json_file(
            scheduler_telemetry_json,
            artifact="remediation scheduler telemetry",
        )
    )
    replay = _load_json_file(
        acknowledgement_replay_json,
        artifact="remediation scheduler acknowledgement replay",
    )
    if replay.get("schema") != (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_replay_v1"
    ):
        raise click.ClickException(
            "remediation scheduler execution dashboard schema mismatch: "
            "unexpected acknowledgement replay schema"
        )
    _require_sha256(replay.get("replay_hash"), "replay_hash")
    telemetry_hash = _require_sha256(telemetry.get("telemetry_hash"), "telemetry_hash")
    replay_telemetry_hash = _require_sha256(
        replay.get("telemetry_hash"), "telemetry_hash"
    )
    if replay_telemetry_hash != telemetry_hash:
        raise click.ClickException(
            "remediation scheduler execution dashboard schema mismatch: "
            "telemetry_hash mismatch"
        )
    replay_rows = cast(list[dict[str, object]], replay.get("rows", []))
    ack_state_by_action_hash: dict[str, str] = {}
    for row in replay_rows:
        action_hash = _require_sha256(row.get("action_hash"), "action_hash")
        state = row.get("state")
        if not isinstance(state, str) or state not in {
            "in_progress",
            "completed",
            "blocked",
        }:
            raise click.ClickException(
                "remediation scheduler execution dashboard schema mismatch: "
                "unsupported replay state"
            )
        if action_hash in ack_state_by_action_hash:
            raise click.ClickException(
                "remediation scheduler execution dashboard schema mismatch: "
                "duplicate replay action_hash"
            )
        ack_state_by_action_hash[action_hash] = state

    telemetry_rows = cast(list[dict[str, object]], telemetry["rows"])
    rows: list[dict[str, object]] = []
    dashboard_counts: dict[str, int] = {
        "pending": 0,
        "in_progress": 0,
        "completed": 0,
        "blocked": 0,
        "overdue": 0,
    }
    for row in sorted(
        telemetry_rows,
        key=lambda item: (
            cast(int, item["priority"]),
            cast(int, item["schedule_epoch"]),
            str(item["action_hash"]),
        ),
    ):
        action_hash = _require_sha256(row.get("action_hash"), "action_hash")
        telemetry_state = cast(str, row["state"])
        ack_state = ack_state_by_action_hash.get(action_hash)
        effective_state = ack_state if ack_state is not None else telemetry_state
        if effective_state not in {"pending", "in_progress", "completed", "blocked"}:
            raise click.ClickException(
                "remediation scheduler execution dashboard schema mismatch: "
                "unsupported effective state"
            )
        overdue = bool(row["overdue"]) and effective_state != "completed"
        dashboard_counts[effective_state] += 1
        if overdue:
            dashboard_counts["overdue"] += 1
        output_row: dict[str, object] = {
            "entry_hash": row["entry_hash"],
            "action_hash": action_hash,
            "request_hash": row["request_hash"],
            "action_type": row["action_type"],
            "priority": row["priority"],
            "schedule_epoch": row["schedule_epoch"],
            "telemetry_state": telemetry_state,
            "acknowledgement_state": ack_state,
            "effective_state": effective_state,
            "overdue": overdue,
        }
        output_row["dashboard_row_hash"] = _record_hash(output_row)
        rows.append(output_row)

    dashboard_payload: dict[str, object] = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_"
            "remediation_scheduler_execution_dashboard_v1"
        ),
        "version": "1.0.0",
        "plan_hash": _require_sha256(telemetry.get("plan_hash"), "plan_hash"),
        "execution_hash": _require_sha256(
            telemetry.get("execution_hash"), "execution_hash"
        ),
        "handoff_hash": _require_sha256(telemetry.get("handoff_hash"), "handoff_hash"),
        "scheduler_hash": _require_sha256(
            telemetry.get("scheduler_hash"), "scheduler_hash"
        ),
        "telemetry_hash": telemetry_hash,
        "replay_hash": _require_sha256(replay.get("replay_hash"), "replay_hash"),
        "row_count": len(rows),
        "state_counts": dashboard_counts,
        "rows": rows,
        "created_by": created_by,
    }
    dashboard_payload["dashboard_hash"] = _record_hash(dashboard_payload)
    click.echo(json.dumps(dashboard_payload, indent=2, sort_keys=True))
