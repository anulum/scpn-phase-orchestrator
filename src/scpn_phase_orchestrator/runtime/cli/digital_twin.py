# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI digital-twin observability and deployment commands

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

import click

from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _record_hash,
    _require_sha256,
)
from scpn_phase_orchestrator.runtime.observability import RuntimeObservability


@main.command("digital-twin-observability-bundle")
@click.argument(
    "operator_evidence_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--scheduler-dashboard-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional scheduler execution dashboard JSON for replay linkage.",
)
@click.option(
    "--scheduler-replay-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional scheduler acknowledgement replay JSON for replay linkage.",
)
@click.option(
    "--metric-prefix",
    default="spo",
    show_default=True,
    help="Prometheus metric prefix for rendered observability text.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator component creating the observability bundle artifact.",
)
def digital_twin_observability_bundle(
    operator_evidence_json: Path,
    scheduler_dashboard_json: Path | None,
    scheduler_replay_json: Path | None,
    metric_prefix: str,
    created_by: str,
) -> None:
    """Bundle digital-twin Prometheus telemetry with replay linkage evidence.

    Parameters
    ----------
    operator_evidence_json : Path
        Path to the operator-evidence JSON.
    scheduler_dashboard_json : Path | None
        Path to the scheduler dashboard JSON, or ``None``.
    scheduler_replay_json : Path | None
        Path to the scheduler replay JSON, or ``None``.
    metric_prefix : str
        Prefix applied to emitted metric names.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "digital-twin observability bundle schema mismatch: "
            "created_by must be non-empty"
        )
    evidence = _load_json_file(
        operator_evidence_json,
        artifact="digital-twin operator evidence",
    )
    observability = RuntimeObservability(metric_prefix=metric_prefix)
    try:
        prometheus_text = observability.digital_twin_prometheus_text(evidence)
    except ValueError as exc:
        raise click.ClickException(
            f"digital-twin observability bundle schema mismatch: {exc}"
        ) from exc

    replay_linkage: dict[str, object] = {
        "scheduler_dashboard_present": scheduler_dashboard_json is not None,
        "scheduler_replay_present": scheduler_replay_json is not None,
        "scheduler_row_count": 0,
        "scheduler_overdue_count": 0,
        "scheduler_blocked_count": 0,
        "scheduler_completed_count": 0,
        "scheduler_replay_count": 0,
        "scheduler_replay_blocked_count": 0,
        "scheduler_replay_completed_count": 0,
        "scheduler_dashboard_hash": None,
        "scheduler_replay_hash": None,
    }

    if scheduler_dashboard_json is not None:
        dashboard = _load_json_file(
            scheduler_dashboard_json,
            artifact="remediation scheduler execution dashboard",
        )
        if dashboard.get("schema") != (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_execution_dashboard_v1"
        ):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "unexpected scheduler dashboard schema"
            )
        dashboard_hash = _require_sha256(
            dashboard.get("dashboard_hash"), "dashboard_hash"
        )
        rows = dashboard.get("rows")
        if not isinstance(rows, list):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "scheduler dashboard rows must be list"
            )
        blocked_count = 0
        completed_count = 0
        overdue_count = 0
        for row in rows:
            if not isinstance(row, dict):
                raise click.ClickException(
                    "digital-twin observability bundle schema mismatch: "
                    "scheduler dashboard row must be object"
                )
            state = row.get("effective_state")
            if state == "blocked":
                blocked_count += 1
            if state == "completed":
                completed_count += 1
            if bool(row.get("overdue", False)):
                overdue_count += 1
        replay_linkage["scheduler_row_count"] = len(rows)
        replay_linkage["scheduler_overdue_count"] = overdue_count
        replay_linkage["scheduler_blocked_count"] = blocked_count
        replay_linkage["scheduler_completed_count"] = completed_count
        replay_linkage["scheduler_dashboard_hash"] = dashboard_hash

    if scheduler_replay_json is not None:
        replay = _load_json_file(
            scheduler_replay_json,
            artifact="remediation scheduler acknowledgement replay",
        )
        if replay.get("schema") != (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_replay_v1"
        ):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "unexpected scheduler replay schema"
            )
        replay_hash = _require_sha256(replay.get("replay_hash"), "replay_hash")
        replay_rows = replay.get("rows")
        if not isinstance(replay_rows, list):
            raise click.ClickException(
                "digital-twin observability bundle schema mismatch: "
                "scheduler replay rows must be list"
            )
        replay_blocked = 0
        replay_completed = 0
        for row in replay_rows:
            if not isinstance(row, dict):
                raise click.ClickException(
                    "digital-twin observability bundle schema mismatch: "
                    "scheduler replay row must be object"
                )
            state = row.get("state")
            if state == "blocked":
                replay_blocked += 1
            if state == "completed":
                replay_completed += 1
        replay_linkage["scheduler_replay_count"] = len(replay_rows)
        replay_linkage["scheduler_replay_blocked_count"] = replay_blocked
        replay_linkage["scheduler_replay_completed_count"] = replay_completed
        replay_linkage["scheduler_replay_hash"] = replay_hash

    accepted_count = evidence.get("accepted_count", 0)
    rejected_count = evidence.get("rejected_count", 0)
    if not isinstance(accepted_count, int) or isinstance(accepted_count, bool):
        raise click.ClickException(
            "digital-twin observability bundle schema mismatch: "
            "accepted_count must be an integer"
        )
    if not isinstance(rejected_count, int) or isinstance(rejected_count, bool):
        raise click.ClickException(
            "digital-twin observability bundle schema mismatch: "
            "rejected_count must be an integer"
        )

    bundle_payload: dict[str, object] = {
        "schema": "scpn_digital_twin_observability_bundle_v1",
        "version": "1.0.0",
        "contract_hash": _require_sha256(
            evidence.get("contract_hash"), "contract_hash"
        ),
        "status": str(evidence.get("status")),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "prometheus_metric_prefix": metric_prefix,
        "prometheus_text": prometheus_text,
        "replay_linkage": replay_linkage,
        "created_by": created_by,
    }
    bundle_payload["bundle_hash"] = _record_hash(bundle_payload)
    click.echo(json.dumps(bundle_payload, indent=2, sort_keys=True))


@main.command("digital-twin-grafana-dashboard-pack")
@click.argument(
    "observability_bundle_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--adapter-family",
    required=True,
    help="Adapter family label (for example: rest, grpc, kafka, hardware).",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating Grafana dashboard pack artifact.",
)
def digital_twin_grafana_dashboard_pack(
    observability_bundle_json: Path,
    adapter_family: str,
    created_by: str,
) -> None:
    """Emit deterministic Grafana dashboard pack from observability bundle.

    Parameters
    ----------
    observability_bundle_json : Path
        Path to the observability bundle JSON file.
    adapter_family : str
        Adapter family label.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "created_by must be non-empty"
        )
    if not adapter_family:
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "adapter_family must be non-empty"
        )
    bundle = _load_json_file(
        observability_bundle_json,
        artifact="digital-twin observability bundle",
    )
    if bundle.get("schema") != "scpn_digital_twin_observability_bundle_v1":
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "unexpected observability bundle schema"
        )
    bundle_hash = _require_sha256(bundle.get("bundle_hash"), "bundle_hash")
    contract_hash = _require_sha256(bundle.get("contract_hash"), "contract_hash")
    metric_prefix = bundle.get("prometheus_metric_prefix")
    if not isinstance(metric_prefix, str) or not metric_prefix:
        raise click.ClickException(
            "digital-twin grafana dashboard pack schema mismatch: "
            "prometheus_metric_prefix must be non-empty string"
        )
    panels = [
        {
            "title": "Sync Acceptance Ratio",
            "kind": "timeseries",
            "query_template": (
                f"sum({metric_prefix}_digital_twin_sync_accepted_total"
                f'{{contract_hash="{contract_hash}"}}) / '
                f"(sum({metric_prefix}_digital_twin_sync_accepted_total"
                f'{{contract_hash="{contract_hash}"}}) + '
                f"sum({metric_prefix}_digital_twin_sync_rejected_total"
                f'{{contract_hash="{contract_hash}"}}))'
            ),
            "unit": "percentunit",
        },
        {
            "title": "Twin Residual Max",
            "kind": "timeseries",
            "query_template": (
                f'{metric_prefix}_digital_twin_max_abs_residual{{contract_hash="{contract_hash}"}}'
            ),
            "unit": "none",
        },
        {
            "title": "Unhealthy Adapter Count",
            "kind": "stat",
            "query_template": (
                f'{metric_prefix}_digital_twin_unhealthy_adapter_count{{contract_hash="{contract_hash}"}}'
            ),
            "unit": "short",
        },
        {
            "title": "Twin Mismatch Reasons",
            "kind": "barchart",
            "query_template": (
                f"sum by (reason) "
                f"({metric_prefix}_digital_twin_mismatch_reason_count"
                f'{{contract_hash="{contract_hash}"}})'
            ),
            "unit": "short",
        },
        {
            "title": "Scheduler Overdue Actions",
            "kind": "stat",
            "query_template": "linked_bundle.replay_linkage.scheduler_overdue_count",
            "unit": "short",
        },
    ]
    for panel in panels:
        panel["panel_hash"] = _record_hash(panel)
    payload: dict[str, object] = {
        "schema": "scpn_digital_twin_grafana_dashboard_pack_v1",
        "version": "1.0.0",
        "adapter_family": adapter_family,
        "contract_hash": contract_hash,
        "observability_bundle_hash": bundle_hash,
        "panel_count": len(panels),
        "panels": panels,
        "created_by": created_by,
    }
    payload["dashboard_pack_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@main.command("digital-twin-live-deployment-playbook")
@click.argument(
    "observability_bundle_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "grafana_dashboard_pack_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--environment-name",
    required=True,
    help="Deployment environment name (for example: prod-eu-west).",
)
@click.option(
    "--created-by",
    required=True,
    help="Component creating live deployment playbook artifact.",
)
def digital_twin_live_deployment_playbook(
    observability_bundle_json: Path,
    grafana_dashboard_pack_json: Path,
    environment_name: str,
    created_by: str,
) -> None:
    """Emit deterministic live deployment playbook from observability artifacts.

    Parameters
    ----------
    observability_bundle_json : Path
        Path to the observability bundle JSON file.
    grafana_dashboard_pack_json : Path
        Path to the Grafana dashboard-pack JSON.
    environment_name : str
        Target environment name.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "created_by must be non-empty"
        )
    if not environment_name:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "environment_name must be non-empty"
        )
    bundle = _load_json_file(
        observability_bundle_json,
        artifact="digital-twin observability bundle",
    )
    if bundle.get("schema") != "scpn_digital_twin_observability_bundle_v1":
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "unexpected observability bundle schema"
        )
    dashboard_pack = _load_json_file(
        grafana_dashboard_pack_json,
        artifact="digital-twin grafana dashboard pack",
    )
    if dashboard_pack.get("schema") != "scpn_digital_twin_grafana_dashboard_pack_v1":
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "unexpected grafana dashboard pack schema"
        )
    bundle_hash = _require_sha256(bundle.get("bundle_hash"), "bundle_hash")
    dashboard_linked_bundle_hash = _require_sha256(
        dashboard_pack.get("observability_bundle_hash"),
        "observability_bundle_hash",
    )
    if bundle_hash != dashboard_linked_bundle_hash:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "observability_bundle_hash mismatch"
        )
    replay_linkage = bundle.get("replay_linkage")
    if not isinstance(replay_linkage, dict):
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "replay_linkage must be object"
        )
    overdue = replay_linkage.get("scheduler_overdue_count")
    blocked = replay_linkage.get("scheduler_blocked_count")
    if not isinstance(overdue, int) or overdue < 0:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "scheduler_overdue_count must be non-negative integer"
        )
    if not isinstance(blocked, int) or blocked < 0:
        raise click.ClickException(
            "digital-twin live deployment playbook schema mismatch: "
            "scheduler_blocked_count must be non-negative integer"
        )
    rollout_gate = (
        "blocked" if blocked > 0 else ("degraded" if overdue > 0 else "ready")
    )
    steps = [
        {
            "id": "publish-metrics",
            "description": (
                "Expose Prometheus text from digital-twin observability bundle."
            ),
            "command_template": (
                "spo digital-twin-observability-bundle EVIDENCE_JSON "
                "--created-by OPERATOR"
            ),
        },
        {
            "id": "publish-dashboards",
            "description": "Deploy Grafana dashboard pack for adapter family.",
            "command_template": (
                "spo digital-twin-grafana-dashboard-pack OBS_BUNDLE_JSON "
                "--adapter-family FAMILY --created-by OPERATOR"
            ),
        },
        {
            "id": "verify-scheduler-health",
            "description": "Review overdue/blocked scheduler telemetry linkage.",
            "command_template": (
                "Inspect replay_linkage.scheduler_overdue_count and "
                "replay_linkage.scheduler_blocked_count in observability bundle"
            ),
        },
    ]
    for step in steps:
        step["step_hash"] = _record_hash(step)
    payload: dict[str, object] = {
        "schema": "scpn_digital_twin_live_deployment_playbook_v1",
        "version": "1.0.0",
        "environment_name": environment_name,
        "contract_hash": _require_sha256(bundle.get("contract_hash"), "contract_hash"),
        "observability_bundle_hash": bundle_hash,
        "dashboard_pack_hash": _require_sha256(
            dashboard_pack.get("dashboard_pack_hash"),
            "dashboard_pack_hash",
        ),
        "rollout_gate": rollout_gate,
        "step_count": len(steps),
        "steps": steps,
        "created_by": created_by,
    }
    payload["playbook_hash"] = _record_hash(payload)
    click.echo(json.dumps(payload, indent=2, sort_keys=True))
