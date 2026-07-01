# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin execution-request lifecycle commands

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

from scpn_phase_orchestrator.plugins import (
    PluginExecutionRequestRevocationList,
    PluginExecutionRequestStorageManifest,
    build_plugin_execution_request_lifecycle_policy_report,
    build_plugin_execution_request_lifecycle_record,
    build_plugin_execution_request_lifecycle_summary,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _load_lifecycle_from_payload,
    _load_lifecycle_policy_report_payload,
    _load_lifecycle_summary_from_payload,
    _load_request_from_payload,
    _load_revocation_list_from_payload,
    _load_storage_adapter_from_payload,
    _load_storage_manifest_from_payload,
    _record_hash,
    _require_sha256,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group,
)


@plugins_group.command("lifecycle-status")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-bundle",
    "storage_bundle_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Persisted request bundle JSON to include storage status.",
)
@click.option(
    "--revocation-list",
    "revocation_list_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Aggregate revocation-list JSON to include lifecycle status.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the lifecycle status.",
)
def plugins_lifecycle_status(
    request_json: Path,
    storage_bundle_path: Path | None,
    revocation_list_path: Path | None,
    created_by: str,
) -> None:
    """Emit an operator lifecycle status record for an execution request.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    storage_bundle_path : Path | None
        Filesystem path to the storage bundle.
    revocation_list_path : Path | None
        Filesystem path to the revocation list.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)
    storage_manifest: PluginExecutionRequestStorageManifest | None = None
    revocation_list: PluginExecutionRequestRevocationList | None = None

    try:
        if storage_bundle_path is not None:
            bundle = _load_json_file(storage_bundle_path, artifact="storage bundle")
            manifest_payload = bundle.get("storage_manifest")
            if not isinstance(manifest_payload, dict):
                raise click.ClickException(
                    "storage bundle storage_manifest must be an object"
                )
            storage_manifest = _load_storage_manifest_from_payload(
                cast(dict[str, object], manifest_payload)
            )
        if revocation_list_path is not None:
            revocation_list = _load_revocation_list_from_payload(
                _load_json_file(revocation_list_path, artifact="revocation list")
            )
        lifecycle_record = build_plugin_execution_request_lifecycle_record(
            request,
            created_by=created_by,
            storage_manifest=storage_manifest,
            revocation_list=revocation_list,
        )
    except (PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(lifecycle_record.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-summary")
@click.argument(
    "lifecycle_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the lifecycle summary.",
)
def plugins_lifecycle_summary(
    lifecycle_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic batch summary for lifecycle-status records.

    Parameters
    ----------
    lifecycle_json : tuple[Path, ...]
        Path to the lifecycle JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    lifecycle_records = tuple(
        _load_lifecycle_from_payload(_load_json_file(path, artifact="lifecycle"))
        for path in lifecycle_json
    )
    try:
        summary = build_plugin_execution_request_lifecycle_summary(
            lifecycle_records,
            created_by=created_by,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(summary.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-policy-report")
@click.argument(
    "summary_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-adapter",
    "storage_adapter_paths",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Storage-adapter manifest JSON to include in the policy report.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the policy report.",
)
def plugins_lifecycle_policy_report(
    summary_json: Path,
    storage_adapter_paths: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic lifecycle policy report for operator dashboards.

    Parameters
    ----------
    summary_json : Path
        Path to the summary JSON.
    storage_adapter_paths : tuple[Path, ...]
        Paths to the storage-adapter handoff manifests.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    summary = _load_lifecycle_summary_from_payload(
        _load_json_file(summary_json, artifact="lifecycle summary")
    )
    storage_adapters = tuple(
        _load_storage_adapter_from_payload(
            _load_json_file(path, artifact="storage adapter")
        )
        for path in storage_adapter_paths
    )
    try:
        report = build_plugin_execution_request_lifecycle_policy_report(
            summary,
            storage_adapters=storage_adapters,
            created_by=created_by,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(report.audit_record, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-renewal-queue")
@click.argument(
    "summary_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--policy-report",
    "policy_report_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Lifecycle policy report JSON to add adapter/write follow-up queues.",
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the renewal queue.",
)
def plugins_lifecycle_renewal_queue(
    summary_json: Path,
    policy_report_path: Path | None,
    created_by: str,
) -> None:
    """Emit a deterministic renewal/follow-up queue for lifecycle operations.

    Parameters
    ----------
    summary_json : Path
        Path to the summary JSON.
    policy_report_path : Path | None
        Filesystem path to the policy report.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    summary = _load_lifecycle_summary_from_payload(
        _load_json_file(summary_json, artifact="lifecycle summary")
    )
    policy_payload: dict[str, object] | None = None
    if policy_report_path is not None:
        policy_payload = _load_lifecycle_policy_report_payload(
            _load_json_file(policy_report_path, artifact="lifecycle policy")
        )
        if policy_payload["summary_hash"] != summary.summary_hash:
            raise click.ClickException(
                "lifecycle policy summary_hash does not match lifecycle summary"
            )
        if policy_payload["request_count"] != summary.request_count:
            raise click.ClickException(
                "lifecycle policy request_count does not match lifecycle summary"
            )

    renewal_hashes = tuple(sorted(summary.renewal_required_request_hashes))
    storage_missing_hashes = tuple(sorted(summary.storage_missing_request_hashes))
    missing_adapter_hashes: tuple[str, ...] = ()
    external_followup_hashes: tuple[str, ...] = ()
    if policy_payload is not None:
        missing_adapter_hashes = tuple(
            sorted(
                cast(
                    list[str],
                    policy_payload["missing_adapter_request_hashes"],
                )
            )
        )
        external_followup_hashes = tuple(
            sorted(
                cast(
                    list[str],
                    policy_payload["external_write_followup_request_hashes"],
                )
            )
        )

    queue_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_renewal_queue_v1",
        "version": "1.0.0",
        "summary_hash": summary.summary_hash,
        "request_count": summary.request_count,
        "renewal_required_request_hashes": list(renewal_hashes),
        "storage_missing_request_hashes": list(storage_missing_hashes),
        "missing_adapter_request_hashes": list(missing_adapter_hashes),
        "external_write_followup_request_hashes": list(external_followup_hashes),
        "created_by": created_by,
    }
    queue_payload["queue_hash"] = _record_hash(queue_payload)
    click.echo(json.dumps(queue_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-multistore-dashboard")
@click.argument(
    "policy_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the multi-store dashboard.",
)
def plugins_lifecycle_multistore_dashboard(
    policy_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic aggregate dashboard across policy reports.

    Parameters
    ----------
    policy_json : tuple[Path, ...]
        Path to the policy JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "multi-store dashboard schema mismatch: created_by must be non-empty"
        )
    policies = tuple(
        _load_lifecycle_policy_report_payload(
            _load_json_file(path, artifact="lifecycle policy")
        )
        for path in policy_json
    )
    policy_hashes = tuple(
        sorted(
            _require_sha256(policy["policy_hash"], "policy_hash") for policy in policies
        )
    )
    if len(set(policy_hashes)) != len(policy_hashes):
        raise click.ClickException("duplicate lifecycle policy hash")
    summary_hashes = tuple(
        sorted(
            _require_sha256(policy["summary_hash"], "summary_hash")
            for policy in policies
        )
    )
    unique_requests: set[str] = set()
    action_totals: dict[str, int] = {
        "confirm_external_write": 0,
        "persist_request": 0,
        "register_storage_adapter": 0,
        "renew_approval": 0,
    }
    renewal_required: set[str] = set()
    storage_missing: set[str] = set()
    missing_adapters: set[str] = set()
    external_followup: set[str] = set()

    for policy in policies:
        request_count = policy["request_count"]
        if not isinstance(request_count, int):  # pragma: no cover - loader enforces int
            raise click.ClickException(
                "lifecycle policy schema mismatch: request_count must be an integer"
            )
        unique_requests.update(
            cast(list[str], policy["renewal_required_request_hashes"])
        )
        unique_requests.update(
            cast(list[str], policy["storage_missing_request_hashes"])
        )
        unique_requests.update(
            cast(list[str], policy["missing_adapter_request_hashes"])
        )
        unique_requests.update(
            cast(list[str], policy["external_write_followup_request_hashes"])
        )
        policy_actions = policy.get("policy_action_counts")
        if not isinstance(policy_actions, dict):
            raise click.ClickException(
                "lifecycle policy schema mismatch: policy_action_counts is malformed"
            )
        for key in action_totals:
            value = policy_actions.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise click.ClickException(
                    "lifecycle policy schema mismatch: "
                    "policy_action_counts is malformed"
                )
            action_totals[key] += value
        renewal_required.update(
            cast(list[str], policy["renewal_required_request_hashes"])
        )
        storage_missing.update(
            cast(list[str], policy["storage_missing_request_hashes"])
        )
        missing_adapters.update(
            cast(list[str], policy["missing_adapter_request_hashes"])
        )
        external_followup.update(
            cast(list[str], policy["external_write_followup_request_hashes"])
        )

    dashboard_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_multistore_dashboard_v1",
        "version": "1.0.0",
        "policy_count": len(policies),
        "policy_hashes": list(policy_hashes),
        "summary_hashes": list(summary_hashes),
        "aggregated_policy_action_counts": action_totals,
        "renewal_required_request_hashes": sorted(renewal_required),
        "storage_missing_request_hashes": sorted(storage_missing),
        "missing_adapter_request_hashes": sorted(missing_adapters),
        "external_write_followup_request_hashes": sorted(external_followup),
        "unique_flagged_request_count": len(unique_requests),
        "created_by": created_by,
    }
    dashboard_payload["dashboard_hash"] = _record_hash(dashboard_payload)
    click.echo(json.dumps(dashboard_payload, indent=2, sort_keys=True))


@plugins_group.command("lifecycle-multistore-drilldown")
@click.argument(
    "policy_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Operator or deployment component creating the cross-store drill-down.",
)
def plugins_lifecycle_multistore_drilldown(
    policy_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit deterministic per-store lifecycle queues with provenance hashes.

    Parameters
    ----------
    policy_json : tuple[Path, ...]
        Path to the policy JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not created_by:
        raise click.ClickException(
            "multi-store drilldown schema mismatch: created_by must be non-empty"
        )
    policies = tuple(
        _load_lifecycle_policy_report_payload(
            _load_json_file(path, artifact="lifecycle policy")
        )
        for path in policy_json
    )
    if not policies:
        raise click.ClickException(
            "multi-store drilldown requires at least one policy report"
        )
    policy_hashes = tuple(
        sorted(
            _require_sha256(policy["policy_hash"], "policy_hash") for policy in policies
        )
    )
    if len(set(policy_hashes)) != len(policy_hashes):
        raise click.ClickException("duplicate lifecycle policy hash")

    per_store: list[dict[str, object]] = []
    global_requests: set[str] = set()
    for policy in sorted(
        policies,
        key=lambda item: str(item["policy_hash"]),
    ):
        policy_hash = _require_sha256(policy["policy_hash"], "policy_hash")
        summary_hash = _require_sha256(policy["summary_hash"], "summary_hash")
        request_count = policy["request_count"]
        if (
            not isinstance(request_count, int) or request_count < 1
        ):  # pragma: no cover - loader enforces positive int
            raise click.ClickException(
                "lifecycle policy schema mismatch: "
                "request_count must be a positive integer"
            )
        status_counts = policy.get("status_counts")
        action_counts = policy.get("policy_action_counts")
        if not isinstance(status_counts, dict) or not isinstance(action_counts, dict):
            raise click.ClickException(
                "lifecycle policy schema mismatch: status/action counts are malformed"
            )
        store_payload: dict[str, object] = {
            "policy_hash": policy_hash,
            "summary_hash": summary_hash,
            "request_count": request_count,
            "status_counts": dict(cast(dict[str, int], status_counts)),
            "policy_action_counts": dict(cast(dict[str, int], action_counts)),
            "renewal_required_request_hashes": sorted(
                cast(list[str], policy["renewal_required_request_hashes"])
            ),
            "storage_missing_request_hashes": sorted(
                cast(list[str], policy["storage_missing_request_hashes"])
            ),
            "missing_adapter_request_hashes": sorted(
                cast(list[str], policy["missing_adapter_request_hashes"])
            ),
            "external_write_followup_request_hashes": sorted(
                cast(list[str], policy["external_write_followup_request_hashes"])
            ),
        }
        store_payload["store_hash"] = _record_hash(store_payload)
        global_requests.update(
            cast(list[str], store_payload["renewal_required_request_hashes"])
        )
        global_requests.update(
            cast(list[str], store_payload["storage_missing_request_hashes"])
        )
        global_requests.update(
            cast(list[str], store_payload["missing_adapter_request_hashes"])
        )
        global_requests.update(
            cast(
                list[str],
                store_payload["external_write_followup_request_hashes"],
            )
        )
        per_store.append(store_payload)

    drilldown_payload: dict[str, object] = {
        "schema": "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1",
        "version": "1.0.0",
        "policy_count": len(per_store),
        "policy_hashes": list(policy_hashes),
        "stores": per_store,
        "global_flagged_request_hashes": sorted(global_requests),
        "global_flagged_request_count": len(global_requests),
        "created_by": created_by,
    }
    drilldown_payload["drilldown_hash"] = _record_hash(drilldown_payload)
    click.echo(json.dumps(drilldown_payload, indent=2, sort_keys=True))
