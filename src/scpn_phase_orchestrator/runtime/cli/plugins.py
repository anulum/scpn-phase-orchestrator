# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugins commands

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

from scpn_phase_orchestrator import plugins as plugin_api
from scpn_phase_orchestrator.plugins import (
    PluginExecutionApproval,
    PluginExecutionPlan,
    PluginExecutionRequestRevocationList,
    PluginExecutionRequestStorageManifest,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request_lifecycle_policy_report,
    build_plugin_execution_request_lifecycle_record,
    build_plugin_execution_request_lifecycle_summary,
    build_plugin_execution_request_revocation,
    build_plugin_execution_request_revocation_list,
    build_plugin_execution_request_storage_adapter_manifest,
    build_plugin_execution_request_storage_manifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
    compatibility_report,
    discover_plugin_manifests,
    write_plugin_execution_request_storage_bundle,
)
from scpn_phase_orchestrator.plugins import registry as plugin_registry
from scpn_phase_orchestrator.runtime.cli import binding as binding
from scpn_phase_orchestrator.runtime.cli import diagnostics as diagnostics
from scpn_phase_orchestrator.runtime.cli import monitoring as monitoring
from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _PLUGIN_KIND_OPTIONS,
    _find_capability,
    _find_discovered_plugin,
    _load_approval_from_payload,
    _load_json_file,
    _load_lifecycle_from_payload,
    _load_lifecycle_multistore_drilldown_payload,
    _load_lifecycle_policy_report_payload,
    _load_lifecycle_remediation_action_status_payload,
    _load_lifecycle_remediation_deployment_handoff_payload,
    _load_lifecycle_remediation_execution_dashboard_payload,
    _load_lifecycle_remediation_plan_payload,
    _load_lifecycle_remediation_scheduler_acknowledgement_payload,
    _load_lifecycle_remediation_scheduler_adapter_handoff_payload,
    _load_lifecycle_remediation_scheduler_queue_payload,
    _load_lifecycle_remediation_scheduler_telemetry_payload,
    _load_lifecycle_summary_from_payload,
    _load_plan_from_payload,
    _load_request_from_payload,
    _load_revocation_from_payload,
    _load_revocation_list_from_payload,
    _load_storage_adapter_from_payload,
    _load_storage_manifest_from_payload,
    _normalize_approved_target_hashes,
    _record_hash,
    _require_sha256,
)


def _build_plugin_execution_request(
    plan: PluginExecutionPlan,
    approval: PluginExecutionApproval,
) -> object:
    builder_candidates = (
        "build_plugin_execution_request",
        "build_plugin_execution_request_from_approval",
        "build_plugin_execution_request_from_plan_and_approval",
    )
    for name in builder_candidates:
        for module in (plugin_registry, plugin_api):
            candidate = getattr(module, name, None)
            if not callable(candidate):
                continue
            try:
                return candidate(plan, approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approval=approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approved_execution=approval)
            except TypeError:
                pass
            try:
                return candidate(plan=plan, approval_record=approval)
            except TypeError:
                pass

    raise click.ClickException(
        "registry request builder not available: expected "
        "build_plugin_execution_request"
    )


@main.group("plugins")
def plugins_group() -> None:
    """Inspect extension plugin manifests."""


@plugins_group.command("catalog")
@click.option(
    "--include-incompatible",
    is_flag=True,
    help="Include incompatible manifests and rejection reasons in the output",
)
@click.option(
    "--rust-registry",
    is_flag=True,
    help="Emit flattened Rust-facing capability registry JSON",
)
@click.option(
    "--rust-runtime-handoff",
    is_flag=True,
    help="Emit guarded Rust runtime handoff JSON with loading disabled",
)
def plugins_catalog(
    include_incompatible: bool,
    rust_registry: bool,
    rust_runtime_handoff: bool,
) -> None:
    """Print the discovered plugin marketplace catalogue as JSON.

    Parameters
    ----------
    include_incompatible : bool
        Whether to include incompatible plugins in the catalogue.
    rust_registry : bool
        Whether to include the Rust plugin registry.
    rust_runtime_handoff : bool
        Whether to include the Rust runtime handoff.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if rust_registry and rust_runtime_handoff:
        raise click.ClickException(
            "--rust-registry and --rust-runtime-handoff are mutually exclusive"
        )
    manifests = discover_plugin_manifests()
    if rust_runtime_handoff:
        builder = build_rust_plugin_runtime_handoff
    elif rust_registry:
        builder = build_rust_plugin_registry
    else:
        builder = build_plugin_marketplace_catalog
    catalog = builder(manifests, include_incompatible=include_incompatible)
    click.echo(json.dumps(catalog, indent=2, sort_keys=True))


@plugins_group.command("plan-execution")
@click.argument("plugin_name")
@click.argument("kind", type=click.Choice(_PLUGIN_KIND_OPTIONS))
@click.argument("capability_name")
@click.option(
    "--approved-target-hash",
    "approved_target_hashes",
    multiple=True,
    help="Approved runtime target hash(es) for this execution planning decision.",
)
@click.option(
    "--require-target-hash-approval",
    is_flag=True,
    help="Fail unless the discovered capability target hash is approved.",
)
def plugins_plan_execution(
    plugin_name: str,
    kind: str,
    capability_name: str,
    approved_target_hashes: tuple[str, ...],
    require_target_hash_approval: bool,
) -> None:
    """Emit a non-executing plan for a discovered plugin capability.

    Parameters
    ----------
    plugin_name : str
        Name of the plugin.
    kind : str
        Plugin capability kind.
    capability_name : str
        Name of the plugin capability.
    approved_target_hashes : tuple[str, ...]
        Approved target hashes for the capability.
    require_target_hash_approval : bool
        Whether target-hash approval is required.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    manifests = discover_plugin_manifests()
    manifest = _find_discovered_plugin(manifests, plugin_name)
    compatibility = compatibility_report(manifest)
    capability = _find_capability(manifest, kind, capability_name)
    normalized_hashes = _normalize_approved_target_hashes(approved_target_hashes)

    try:
        plan = build_plugin_execution_plan(
            manifest,
            capability.kind,
            capability_name,
            policy=PluginRuntimeExecutionPolicy(
                loading_permitted=True,
                execution_permitted=True,
                approved_target_hashes=normalized_hashes,
                require_target_hash_approval=require_target_hash_approval,
            ),
        )
    except (LookupError, PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {
        **plan.audit_record,
        "manifest": manifest.to_audit_record(),
        "capability": {
            "kind": capability.kind,
            "name": capability.name,
            "target": capability.target,
            "version": capability.version,
            "channels": list(capability.channels),
            "knobs": list(capability.knobs),
        },
        "compatible": compatibility.compatible,
        "compatibility_reasons": list(compatibility.reasons),
    }
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("approve-execution-plan")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--operator-id",
    required=True,
    type=str,
    help="Operator identity approving the plan",
)
@click.option(
    "--approval-reference",
    required=True,
    type=str,
    help="Reference for the approval decision",
)
@click.option(
    "--approval-reason",
    required=True,
    type=str,
    help="Human reason for this approval",
)
def plugins_approve_execution_plan(
    plan_json: Path,
    operator_id: str,
    approval_reference: str,
    approval_reason: str,
) -> None:
    """Emit a deterministic operator approval artefact for a stored execution plan.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    operator_id : str
        Identifier of the operator.
    approval_reference : str
        External approval reference.
    approval_reason : str
        Reason recorded with the approval.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if not operator_id:
        raise click.ClickException("operator identity is required")
    if not approval_reference:
        raise click.ClickException("approval reference is required")
    if not approval_reason:
        raise click.ClickException("approval reason is required")

    plan_payload = _load_json_file(plan_json)
    plan, _audit_record = _load_plan_from_payload(plan_payload)
    try:
        approval = build_plugin_execution_approval(
            plan,
            operator_identity=operator_id,
            approval_reference=approval_reference,
            approval_reason=approval_reason,
        )
    except (LookupError, PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(approval.audit_record, indent=2, sort_keys=True))


@plugins_group.command("request-execution")
@click.argument(
    "plan_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "approval_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def plugins_request_execution(plan_json: Path, approval_json: Path) -> None:
    """Emit a deterministic execution request from a stored plan and approval.

    Parameters
    ----------
    plan_json : Path
        Path to the plan JSON file.
    approval_json : Path
        Path to the approval JSON file.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    plan_payload = _load_json_file(plan_json, artifact="plan")
    plan, _ = _load_plan_from_payload(plan_payload)
    approval_payload = _load_json_file(approval_json, artifact="approval")
    approval = _load_approval_from_payload(approval_payload)

    if plan.plan_hash != approval.plan_hash:
        raise click.ClickException("plan hash mismatch")
    if plan.target_hash != approval.target_hash:
        raise click.ClickException("target hash mismatch")
    if approval.plugin != plan.manifest.name:
        raise click.ClickException("plugin mismatch between plan and approval")
    if approval.kind != plan.capability.kind:
        raise click.ClickException("kind mismatch between plan and approval")
    if approval.name != plan.capability.name:
        raise click.ClickException("name mismatch between plan and approval")
    if not approval.approved:
        raise click.ClickException("approval is not approved")
    if approval.approved is not True or approval.execution_permitted is not True:
        raise click.ClickException("approval does not permit execution")

    try:
        request = _build_plugin_execution_request(plan, approval)
    except (PermissionError, TypeError, ValueError, KeyError, LookupError) as exc:
        raise click.ClickException(str(exc)) from exc

    if isinstance(request, PluginExecutionApproval):
        payload = request.audit_record
    else:
        payload = cast(dict[str, object], getattr(request, "audit_record", request))
    click.echo(json.dumps(payload, indent=2, sort_keys=True))


@plugins_group.command("persist-execution-request")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-uri",
    required=True,
    help="Deployment-owned URI for the persisted request bundle.",
)
@click.option(
    "--storage-backend",
    default="local_file",
    show_default=True,
    help="Storage backend identifier; local writes require local_file.",
)
@click.option(
    "--retention-policy",
    default="retain_until_revoked",
    show_default=True,
    help="Retention policy identifier for the request bundle.",
)
@click.option(
    "--created-by",
    required=True,
    help="Deployment component creating the request bundle.",
)
@click.option(
    "--revoked-request-hash",
    "revoked_request_hashes",
    multiple=True,
    help="Revoked request hash to bind into the storage manifest.",
)
@click.option(
    "--revocation-list",
    "revocation_list_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Aggregate revocation-list JSON to bind into the storage manifest.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Allow replacing an existing local request bundle.",
)
def plugins_persist_execution_request(
    request_json: Path,
    output_path: Path,
    storage_uri: str,
    storage_backend: str,
    retention_policy: str,
    created_by: str,
    revoked_request_hashes: tuple[str, ...],
    revocation_list_path: Path | None,
    overwrite: bool,
) -> None:
    """Persist a validated execution request as a local storage bundle.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    output_path : Path
        Destination path for the artefact.
    storage_uri : str
        Storage URI for the request bundle.
    storage_backend : str
        Storage backend identifier.
    retention_policy : str
        Retention policy label.
    created_by : str
        Identifier of the creating actor.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.
    revocation_list_path : Path | None
        Filesystem path to the revocation list.
    overwrite : bool
        Whether to overwrite an existing artefact.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)
    direct_revocations = _normalize_approved_target_hashes(revoked_request_hashes)
    revocation_list_hashes: tuple[str, ...] = ()

    try:
        if revocation_list_path is not None:
            revocation_list = _load_revocation_list_from_payload(
                _load_json_file(revocation_list_path, artifact="revocation list")
            )
            revocation_list_hashes = revocation_list.as_revoked_request_hashes()
        normalized_revocations = tuple(
            dict.fromkeys((*direct_revocations, *revocation_list_hashes))
        )
        storage_manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri=storage_uri,
            storage_backend=storage_backend,
            retention_policy=retention_policy,
            created_by=created_by,
            revoked_request_hashes=normalized_revocations,
        )
        bundle = write_plugin_execution_request_storage_bundle(
            request,
            storage_manifest,
            output_path,
            overwrite=overwrite,
        )
    except (OSError, PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(bundle, indent=2, sort_keys=True))


@plugins_group.command("storage-adapter-manifest")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--storage-uri",
    required=True,
    help="Deployment-owned URI for the request storage target.",
)
@click.option(
    "--storage-backend",
    required=True,
    help=(
        "Storage backend identifier: local_file, s3_object, gcs_object, "
        "azure_blob, oci_object, or https_api."
    ),
)
@click.option(
    "--retention-policy",
    default="retain_until_revoked",
    show_default=True,
    help="Retention policy identifier for the request bundle.",
)
@click.option(
    "--created-by",
    required=True,
    help="Deployment component creating the adapter manifest.",
)
@click.option(
    "--revoked-request-hash",
    "revoked_request_hashes",
    multiple=True,
    help="Revoked request hash to bind into the storage manifest.",
)
@click.option(
    "--revocation-list",
    "revocation_list_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Aggregate revocation-list JSON to bind into the storage manifest.",
)
def plugins_storage_adapter_manifest(
    request_json: Path,
    storage_uri: str,
    storage_backend: str,
    retention_policy: str,
    created_by: str,
    revoked_request_hashes: tuple[str, ...],
    revocation_list_path: Path | None,
) -> None:
    """Emit a deterministic storage-adapter handoff manifest without writing.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    storage_uri : str
        Storage URI for the request bundle.
    storage_backend : str
        Storage backend identifier.
    retention_policy : str
        Retention policy label.
    created_by : str
        Identifier of the creating actor.
    revoked_request_hashes : tuple[str, ...]
        Hashes of revoked execution requests.
    revocation_list_path : Path | None
        Filesystem path to the revocation list.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)
    direct_revocations = _normalize_approved_target_hashes(revoked_request_hashes)
    revocation_list_hashes: tuple[str, ...] = ()

    try:
        if revocation_list_path is not None:
            revocation_list = _load_revocation_list_from_payload(
                _load_json_file(revocation_list_path, artifact="revocation list")
            )
            revocation_list_hashes = revocation_list.as_revoked_request_hashes()
        normalized_revocations = tuple(
            dict.fromkeys((*direct_revocations, *revocation_list_hashes))
        )
        storage_manifest = build_plugin_execution_request_storage_manifest(
            request,
            storage_uri=storage_uri,
            storage_backend=storage_backend,
            retention_policy=retention_policy,
            created_by=created_by,
            revoked_request_hashes=normalized_revocations,
        )
        adapter_manifest = build_plugin_execution_request_storage_adapter_manifest(
            request,
            storage_manifest,
        )
    except (PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(adapter_manifest.audit_record, indent=2, sort_keys=True))


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
        if not isinstance(request_count, int):
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
        if not isinstance(request_count, int) or request_count < 1:
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


@plugins_group.command("lifecycle-remediation-scheduler-queue")
@click.argument(
    "handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--window-start-epoch",
    required=True,
    type=int,
    help="Scheduler window start as Unix epoch seconds (UTC).",
)
@click.option(
    "--window-duration-seconds",
    default=3600,
    show_default=True,
    type=int,
    help="Scheduler execution window length in seconds.",
)
@click.option(
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


@plugins_group.command("lifecycle-remediation-scheduler-telemetry")
@click.argument(
    "scheduler_queue_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "action_status_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--as-of-epoch",
    required=True,
    type=int,
    help="Telemetry snapshot epoch seconds (UTC).",
)
@click.option(
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


@plugins_group.command("lifecycle-remediation-scheduler-adapter-handoff")
@click.argument(
    "scheduler_telemetry_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--adapter-name",
    required=True,
    help="External scheduler adapter name.",
)
@click.option(
    "--adapter-endpoint",
    required=True,
    help="External scheduler adapter endpoint identifier.",
)
@click.option(
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


@plugins_group.command("lifecycle-remediation-scheduler-acknowledgement")
@click.argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("entry_hash")
@click.option(
    "--state",
    required=True,
    type=click.Choice(["in_progress", "completed", "blocked"]),
    help="External scheduler execution state.",
)
@click.option(
    "--acknowledged-by",
    required=True,
    help="Actor or component acknowledging execution.",
)
@click.option(
    "--external-reference",
    required=True,
    help="External scheduler job/task reference.",
)
@click.option(
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


@plugins_group.command("lifecycle-remediation-scheduler-acknowledgement-replay")
@click.argument(
    "adapter_handoff_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "acknowledgement_json",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
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


@plugins_group.command("lifecycle-remediation-scheduler-execution-dashboard")
@click.argument(
    "scheduler_telemetry_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "acknowledgement_replay_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
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


@plugins_group.command("revoke-execution-request")
@click.argument(
    "request_json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--revoked-by",
    required=True,
    help="Operator or deployment component revoking the request.",
)
@click.option(
    "--revocation-reference",
    required=True,
    help="Reference for the revocation decision.",
)
@click.option(
    "--revocation-reason",
    required=True,
    help="Human reason for revoking the request.",
)
def plugins_revoke_execution_request(
    request_json: Path,
    revoked_by: str,
    revocation_reference: str,
    revocation_reason: str,
) -> None:
    """Emit a deterministic revocation artefact for an execution request.

    Parameters
    ----------
    request_json : Path
        Path to the request JSON file.
    revoked_by : str
        Identifier of the revoking actor.
    revocation_reference : str
        External revocation reference.
    revocation_reason : str
        Reason recorded with the revocation.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    request_payload = _load_json_file(request_json, artifact="request")
    request = _load_request_from_payload(request_payload)

    try:
        revocation = build_plugin_execution_request_revocation(
            request,
            revoked_by=revoked_by,
            revocation_reference=revocation_reference,
            revocation_reason=revocation_reason,
        )
    except (PermissionError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(revocation.audit_record, indent=2, sort_keys=True))


@plugins_group.command("revocation-list")
@click.argument(
    "revocation_json",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--created-by",
    required=True,
    help="Deployment component creating the revocation list.",
)
def plugins_revocation_list(
    revocation_json: tuple[Path, ...],
    created_by: str,
) -> None:
    """Emit a deterministic aggregate revocation list.

    Parameters
    ----------
    revocation_json : tuple[Path, ...]
        Path to the revocation JSON file.
    created_by : str
        Identifier of the creating actor.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    revocations = tuple(
        _load_revocation_from_payload(_load_json_file(path, artifact="revocation"))
        for path in revocation_json
    )

    try:
        revocation_list = build_plugin_execution_request_revocation_list(
            revocations,
            created_by=created_by,
        )
    except (TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(json.dumps(revocation_list.audit_record, indent=2, sort_keys=True))
