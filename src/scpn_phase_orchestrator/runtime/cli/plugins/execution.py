# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin execution planning and persistence commands

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
    PluginExecutionApproval,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
    build_plugin_execution_request_storage_manifest,
    build_plugin_marketplace_catalog,
    build_rust_plugin_registry,
    build_rust_plugin_runtime_handoff,
    compatibility_report,
    discover_plugin_manifests,
    write_plugin_execution_request_storage_bundle,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _PLUGIN_KIND_OPTIONS,
    _find_capability,
    _find_discovered_plugin,
    _load_approval_from_payload,
    _load_json_file,
    _load_plan_from_payload,
    _load_request_from_payload,
    _load_revocation_list_from_payload,
    _normalize_approved_target_hashes,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    _build_plugin_execution_request,
    plugins_group,
)


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
