# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin storage-adapter manifest command

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

from scpn_phase_orchestrator.plugins import (
    build_plugin_execution_request_storage_adapter_manifest,
    build_plugin_execution_request_storage_manifest,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _load_request_from_payload,
    _load_revocation_list_from_payload,
    _normalize_approved_target_hashes,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group,
)


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
