# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI plugin execution-request revocation commands

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
    build_plugin_execution_request_revocation,
    build_plugin_execution_request_revocation_list,
)
from scpn_phase_orchestrator.runtime.cli._payloads import (
    _load_json_file,
    _load_request_from_payload,
    _load_revocation_from_payload,
)
from scpn_phase_orchestrator.runtime.cli.plugins._group import (
    plugins_group,
)


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
