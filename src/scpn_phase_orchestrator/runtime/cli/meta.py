# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI cross-domain meta-transfer command

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

from scpn_phase_orchestrator.meta import CrossDomainMetaTransfer
from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)


@main.command("meta-transfer-manifest")
@click.argument(
    "audit_paths",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--audit-directory",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Nested audit-history directory to discover with --pattern.",
)
@click.option(
    "--pattern",
    default="**/*.jsonl",
    show_default=True,
    help="Glob pattern used with --audit-directory.",
)
@click.option("--min-records", default=1, show_default=True, type=int)
@click.option("--package-name", default="scpn-meta", show_default=True)
@click.option(
    "--import-target",
    default="scpn_phase_orchestrator.meta",
    show_default=True,
)
@click.option("--console-script", default="scpn-meta", show_default=True)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(),
    help="Write manifest JSON to a file instead of stdout.",
)
def meta_transfer_manifest(
    audit_paths: tuple[str, ...],
    audit_directory: str | None,
    pattern: str,
    min_records: int,
    package_name: str,
    import_target: str,
    console_script: str,
    output: str | None,
) -> None:
    """Emit a review-only meta-transfer package manifest from audit history.

    Parameters
    ----------
    audit_paths : tuple[str, ...]
        Audit-log paths to package.
    audit_directory : str | None
        Directory of audit logs, or ``None``.
    pattern : str
        Glob pattern for audit-log discovery.
    min_records : int
        Minimum number of records required.
    package_name : str
        Name for the emitted package.
    import_target : str
        Import target for the generated package.
    console_script : str
        Console-script entry-point name.
    output : str | None
        Destination path, or ``None`` for stdout.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    if min_records < 1:
        raise click.ClickException("--min-records must be at least 1")
    if audit_directory is None and not audit_paths:
        raise click.ClickException(
            "provide one or more audit JSONL files or --audit-directory"
        )
    if audit_directory is not None and audit_paths:
        raise click.ClickException(
            "audit JSONL files and --audit-directory are mutually exclusive"
        )
    try:
        if audit_directory is not None:
            model = CrossDomainMetaTransfer.fit_audit_directory(
                audit_directory,
                pattern=pattern,
                min_records=min_records,
            )
        else:
            model = CrossDomainMetaTransfer.fit_audit_history(
                audit_paths,
                min_records=min_records,
            )
        manifest = model.to_package_manifest(
            package_name=package_name,
            import_target=import_target,
            console_script=console_script,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        raise click.ClickException(str(exc)) from exc

    text = json.dumps(manifest.to_audit_record(), indent=2, sort_keys=True) + "\n"
    if output is None:
        click.echo(text, nl=False)
        return
    Path(output).write_text(text, encoding="utf-8")
    click.echo(f"Meta-transfer package manifest written: {output}")
