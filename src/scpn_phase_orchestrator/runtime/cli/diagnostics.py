# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI environment diagnostics command

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

import click

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.doctor import (
    render_report,
    run_environment_diagnostics,
)


@main.command()
@click.option("--json-out", is_flag=True, help="Output the readiness record as JSON.")
def doctor(json_out: bool) -> None:
    """Check environment readiness: interpreter, required deps, backends, extras.

    Exits non-zero when the interpreter is outside the supported window or a
    required runtime dependency is missing; missing optional accelerators
    (Rust/Julia/Go/Mojo) and feature extras are reported as warnings only.

    Parameters
    ----------
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    report = run_environment_diagnostics()
    if json_out:
        click.echo(json.dumps(report.to_audit_record(), indent=2, sort_keys=True))
    else:
        for line in render_report(report):
            click.echo(line)
    if not report.ok:
        raise SystemExit(report.exit_code)
