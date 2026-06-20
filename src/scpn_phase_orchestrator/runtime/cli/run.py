# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI simulation run command

"""Command-line entry point for validation, replay, export, and review workflows.

The CLI wraps public SPO APIs behind explicit commands for binding validation,
inspection, auto-binding proposals, coupling estimation, formal export, replay,
plugin catalogs, scaffolding, and selected runtime utilities. Commands validate
local inputs and emit text or JSON review artifacts; they do not push commits,
start network services, or perform live actuation unless an explicit subcommand
is invoked for that runtime path.
"""

from __future__ import annotations

from pathlib import Path

import click

from scpn_phase_orchestrator.binding import (
    format_resolved_binding_config,
    load_binding_spec,
    resolved_binding_config,
    validate_binding_spec,
)
from scpn_phase_orchestrator.runtime.audit_logger import AuditLogger
from scpn_phase_orchestrator.runtime.cli._app import (
    main,
)
from scpn_phase_orchestrator.runtime.simulation import simulate


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option("--steps", default=100, type=int, help="Simulation steps")
@click.option("--audit", default=None, type=click.Path(), help="Audit log (JSONL)")
@click.option(
    "--audit-stream",
    default=None,
    type=click.Path(),
    help="Audit event stream (length-delimited protobuf)",
)
@click.option("--seed", default=42, type=int, help="RNG seed")
def run(
    binding_spec: str,
    steps: int,
    audit: str | None,
    audit_stream: str | None,
    seed: int,
) -> None:
    """Run simulation from a binding spec.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    steps : int
        Number of simulation steps to run.
    audit : str | None
        Destination audit-log path, or ``None``.
    audit_stream : str | None
        Destination audit-stream path, or ``None``.
    seed : int
        Seed for the deterministic RNG, or ``None``.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    ClickException
        If the inputs are invalid or the operation fails.
    """
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    if spec.safety_tier != "research":
        raise click.ClickException(
            f"safety_tier={spec.safety_tier!r} is not enforced by the local "
            "runtime; use the formal export and certified controller pipeline "
            "before executing non-research specs"
        )
    binding_summary = resolved_binding_config(spec)
    for line in format_resolved_binding_config(binding_summary):
        click.echo(line)

    spec_path = Path(binding_spec)
    audit_logger = (
        AuditLogger(audit, event_stream=audit_stream)
        if audit
        else AuditLogger(
            Path(audit_stream).with_suffix(".jsonl"),
            event_stream=audit_stream,
        )
        if audit_stream
        else None
    )
    try:
        result = simulate(
            spec,
            steps=steps,
            seed=seed,
            policy_enabled=True,
            audit_logger=audit_logger,
            binding_spec_path=spec_path,
        )
        msg = (
            f"R_good={result.r_good:.4f}  "
            f"R_bad={result.r_bad:.4f}  "
            f"regime={result.final_regime}"
        )
        if result.mean_amplitude is not None:
            msg += f"  mean_amplitude={result.mean_amplitude:.4f}"
        click.echo(msg)
    except ValueError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    finally:
        if audit_logger is not None:
            audit_logger.close()
