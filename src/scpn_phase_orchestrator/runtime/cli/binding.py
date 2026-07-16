# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI binding validation and inference commands

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import click
import numpy as np

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.autotune.sindy_confidence import SindyConfidencePolicy
from scpn_phase_orchestrator.autotune.sindy_options import SindyOptions
from scpn_phase_orchestrator.binding import (
    format_resolved_binding_config,
    load_binding_spec,
    resolved_binding_config,
    scan_unsafe_patterns,
    validate_binding_spec,
    validate_binding_spec_security,
)
from scpn_phase_orchestrator.coupling.infer import auto_coupling_estimation
from scpn_phase_orchestrator.runtime.cli._app import FloatArray, main
from scpn_phase_orchestrator.studio.workflow import StudioProjectState


@main.command()
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option(
    "--security",
    "security_checks",
    is_flag=True,
    help="Run stricter security linting for production-facing binding specs.",
)
@click.option(
    "--hard",
    "hard_scan",
    is_flag=True,
    help="Also statically scan the domainpack's files for eval/pickle/unsafe-YAML.",
)
def validate(binding_spec: str, security_checks: bool, hard_scan: bool) -> None:
    """Validate a binding specification file.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    security_checks : bool
        Whether to run the stricter security validation pass over the spec.
    hard_scan : bool
        Whether to additionally scan the domainpack's files for dangerous code
        and configuration patterns; implies ``--security``.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    run_security = security_checks or hard_scan
    if run_security:
        errors.extend(validate_binding_spec_security(spec))
    if hard_scan:
        root = Path(binding_spec).resolve().parent
        for finding in scan_unsafe_patterns(root):
            errors.append(
                f"hard scan: {finding.path}:{finding.line} "
                f"[{finding.category}] {finding.snippet}"
            )
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)
    click.echo("Valid")
    if run_security:
        click.echo("Security checks passed")
    if hard_scan:
        click.echo("Hard scan: no dangerous patterns found")
    summary = resolved_binding_config(spec)
    for line in format_resolved_binding_config(summary):
        click.echo(line)


@main.command("inspect")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option("--json-out", is_flag=True, help="Output resolved summary as JSON")
def inspect_binding(binding_spec: str, json_out: bool) -> None:
    """Inspect resolved runtime choices for a binding spec.

    Parameters
    ----------
    binding_spec : str
        Filesystem path to the binding-spec file.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    spec = load_binding_spec(Path(binding_spec))
    errors = validate_binding_spec(spec)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    summary = resolved_binding_config(spec)
    if json_out:
        click.echo(json.dumps(summary, indent=2, sort_keys=True))
        return

    for line in format_resolved_binding_config(summary):
        click.echo(line)


@main.command("auto-bind")
@click.argument(
    "source_kind",
    type=click.Choice(["time-series-csv", "event-log-json", "graph-json"]),
)
@click.argument("source_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--project-name",
    required=True,
    help="Name to embed in the review-only proposed binding spec.",
)
@click.option(
    "--sample-rate-hz",
    type=float,
    default=None,
    help="Sampling rate for time-series CSV sources.",
)
@click.option(
    "--sindy-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="Phase-SINDy sparsity threshold (time-series CSV only).",
)
@click.option(
    "--sindy-min-r-squared",
    type=float,
    default=0.9,
    show_default=True,
    help="Minimum R² before a phase-SINDy fit is called discovered.",
)
@click.option(
    "--sindy-min-samples-per-parameter",
    type=float,
    default=5.0,
    show_default=True,
    help="Minimum derivative samples per parameter for a discovered fit.",
)
@click.option(
    "--emit-equations",
    is_flag=True,
    help="Print the discovered phase dynamics and their confidence verdict.",
)
@click.option("--json-out", is_flag=True, help="Output proposal audit record as JSON")
def auto_bind(
    source_kind: str,
    source_path: str,
    project_name: str,
    sample_rate_hz: float | None,
    sindy_threshold: float,
    sindy_min_r_squared: float,
    sindy_min_samples_per_parameter: float,
    emit_equations: bool,
    json_out: bool,
) -> None:
    """Propose a review-only binding spec from raw local source data.

    Parameters
    ----------
    source_kind : str
        Kind of raw source data (e.g. ``csv`` or ``events``).
    source_path : str
        Filesystem path to the raw source data.
    project_name : str
        Name for the generated project.
    sample_rate_hz : float | None
        Sampling rate in Hz, or ``None`` to infer.
    sindy_threshold : float
        Phase-SINDy sparsity threshold; time-series CSV sources only.
    sindy_min_r_squared : float
        Minimum R² before a phase-SINDy fit is called ``discovered``.
    sindy_min_samples_per_parameter : float
        Minimum derivative samples per parameter for a ``discovered`` fit.
    emit_equations : bool
        Whether to print the discovered phase dynamics and confidence verdict.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    SystemExit
        If the command fails; the error is reported and the process exits non-zero.
    """
    try:
        source_text = Path(source_path).read_text(encoding="utf-8")
        if source_kind == "time-series-csv":
            sindy_options = SindyOptions(
                phase_sindy_threshold=sindy_threshold,
                confidence_policy=SindyConfidencePolicy(
                    min_r_squared=sindy_min_r_squared,
                    min_samples_per_parameter=sindy_min_samples_per_parameter,
                ),
            )
            proposal = propose_binding_from_time_series_csv(
                source_text,
                sample_rate_hz=sample_rate_hz,
                project_name=project_name,
                sindy_options=sindy_options,
            )
        elif source_kind == "event-log-json":
            proposal = propose_binding_from_event_log(
                source_text,
                project_name=project_name,
            )
        else:
            proposal = propose_binding_from_graph(
                source_text,
                project_name=project_name,
            )
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        click.echo(f"ERROR: {exc}", err=True)
        raise SystemExit(1) from exc
    except OSError as exc:
        click.echo(f"ERROR: could not read source file: {exc.strerror}", err=True)
        raise SystemExit(1) from exc

    if json_out:
        click.echo(json.dumps(proposal.to_audit_record(), indent=2, sort_keys=True))
        return
    if emit_equations:
        _echo_discovered_dynamics(proposal)
        return
    click.echo(proposal.binding.yaml_text, nl=False)


def _echo_discovered_dynamics(proposal: StudioProjectState) -> None:
    """Print the discovered phase dynamics and their honest confidence verdict.

    Parameters
    ----------
    proposal : StudioProjectState
        The proposal whose provenance may carry a ``discovered_dynamics`` record;
        source kinds without phase-SINDy discovery carry none.
    """
    record = proposal.binding.provenance.get("discovered_dynamics")
    if not isinstance(record, Mapping):
        click.echo("No discovered phase dynamics for this source kind.")
        return
    confidence = cast(Mapping[str, object], record["confidence"])
    click.echo(f"Discovered dynamics ({record['library']})")
    click.echo(f"  posture: {confidence['posture']}  [tier: {confidence['tier']}]")
    click.echo(f"  content hash: {record['content_hash']}")
    reasons = cast("Sequence[object]", confidence.get("reasons", ()))
    if reasons:
        click.echo("  reasons:")
        for reason in reasons:
            click.echo(f"    - {reason}")
    equations = cast("Sequence[object]", record["equations"])
    if equations:
        click.echo("  equations:")
        for equation in equations:
            click.echo(f"    {equation}")
    edges = cast("Sequence[Mapping[str, object]]", record["coupling_edges"])
    if edges:
        click.echo("  coupling:")
        for edge in edges:
            click.echo(
                f"    {edge['source']} -> {edge['target']}  "
                f"({float(cast(float, edge['coefficient'])):+.4f})"
            )


def _load_phase_series_table(source_path: Path) -> FloatArray:
    """Load a validated phase-series table from a file, else raise."""
    try:
        if source_path.suffix.lower() == ".npy":
            values = np.load(source_path, allow_pickle=False)
        else:
            values = np.loadtxt(source_path, delimiter=",")
    except (OSError, ValueError) as exc:
        raise click.ClickException(
            f"could not read numeric phase-series data: {exc}"
        ) from exc
    series = np.asarray(values, dtype=np.float64)
    if series.ndim != 2:
        raise click.ClickException(
            f"phase-series source must be a 2-D table, got shape {series.shape}"
        )
    return np.ascontiguousarray(series, dtype=np.float64)


@main.command("auto-coupling-estimation")
@click.argument("source_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--orientation",
    type=click.Choice(["oscillator-by-time", "time-by-oscillator"]),
    default="oscillator-by-time",
    show_default=True,
    help="Input table orientation.",
)
@click.option("--n-bins", type=int, default=8, show_default=True)
@click.option("--threshold-quantile", type=float, default=0.75, show_default=True)
@click.option("--threshold-absolute", type=float, default=None)
@click.option(
    "--normalisation",
    type=click.Choice(["max", "none"]),
    default="max",
    show_default=True,
)
@click.option("--json-out", is_flag=True, help="Output JSON audit record")
def auto_coupling_estimation_command(
    source_path: str,
    orientation: str,
    n_bins: int,
    threshold_quantile: float,
    threshold_absolute: float | None,
    normalisation: str,
    json_out: bool,
) -> None:
    """Infer directed K_nm from phase time-series data.

    Parameters
    ----------
    source_path : str
        Filesystem path to the raw source data.
    orientation : str
        Time-series orientation (rows or columns).
    n_bins : int
        Number of histogram bins.
    threshold_quantile : float
        Quantile threshold for edge pruning.
    threshold_absolute : float | None
        Absolute threshold for edge pruning, or ``None``.
    normalisation : str
        Edge-weight normalisation mode.
    json_out : bool
        Whether to print machine-readable JSON output.

    Raises
    ------
    ClickException
        If the inputs are invalid or the operation fails.
    """
    series = _load_phase_series_table(Path(source_path))
    if orientation == "time-by-oscillator":
        series = np.ascontiguousarray(series.T, dtype=np.float64)
    try:
        result = auto_coupling_estimation(
            series,
            n_bins=n_bins,
            threshold_quantile=threshold_quantile,
            threshold_absolute=threshold_absolute,
            normalisation=cast(Literal["max", "none"], normalisation),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc

    record = result.to_audit_record()
    if json_out:
        click.echo(json.dumps(record, indent=2, sort_keys=True))
        return

    click.echo(
        "auto-coupling-estimation "
        f"method={record['method']} orientation={record['orientation']} "
        f"shape={record['shape']} edges={result.edge_count} "
        f"density={result.density:.6g}"
    )
    for row in result.knm:
        click.echo(",".join(f"{value:.17g}" for value in row))
