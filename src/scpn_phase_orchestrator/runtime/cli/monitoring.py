# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI twin-confidence and chaos monitoring commands

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
from collections.abc import Mapping
from pathlib import Path

import click
import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import (
    load_binding_spec,
)
from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceCalibrator,
    TwinConfidenceSummary,
    phase_order_divergence,
    score_twin_confidence,
    summarise_twin_confidence,
    twin_confidence_prometheus_text,
)
from scpn_phase_orchestrator.runtime.chaos import (
    ChaosFault,
    ChaosSchedule,
    run_resilience_experiment,
)
from scpn_phase_orchestrator.runtime.cli._app import main


def _load_twin_confidence_ticks(
    path: Path,
) -> list[tuple[NDArray[np.float64], ...]]:
    """Load model/observed phase + order ticks from a JSONL file.

    Each non-blank line must be a JSON object with ``model_phases``,
    ``observed_phases``, ``model_order``, and ``observed_order`` arrays.

    Parameters
    ----------
    path : Path
        The JSONL file to read.

    Returns
    -------
    list[tuple[NDArray[np.float64], ...]]
        One ``(model_phases, observed_phases, model_order, observed_order)``
        tuple per tick.

    Raises
    ------
    click.ClickException
        If the file is empty, a line is malformed, or a tick is missing a field.
    """
    ticks: list[tuple[NDArray[np.float64], ...]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise click.ClickException(f"{path}:{line_no}: malformed JSON") from exc
        if not isinstance(record, Mapping):
            raise click.ClickException(f"{path}:{line_no}: tick must be a JSON object")
        try:
            tick = tuple(
                np.asarray(record[field], dtype=np.float64)
                for field in (
                    "model_phases",
                    "observed_phases",
                    "model_order",
                    "observed_order",
                )
            )
        except KeyError as exc:
            raise click.ClickException(
                f"{path}:{line_no}: missing field {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                f"{path}:{line_no}: non-numeric tick field"
            ) from exc
        ticks.append(tick)
    if not ticks:
        raise click.ClickException(f"{path}: no ticks found")
    return ticks


def _twin_confidence_summary_lines(summary: TwinConfidenceSummary) -> list[str]:
    """Return the human-readable twin-confidence summary lines."""
    return [
        f"ticks scored:      {summary.tick_count}",
        f"worst status:      {summary.worst_status}",
        f"latest status:     {summary.latest_status}",
        f"mean confidence:   {summary.mean_confidence:.4f}",
        f"min confidence:    {summary.min_confidence:.4f}",
        f"latest confidence: {summary.latest_confidence:.4f}",
        (
            f"status counts:     healthy={summary.healthy_count} "
            f"warning={summary.warning_count} critical={summary.critical_count}"
        ),
    ]


@main.command(name="twin-confidence")
@click.option(
    "--calibration",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSONL of trusted nominal ticks used to fit the baseline.",
)
@click.option(
    "--observations",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="JSONL of ticks to score against the calibrated baseline.",
)
@click.option("--n-bins", default=36, type=int, help="Phase histogram bins.")
@click.option("--sensitivity", default=3.0, type=float, help="Confidence decay scale.")
@click.option("--warning-confidence", default=0.6, type=float)
@click.option("--critical-confidence", default=0.3, type=float)
@click.option("--band-z", default=3.0, type=float, help="Operating-band multiplier.")
@click.option("--json-out", is_flag=True, help="Emit the summary as JSON.")
@click.option("--prometheus", is_flag=True, help="Emit Prometheus exposition text.")
@click.option(
    "--fail-on-critical",
    is_flag=True,
    help="Exit non-zero when the worst scored status is critical.",
)
def twin_confidence(
    calibration: Path,
    observations: Path,
    n_bins: int,
    sensitivity: float,
    warning_confidence: float,
    critical_confidence: float,
    band_z: float,
    json_out: bool,
    prometheus: bool,
    fail_on_critical: bool,
) -> None:
    """Score digital-twin confidence over an observation stream.

    Fits a nominal baseline from the calibration JSONL, scores each observation
    tick (phase Jensen-Shannon divergence + order Wasserstein-1 against the
    baseline), and reports the operator-facing summary as human lines, JSON, or
    Prometheus text.

    Parameters
    ----------
    calibration : Path
        JSONL of trusted nominal ticks.
    observations : Path
        JSONL of ticks to score.
    n_bins : int
        Phase histogram bins.
    sensitivity : float
        Confidence decay scale.
    warning_confidence, critical_confidence : float
        Operator status thresholds.
    band_z : float
        Operating-band normal-quantile multiplier.
    json_out, prometheus : bool
        Output-format switches.
    fail_on_critical : bool
        Whether to exit non-zero on a critical worst status.

    Raises
    ------
    SystemExit
        If ``--fail-on-critical`` is set and the worst scored status is critical.
    """
    try:
        calibrator = TwinConfidenceCalibrator(band_z=band_z)
        for tick in _load_twin_confidence_ticks(calibration):
            calibrator.observe(phase_order_divergence(*tick, n_bins=n_bins))
        baseline = calibrator.baseline()
        scores = [
            score_twin_confidence(
                phase_order_divergence(*tick, n_bins=n_bins),
                baseline,
                sensitivity=sensitivity,
                warning_confidence=warning_confidence,
                critical_confidence=critical_confidence,
            )
            for tick in _load_twin_confidence_ticks(observations)
        ]
        summary = summarise_twin_confidence(scores)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if prometheus:
        click.echo(twin_confidence_prometheus_text(summary), nl=False)
    elif json_out:
        click.echo(json.dumps(summary.to_audit_record(), indent=2, sort_keys=True))
    else:
        for line in _twin_confidence_summary_lines(summary):
            click.echo(line)
    if fail_on_critical and summary.worst_status == "critical":
        raise SystemExit(2)


def _parse_chaos_fault(spec: str) -> ChaosFault:
    """Parse a ``kind:start:duration:magnitude`` fault specifier.

    Parameters
    ----------
    spec : str
        Colon-separated fault specifier.

    Returns
    -------
    ChaosFault
        The parsed fault.

    Raises
    ------
    click.ClickException
        If the specifier is malformed or the fault parameters are invalid.
    """
    parts = spec.split(":")
    if len(parts) != 4:
        raise click.ClickException(
            f"fault {spec!r} must be 'kind:start:duration:magnitude'"
        )
    kind, start, duration, magnitude = parts
    try:
        return ChaosFault(
            kind=kind,
            start_step=int(start),
            duration_steps=int(duration),
            magnitude=float(magnitude),
        )
    except (ValueError, TypeError) as exc:
        raise click.ClickException(f"invalid fault {spec!r}: {exc}") from exc


@main.command(name="chaos")
@click.argument("binding_spec", type=click.Path(exists=True))
@click.option(
    "--fault",
    "faults",
    multiple=True,
    required=True,
    help="Fault as 'kind:start:duration:magnitude'; repeatable.",
)
@click.option("--steps", default=200, type=int, help="Simulation steps.")
@click.option("--seed", default=42, type=int, help="Shared RNG seed.")
@click.option("--recovery-tolerance", default=0.05, type=float)
@click.option("--json-out", is_flag=True, help="Emit the result as JSON.")
@click.option(
    "--fail-unrecovered",
    is_flag=True,
    help="Exit non-zero when the perturbed run does not recover.",
)
def chaos(
    binding_spec: str,
    faults: tuple[str, ...],
    steps: int,
    seed: int,
    recovery_tolerance: float,
    json_out: bool,
    fail_unrecovered: bool,
) -> None:
    """Inject faults into a binding spec and score its resilience.

    Runs the spec once nominally and once with the injected fault schedule, then
    reports recovery time, peak coherence drop, stability-margin erosion, and the
    final deviation. The runs are review-only and never actuate.

    Parameters
    ----------
    binding_spec : str
        Path to the binding spec YAML.
    faults : tuple[str, ...]
        Fault specifiers ``kind:start:duration:magnitude``.
    steps : int
        Simulation steps.
    seed : int
        Shared RNG seed.
    recovery_tolerance : float
        Recovery tolerance on ``|nominal_R - perturbed_R|``.
    json_out : bool
        Whether to emit JSON.
    fail_unrecovered : bool
        Whether to exit non-zero when the run does not recover.

    Raises
    ------
    SystemExit
        If ``--fail-unrecovered`` is set and the perturbed run did not recover.
    """
    schedule = ChaosSchedule(faults=tuple(_parse_chaos_fault(item) for item in faults))
    try:
        result = run_resilience_experiment(
            load_binding_spec(Path(binding_spec)),
            schedule,
            steps=steps,
            seed=seed,
            recovery_tolerance=recovery_tolerance,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    metrics = result.metrics
    if json_out:
        click.echo(json.dumps(result.to_audit_record(), indent=2, sort_keys=True))
    else:
        click.echo(f"spec:              {result.spec_name}")
        click.echo(f"faults:            {len(schedule.faults)}")
        click.echo(f"recovered:         {metrics.recovered}")
        click.echo(f"recovery steps:    {metrics.recovery_steps}")
        click.echo(f"max coherence drop:{metrics.max_coherence_drop:.4f}")
        click.echo(f"margin erosion:    {metrics.stability_margin_erosion:.4f}")
        click.echo(f"final deviation:   {metrics.final_deviation:.4f}")
        click.echo(
            f"nominal/perturbed R: {result.nominal_final_r:.4f}"
            f" / {result.perturbed_final_r:.4f}"
        )
    if fail_unrecovered and not metrics.recovered:
        raise SystemExit(2)
