# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI Koopman-MPC oscillation-damping command

"""Command-line entry point for the closed-loop Koopman-MPC damping pipeline.

``spo koopman-mpc`` builds an underdamped oscillator, detects and screens its
poorly-damped mode, damps it in closed loop with the Koopman MPC, re-screens the
controlled ringdown, and reports the before/after damping with both hash-sealed
PRC evidence records. It is review-only and offline; it performs no live
actuation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import click
import numpy as np

from scpn_phase_orchestrator.runtime.cli._app import main
from scpn_phase_orchestrator.runtime.dvoc_oscillation_damping import (
    damp_oscillation,
    underdamped_oscillator,
)
from scpn_phase_orchestrator.runtime.ibr_ride_through import (
    screen_ibr_ride_through_csv,
)
from scpn_phase_orchestrator.runtime.pmu_ringdown import screen_pmu_ringdown_csv


@main.command("koopman-mpc")
@click.option("--frequency-hz", default=0.5, type=float, help="Natural frequency")
@click.option("--damping-ratio", default=0.02, type=float, help="Open-loop damping")
@click.option("--dt", default=0.02, type=float, help="Sampling interval (s)")
@click.option("--horizon", default=300, type=int, help="Ringdown steps")
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the before/after PRC evidence here as JSON",
)
def koopman_mpc(
    frequency_hz: float,
    damping_ratio: float,
    dt: float,
    horizon: int,
    output: str | None,
) -> None:
    """Damp an underdamped oscillator with Koopman MPC and screen the result.

    Parameters
    ----------
    frequency_hz : float
        Natural frequency of the demonstration oscillator.
    damping_ratio : float
        Open-loop damping ratio (a small value is poorly damped).
    dt : float
        Sampling interval in seconds.
    horizon : int
        Number of ringdown steps simulated for each pass.
    output : str | None
        Optional path for the before/after PRC evidence JSON.

    Raises
    ------
    ClickException
        If the oscillator parameters are invalid.
    """
    try:
        state_matrix, input_matrix = underdamped_oscillator(
            frequency_hz=frequency_hz, damping_ratio=damping_ratio, dt=dt
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    result = damp_oscillation(
        state_matrix,
        input_matrix,
        initial_state=np.array([1.0, 0.0]),
        horizon=horizon,
        fs=1.0 / dt,
        captured_at=datetime.now(UTC).isoformat(),
    )

    before_flagged = sum(f.flagged for f in result.before_evidence.findings)
    after_flagged = sum(f.flagged for f in result.after_evidence.findings)
    click.echo("=== Koopman-MPC oscillation damping ===")
    click.echo(
        f"plant: f={frequency_hz} Hz  open-loop zeta={damping_ratio} "
        f"(fit residual {result.fit_residual:.2e})"
    )
    click.echo(
        f"[before] weakest damping={result.uncontrolled_damping_ratio:.4f}  "
        f"flagged={before_flagged}/{len(result.before_evidence.findings)}"
    )
    click.echo(
        f"[after ] weakest damping={result.controlled_damping_ratio:.4f}  "
        f"flagged={after_flagged}/{len(result.after_evidence.findings)}"
    )
    click.echo(f"damping improved: {result.damping_improved}")

    if output is not None:
        payload = result.to_audit_record()
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nPRC evidence written to {output}")


@main.command("pmu-ringdown")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--event-id", required=True, help="Operator event identifier")
@click.option("--captured-at", required=True, help="Capture timestamp for evidence")
@click.option("--signal-source", required=True, help="PMU or historian signal label")
@click.option("--time-column", default="time_s", help="Timestamp column in seconds")
@click.option(
    "--frequency-column",
    default="frequency_hz",
    help="Measured frequency column in hertz",
)
@click.option(
    "--nominal-frequency-hz",
    default=60.0,
    type=float,
    help="Nominal grid frequency subtracted before screening",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the sealed PMU PRC evidence JSON here",
)
def pmu_ringdown(
    csv_path: str,
    event_id: str,
    captured_at: str,
    signal_source: str,
    time_column: str,
    frequency_column: str,
    nominal_frequency_hz: float,
    output: str | None,
) -> None:
    """Screen an operator PMU frequency ringdown CSV for PRC review evidence.

    Parameters
    ----------
    csv_path : str
        PMU CSV path with timestamp and measured-frequency columns.
    event_id : str
        Operator event identifier stamped into the evidence.
    captured_at : str
        Capture timestamp stamped into the evidence.
    signal_source : str
        PMU or historian signal label.
    time_column : str
        CSV timestamp column in seconds.
    frequency_column : str
        CSV measured-frequency column in hertz.
    nominal_frequency_hz : float
        Nominal grid frequency subtracted before mode estimation.
    output : str | None
        Optional destination for the sealed PMU PRC evidence JSON.

    Raises
    ------
    ClickException
        If the PMU CSV or screening controls are invalid.
    """
    try:
        evidence = screen_pmu_ringdown_csv(
            Path(csv_path),
            event_id=event_id,
            captured_at=captured_at,
            signal_source=signal_source,
            time_column=time_column,
            frequency_column=frequency_column,
            nominal_frequency_hz=nominal_frequency_hz,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    flagged = sum(finding.flagged for finding in evidence.prc_evidence.findings)
    finding_count = len(evidence.prc_evidence.findings)
    click.echo("=== PMU ringdown PRC screening ===")
    click.echo(
        f"source: {evidence.signal_source}  samples={evidence.sample_count}  "
        f"fs={evidence.sampling_rate_hz:.4f} Hz"
    )
    click.echo(f"flagged={flagged}/{finding_count}")
    click.echo(f"source sha256: {evidence.source_sha256}")

    if output is not None:
        payload = evidence.to_audit_record()
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nPMU PRC evidence written to {output}")


@main.command("ibr-ride-through")
@click.argument("csv_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--event-id", required=True, help="Operator event identifier")
@click.option("--captured-at", required=True, help="Capture timestamp for evidence")
@click.option("--signal-source", required=True, help="IBR measurement source label")
@click.option(
    "--ibr-category",
    default="other_ibr",
    type=click.Choice(["ac_wind", "other_ibr"]),
    help="PRC-029 voltage ride-through table selector",
)
@click.option("--time-column", default="time_s", help="Timestamp column in seconds")
@click.option("--voltage-column", default="voltage_pu", help="Voltage column in p.u.")
@click.option(
    "--frequency-column",
    default="frequency_hz",
    help="Measured frequency column in hertz",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Write the sealed PRC-029 ride-through evidence JSON here",
)
def ibr_ride_through(
    csv_path: str,
    event_id: str,
    captured_at: str,
    signal_source: str,
    ibr_category: str,
    time_column: str,
    voltage_column: str,
    frequency_column: str,
    output: str | None,
) -> None:
    """Screen an IBR voltage/frequency CSV for PRC-029 review evidence.

    Parameters
    ----------
    csv_path : str
        CSV path with timestamp, voltage, and measured-frequency columns.
    event_id : str
        Operator event identifier stamped into the evidence.
    captured_at : str
        Capture timestamp stamped into the evidence.
    signal_source : str
        IBR measurement source label.
    ibr_category : str
        PRC-029 voltage table selector.
    time_column : str
        CSV timestamp column in seconds.
    voltage_column : str
        CSV voltage column in per unit.
    frequency_column : str
        CSV measured-frequency column in hertz.
    output : str | None
        Optional destination for the sealed PRC-029 ride-through evidence JSON.

    Raises
    ------
    ClickException
        If the CSV or screening controls are invalid.
    """
    try:
        evidence = screen_ibr_ride_through_csv(
            Path(csv_path),
            event_id=event_id,
            captured_at=captured_at,
            signal_source=signal_source,
            ibr_category=ibr_category,
            time_column=time_column,
            voltage_column=voltage_column,
            frequency_column=frequency_column,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    flagged = sum(finding.flagged for finding in evidence.prc029_evidence.findings)
    finding_count = len(evidence.prc029_evidence.findings)
    click.echo("=== IBR ride-through PRC-029 screening ===")
    click.echo(
        f"source: {evidence.signal_source}  samples={evidence.sample_count}  "
        f"duration={evidence.duration_s:.4f} s"
    )
    click.echo(f"flagged={flagged}/{finding_count}")
    click.echo(f"source sha256: {evidence.source_sha256}")

    if output is not None:
        payload = evidence.to_audit_record()
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nIBR ride-through evidence written to {output}")
