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
        payload = {
            "before": result.before_evidence.to_audit_record(),
            "after": result.after_evidence.to_audit_record(),
            "damping_improved": result.damping_improved,
        }
        Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"\nPRC evidence written to {output}")
