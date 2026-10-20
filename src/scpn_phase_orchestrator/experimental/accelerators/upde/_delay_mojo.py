# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for delayed Kuramoto

"""Mojo backend for ``upde/delay.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._delay_validation import (
    validate_delay_backend_inputs,
    validate_delay_backend_output,
)

__all__ = ["_ensure_exe", "delayed_kuramoto_run_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "delay_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/delay.mojo "
            f"-o mojo/delay_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def delayed_kuramoto_run_mojo(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    delay_steps: int,
    n_steps: int,
) -> FloatArray:
    """Integrate via Mojo, returning an independent input copy for zero steps.

    Invalid controls, process failures and invalid numerical results raise
    ValueError. A positive batch requires the compiled executable.

    Parameters
    ----------
    phases, omegas : FloatArray
        Finite source phases in radians and frequencies in rad/s, shape ``(n,)``.
    knm_flat, alpha_flat : FloatArray
        Row-major coupling and phase-lag matrices, each with ``n*n`` entries.
    n : int
        Positive oscillator count matching the supplied buffers.
    zeta, psi : float
        Finite external drive strength and reference phase in radians.
    dt : float
        Positive finite Euler timestep in seconds.
    delay_steps : int
        Positive discrete delay; steps before this delay use current phases.
    n_steps : int
        Nonnegative number of integration steps.

    Returns
    -------
    FloatArray
        Independent float64 final phases. Positive batches project into
        ``[0, 2*pi)`` with positive zero; zero steps preserve the validated input.

    Raises
    ------
    ValueError
        If inputs, native arithmetic or the returned phase vector are invalid.
    ImportError
        If a positive batch cannot load its actual native artifact.
    """
    ph, om, knm, alpha, n, zeta, psi, dt, delay_steps, n_steps = (
        validate_delay_backend_inputs(
            phases, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, delay_steps, n_steps
        )
    )
    if n_steps == 0:
        return ph
    exe = _ensure_exe()
    tokens: list[str] = [
        "DELAY",
        str(int(n)),
        str(min(delay_steps, n_steps)),
        str(int(n_steps)),
        repr(float(zeta)),
        repr(float(psi)),
        repr(float(dt)),
    ]
    tokens.extend(repr(float(x)) for x in ph.tolist())
    tokens.extend(repr(float(x)) for x in om.tolist())
    tokens.extend(repr(float(x)) for x in knm.tolist())
    tokens.extend(repr(float(x)) for x in alpha.tolist())
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(f"Mojo delay exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    if len(lines) != n:
        raise ValueError(f"Mojo DELAY returned {len(lines)} lines, expected {n}")
    try:
        values = [float(line) for line in lines]
    except ValueError as exc:
        raise ValueError("Mojo DELAY output must contain finite phases") from exc
    return validate_delay_backend_output(values, n=n)
