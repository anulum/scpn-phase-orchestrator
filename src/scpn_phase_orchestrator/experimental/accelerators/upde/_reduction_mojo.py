# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Ott-Antonsen reduction

"""Mojo backend for ``upde/reduction.py``."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

from scpn_phase_orchestrator.upde._reduction_validation import (
    validate_oa_inputs,
    validate_oa_output,
)

from .._mojo_runtime import require_mojo_executable, run_mojo_executable

__all__ = ["_ensure_exe", "oa_run_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "reduction_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/reduction.mojo -o mojo/reduction_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def oa_run_mojo(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    """Integrate the Ott-Antonsen reduced dynamics.

    The calculation is delegated to the Mojo backend.
    """
    z_re, z_im, omega_0, delta, k_coupling, dt, n_steps = validate_oa_inputs(
        z_re,
        z_im,
        omega_0,
        delta,
        k_coupling,
        dt,
        n_steps,
    )
    exe = _ensure_exe()
    tokens = [
        "OARUN",
        repr(float(z_re)),
        repr(float(z_im)),
        repr(float(omega_0)),
        repr(float(delta)),
        repr(float(k_coupling)),
        repr(float(dt)),
        str(int(n_steps)),
    ]
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(f"Mojo OARUN exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    if len(lines) != 4:
        raise ValueError(f"Mojo OARUN returned {len(lines)} lines, expected 4")
    try:
        z_real, z_imag, radius, psi = (float(line) for line in lines)
    except ValueError as exc:
        raise ValueError(
            "Mojo OARUN output must contain finite z_real, z_imag, R, and psi values"
        ) from exc
    values = (z_real, z_imag, radius, psi)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            "Mojo OARUN output must contain finite z_real, z_imag, R, and psi values"
        )
    return validate_oa_output(*values)
