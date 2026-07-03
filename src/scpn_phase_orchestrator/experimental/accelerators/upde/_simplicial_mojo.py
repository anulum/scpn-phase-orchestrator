# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for simplicial Kuramoto

"""Mojo backend for ``upde/simplicial.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde._simplicial_validation import (
    validate_simplicial_inputs,
    validate_simplicial_output,
)

from .._mojo_runtime import require_mojo_executable, run_mojo_executable

__all__ = ["_ensure_exe", "simplicial_run_mojo"]

FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "simplicial_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/simplicial.mojo -o mojo/simplicial_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def simplicial_run_mojo(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    sigma2: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate pairwise-plus-simplicial Kuramoto dynamics.

    The calculation is delegated to the Mojo backend.
    """
    phases, omegas, knm_flat, alpha_flat, n, zeta, psi, sigma2, dt, n_steps = (
        validate_simplicial_inputs(
            phases,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            zeta,
            psi,
            sigma2,
            dt,
            n_steps,
        )
    )
    if n_steps == 0:
        return phases.copy()
    exe = _ensure_exe()
    tokens: list[str] = [
        "SIMP",
        str(int(n)),
        repr(float(zeta)),
        repr(float(psi)),
        repr(float(sigma2)),
        repr(float(dt)),
        str(int(n_steps)),
    ]
    tokens.extend(repr(float(x)) for x in phases.tolist())
    tokens.extend(repr(float(x)) for x in omegas.tolist())
    tokens.extend(repr(float(x)) for x in knm_flat.tolist())
    tokens.extend(repr(float(x)) for x in alpha_flat.tolist())
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo simplicial exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != n:
        raise ValueError(f"Mojo SIMP returned {len(lines)} lines, expected {n}")
    values: list[float] = []
    for line in lines:
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                "Mojo SIMP output must be finite phases in [0, 2*pi)"
            ) from exc
        if not np.isfinite(value) or value < 0.0 or value >= TWO_PI:
            raise ValueError("Mojo SIMP output must be finite phases in [0, 2*pi)")
        values.append(value)
    result: FloatArray = np.array(values, dtype=np.float64)
    return validate_simplicial_output(result, n=n)
