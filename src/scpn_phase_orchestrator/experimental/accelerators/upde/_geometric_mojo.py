# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for torus geometric integrator

"""Mojo backend for ``upde/geometric.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._geometric_validation import (
    TWO_PI,
    validate_torus_inputs,
    validate_torus_output,
)

from .._mojo_runtime import require_mojo_executable, run_mojo_executable

__all__ = ["_ensure_exe", "torus_run_mojo"]

FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "geometric_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/geometric.mojo -o mojo/geometric_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def torus_run_mojo(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate torus phase dynamics.

    The calculation is delegated to the Mojo backend.
    """
    (
        p,
        o,
        k,
        a,
        n_i,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
    ) = validate_torus_inputs(
        phases,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        zeta,
        psi,
        dt,
        n_steps,
    )
    if n_steps_i == 0:
        return p % TWO_PI
    exe = _ensure_exe()
    tokens: list[str] = [
        "TORUS",
        str(n_i),
        repr(zeta_f),
        repr(psi_f),
        repr(dt_f),
        str(n_steps_i),
    ]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in o.tolist())
    tokens.extend(repr(float(x)) for x in k.tolist())
    tokens.extend(repr(float(x)) for x in a.tolist())
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo geometric exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != n_i:
        raise ValueError(f"Mojo TORUS returned {len(lines)} lines, expected {n_i}")
    values: list[float] = []
    for line in lines:
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                "Mojo TORUS output must be finite phases in [0, 2*pi)"
            ) from exc
        if not np.isfinite(value) or value < 0.0 or value >= TWO_PI:
            raise ValueError("Mojo TORUS output must be finite phases in [0, 2*pi)")
        values.append(value)
    return validate_torus_output(np.array(values, dtype=np.float64), n=n_i)
