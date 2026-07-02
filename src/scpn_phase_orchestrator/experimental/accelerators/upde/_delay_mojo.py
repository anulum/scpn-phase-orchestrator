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
from ._delay_validation import validate_delay_backend_inputs

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
    """Integrate delayed Kuramoto dynamics via the Mojo executable."""
    ph, om, knm, alpha, n, zeta, psi, dt, delay_steps, n_steps = (
        validate_delay_backend_inputs(
            phases, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, delay_steps, n_steps
        )
    )
    exe = _ensure_exe()
    tokens: list[str] = [
        "DELAY",
        str(int(n)),
        str(int(delay_steps)),
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
    parsed = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(parsed)):
        raise ValueError("Mojo DELAY output must contain finite phases")
    return parsed
