# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for swarmalator stepper

"""Mojo backend for ``upde/swarmalator.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._swarmalator_validation import (
    validate_swarmalator_inputs,
    validate_swarmalator_output,
)

__all__ = ["_ensure_exe", "swarmalator_step_mojo"]

FloatArray: TypeAlias = NDArray[np.float64]
TWO_PI = 2.0 * np.pi

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "swarmalator_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/swarmalator.mojo "
            f"-o mojo/swarmalator_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def swarmalator_step_mojo(
    pos: FloatArray,
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance one swarmalator position-phase step.

    The calculation is delegated to the Mojo backend.
    """
    p, ph, om, n_i, dim_i, a_f, b_f, j_f, k_f, dt_f = validate_swarmalator_inputs(
        pos,
        phases,
        omegas,
        n,
        dim,
        a,
        b,
        j,
        k,
        dt,
    )
    exe = _ensure_exe()
    tokens: list[str] = [
        "STEP",
        str(n_i),
        str(dim_i),
        repr(a_f),
        repr(b_f),
        repr(j_f),
        repr(k_f),
        repr(dt_f),
    ]
    tokens.extend(repr(float(x)) for x in p.ravel().tolist())
    tokens.extend(repr(float(x)) for x in ph.tolist())
    tokens.extend(repr(float(x)) for x in om.tolist())
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo swarmalator exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    expected = n_i * dim_i + n_i
    if len(lines) != expected:
        raise ValueError(f"Mojo STEP returned {len(lines)} lines, expected {expected}")
    values: list[float] = []
    phase_offset = n_i * dim_i
    for index, line in enumerate(lines):
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                "Mojo STEP output must contain finite positions followed by "
                "finite phases in [0, 2*pi)"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                "Mojo STEP output must contain finite positions followed by "
                "finite phases in [0, 2*pi)"
            )
        if index >= phase_offset and (value < 0.0 or value >= TWO_PI):
            raise ValueError(
                "Mojo STEP output must contain finite positions followed by "
                "finite phases in [0, 2*pi)"
            )
        values.append(value)
    new_pos: FloatArray = np.array(values[:phase_offset], dtype=np.float64)
    new_phases: FloatArray = np.array(values[phase_offset:], dtype=np.float64)
    return validate_swarmalator_output(new_pos, new_phases, n=n_i, dim=dim_i)
