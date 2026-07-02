# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for inertial stepper

"""Mojo backend for ``upde/inertial.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._mojo_runtime import require_mojo_executable, run_mojo_executable
from ._inertial_validation import (
    validate_inertial_inputs,
    validate_inertial_output,
)

__all__ = ["_ensure_exe", "inertial_step_mojo"]

FloatArray: TypeAlias = NDArray[np.float64]
TWO_PI = 2.0 * np.pi

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "inertial_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/inertial.mojo -o mojo/inertial_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def inertial_step_mojo(
    theta: FloatArray,
    omega_dot: FloatArray,
    power: FloatArray,
    knm_flat: FloatArray,
    inertia: FloatArray,
    damping: FloatArray,
    n: int,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance one inertial Kuramoto step.

    The calculation is delegated to the Mojo backend.
    """
    th, od, pw, km, ine, dmp, n_i, dt_f = validate_inertial_inputs(
        theta,
        omega_dot,
        power,
        knm_flat,
        inertia,
        damping,
        n,
        dt,
    )
    exe = _ensure_exe()
    tokens: list[str] = ["INERT", str(n_i), repr(dt_f)]
    for arr in (th, od, pw):
        tokens.extend(repr(float(x)) for x in arr.tolist())
    tokens.extend(repr(float(x)) for x in km.tolist())
    for arr in (ine, dmp):
        tokens.extend(repr(float(x)) for x in arr.tolist())
    proc = run_mojo_executable(exe, " ".join(tokens) + "\n", runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(f"Mojo inertial exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.splitlines()
    if len(lines) != 2 * n_i:
        raise ValueError(f"Mojo INERT returned {len(lines)} lines, expected {2 * n_i}")
    values: list[float] = []
    for index, line in enumerate(lines):
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                "Mojo INERT output must contain finite phases in [0, 2*pi) "
                "followed by finite frequency deviations"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                "Mojo INERT output must contain finite phases in [0, 2*pi) "
                "followed by finite frequency deviations"
            )
        if index < n_i and (value < 0.0 or value >= TWO_PI):
            raise ValueError(
                "Mojo INERT output must contain finite phases in [0, 2*pi) "
                "followed by finite frequency deviations"
            )
        values.append(value)
    new_theta = np.array(values[:n_i], dtype=np.float64)
    new_omega_dot = np.array(values[n_i:], dtype=np.float64)
    return validate_inertial_output(new_theta, new_omega_dot, n=n_i)
