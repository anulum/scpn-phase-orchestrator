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

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "inertial_step_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "inertial_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/inertial.mojo -o mojo/inertial_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def inertial_step_mojo(
    theta: NDArray,
    omega_dot: NDArray,
    power: NDArray,
    knm_flat: NDArray,
    inertia: NDArray,
    damping: NDArray,
    n: int,
    dt: float,
) -> tuple[NDArray, NDArray]:
    exe = _ensure_exe()
    tokens: list[str] = ["INERT", str(int(n)), repr(float(dt))]
    for arr in (theta, omega_dot, power):
        tokens.extend(repr(float(x)) for x in np.asarray(arr).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist())
    for arr in (inertia, damping):
        tokens.extend(repr(float(x)) for x in np.asarray(arr).ravel().tolist())
    proc = subprocess.run(
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(f"Mojo inertial exit {proc.returncode}: {proc.stderr.strip()}")
    lines = proc.stdout.strip().splitlines()
    if len(lines) != 2 * n:
        raise ValueError(f"Mojo INERT returned {len(lines)} lines, expected {2 * n}")
    new_theta = np.array([float(x) for x in lines[:n]], dtype=np.float64)
    new_omega_dot = np.array(
        [float(x) for x in lines[n:]],
        dtype=np.float64,
    )
    return new_theta, new_omega_dot
