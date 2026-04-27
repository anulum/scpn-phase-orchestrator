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

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "torus_run_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "geometric_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/geometric.mojo -o mojo/geometric_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def torus_run_mojo(
    phases: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    exe = _ensure_exe()
    tokens: list[str] = [
        "TORUS",
        str(int(n)),
        repr(float(zeta)),
        repr(float(psi)),
        repr(float(dt)),
        str(int(n_steps)),
    ]
    tokens.extend(repr(float(x)) for x in np.asarray(phases).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(omegas).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(alpha_flat).ravel().tolist())
    proc = subprocess.run(
        [str(exe)],
        input=" ".join(tokens) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo geometric exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != n:
        raise ValueError(f"Mojo TORUS returned {len(lines)} lines, expected {n}")
    return np.array([float(x) for x in lines], dtype=np.float64)
