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

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "simplicial_run_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "simplicial_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build "
            f"mojo/simplicial.mojo -o mojo/simplicial_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def simplicial_run_mojo(
    phases: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    zeta: float,
    psi: float,
    sigma2: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    exe = _ensure_exe()
    tokens: list[str] = [
        "SIMP", str(int(n)),
        repr(float(zeta)), repr(float(psi)),
        repr(float(sigma2)), repr(float(dt)), str(int(n_steps)),
    ]
    tokens.extend(repr(float(x)) for x in np.asarray(phases).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(omegas).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(knm_flat).ravel().tolist())
    tokens.extend(repr(float(x)) for x in np.asarray(alpha_flat).ravel().tolist())
    proc = subprocess.run(
        [str(exe)], input=" ".join(tokens) + "\n",
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo simplicial exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != n:
        raise ValueError(
            f"Mojo SIMP returned {len(lines)} lines, expected {n}"
        )
    return np.array([float(x) for x in lines], dtype=np.float64)
