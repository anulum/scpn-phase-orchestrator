# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Lyapunov spectrum

"""Mojo backend for ``monitor/lyapunov.py`` via a subprocess executable.

Loads ``mojo/lyapunov_mojo``; feeds the arguments as a single
whitespace-separated stdin payload and parses the ``n``-line f64 output.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "lyapunov_spectrum_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "lyapunov_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/lyapunov.mojo "
            f"-o mojo/lyapunov_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[float]:
    exe = _ensure_exe()
    proc = subprocess.run(
        [str(exe)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo lyapunov returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [
        float(line)
        for line in proc.stdout.strip().splitlines()
        if line
    ]


def lyapunov_spectrum_mojo(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> NDArray:
    n = int(phases_init.size)
    tokens: list[str] = [
        "SPEC",
        str(n),
        repr(float(dt)),
        str(int(n_steps)),
        str(int(qr_interval)),
        repr(float(zeta)),
        repr(float(psi)),
    ]
    tokens.extend(repr(float(x)) for x in phases_init.ravel().tolist())
    tokens.extend(repr(float(x)) for x in omegas.ravel().tolist())
    tokens.extend(repr(float(x)) for x in knm.ravel().tolist())
    tokens.extend(repr(float(x)) for x in alpha.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n:
        raise ValueError(
            f"Mojo SPEC returned {len(result)} values, expected {n}"
        )
    return np.array(result, dtype=np.float64)
