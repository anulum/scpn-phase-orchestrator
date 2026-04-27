# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for NPE

"""Mojo backend for ``monitor/npe.py``. Text-stdin subprocess bridge."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_distance_matrix_mojo", "compute_npe_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "npe_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/npe.mojo "
            f"-o mojo/npe_mojo -Xlinker -lm"
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
            f"Mojo npe returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return [float(line) for line in proc.stdout.strip().splitlines() if line]


def phase_distance_matrix_mojo(phases: NDArray) -> NDArray:
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    n = int(p.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    tokens = ["PDM", str(n)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n * n:
        raise ValueError(f"Mojo PDM returned {len(result)} values, expected {n * n}")
    return np.array(result, dtype=np.float64)


def compute_npe_mojo(phases: NDArray, max_radius: float) -> float:
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    n = int(p.size)
    if n < 2:
        return 0.0
    tokens = ["NPE", str(n), repr(float(max_radius))]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])
