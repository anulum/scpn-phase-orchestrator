# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Hodge decomposition

"""Mojo backend for ``coupling/hodge.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "hodge_decomposition_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "hodge_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/hodge.mojo "
            f"-o mojo/hodge_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def hodge_decomposition_mojo(
    knm_flat: NDArray, phases: NDArray, n: int,
) -> tuple[NDArray, NDArray, NDArray]:
    exe = _ensure_exe()
    k = np.ascontiguousarray(knm_flat.ravel(), dtype=np.float64)
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    tokens: list[str] = ["HODGE", str(int(n))]
    tokens.extend(repr(float(x)) for x in k.tolist())
    tokens.extend(repr(float(x)) for x in p.tolist())
    proc = subprocess.run(
        [str(exe)], input=" ".join(tokens) + "\n",
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo hodge exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != 3 * n:
        raise ValueError(
            f"Mojo HODGE returned {len(lines)} lines, expected {3 * n}"
        )
    gradient = np.array([float(x) for x in lines[:n]], dtype=np.float64)
    curl = np.array([float(x) for x in lines[n : 2 * n]], dtype=np.float64)
    harmonic = np.array(
        [float(x) for x in lines[2 * n : 3 * n]], dtype=np.float64,
    )
    return gradient, curl, harmonic
