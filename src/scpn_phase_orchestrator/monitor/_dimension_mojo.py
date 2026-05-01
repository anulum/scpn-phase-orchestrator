# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for dimension kernels

"""Mojo backend for ``monitor/dimension.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "_ensure_exe",
    "correlation_integral_mojo",
    "kaplan_yorke_dimension_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "dimension_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/dimension.mojo "
            f"-o mojo/dimension_mojo -Xlinker -lm"
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
            f"Mojo dimension returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return [float(line) for line in proc.stdout.strip().splitlines() if line]


def correlation_integral_mojo(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    n_p = int(idx_i.size)
    n_k = int(epsilons.size)
    tokens: list[str] = [
        "CI",
        str(int(t)),
        str(int(d)),
        str(n_p),
        str(n_k),
    ]
    tokens.extend(str(int(x)) for x in idx_i.ravel().tolist())
    tokens.extend(str(int(x)) for x in idx_j.ravel().tolist())
    tokens.extend(repr(float(x)) for x in epsilons.ravel().tolist())
    tokens.extend(repr(float(x)) for x in traj_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n_k:
        raise ValueError(f"Mojo CI returned {len(result)} values, expected {n_k}")
    return np.array(result, dtype=np.float64)


def kaplan_yorke_dimension_mojo(lyapunov_exponents: FloatArray) -> float:
    le = np.asarray(lyapunov_exponents, dtype=np.float64).ravel()
    n = int(le.size)
    tokens: list[str] = ["KY", str(n)]
    tokens.extend(repr(float(x)) for x in le.tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])
