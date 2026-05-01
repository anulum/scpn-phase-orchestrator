# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for transfer entropy

"""Mojo backend for ``monitor/transfer_entropy.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_te_mojo", "te_matrix_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "transfer_entropy_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/transfer_entropy.mojo "
            f"-o mojo/transfer_entropy_mojo -Xlinker -lm"
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
            f"Mojo transfer_entropy returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [float(line) for line in proc.stdout.strip().splitlines() if line]


def phase_te_mojo(source: FloatArray, target: FloatArray, n_bins: int) -> float:
    s = np.ascontiguousarray(source.ravel(), dtype=np.float64)
    t = np.ascontiguousarray(target.ravel(), dtype=np.float64)
    n = int(min(s.size, t.size))
    if n < 3:
        return 0.0
    tokens = ["PTE", str(n), str(n_bins)]
    tokens.extend(repr(float(x)) for x in s[:n].tolist())
    tokens.extend(repr(float(x)) for x in t[:n].tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])


def te_matrix_mojo(
    phase_series: FloatArray,
    n_osc: int,
    n_time: int,
    n_bins: int,
) -> FloatArray:
    s = np.ascontiguousarray(phase_series, dtype=np.float64)
    tokens = ["MAT", str(n_osc), str(n_time), str(n_bins)]
    tokens.extend(repr(float(x)) for x in s.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n_osc * n_osc:
        raise ValueError(
            f"Mojo MAT returned {len(result)} values, expected {n_osc * n_osc}"
        )
    return np.array(result, dtype=np.float64)
