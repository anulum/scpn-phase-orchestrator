# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for PAC

"""Mojo backend for ``upde/pac.py``. Text-stdin subprocess bridge."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["modulation_index_mojo", "pac_matrix_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[3] / "mojo" / "pac_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/pac.mojo "
            f"-o mojo/pac_mojo -Xlinker -lm"
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
            f"Mojo pac returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [
        float(line)
        for line in proc.stdout.strip().splitlines()
        if line
    ]


def modulation_index_mojo(
    theta_low: NDArray, amp_high: NDArray, n_bins: int
) -> float:
    t = np.ascontiguousarray(theta_low.ravel(), dtype=np.float64)
    a = np.ascontiguousarray(amp_high.ravel(), dtype=np.float64)
    n = int(min(t.size, a.size))
    if n == 0 or n_bins < 2:
        return 0.0
    tokens = ["MI", str(n), str(n_bins)]
    tokens.extend(repr(float(x)) for x in t[:n].tolist())
    tokens.extend(repr(float(x)) for x in a[:n].tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])


def pac_matrix_mojo(
    phases_flat: NDArray,
    amplitudes_flat: NDArray,
    t: int,
    n: int,
    n_bins: int,
) -> NDArray:
    p = np.ascontiguousarray(phases_flat, dtype=np.float64)
    a = np.ascontiguousarray(amplitudes_flat, dtype=np.float64)
    tokens = ["MAT", str(t), str(n), str(n_bins)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in a.tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n * n:
        raise ValueError(
            f"Mojo PAC matrix returned {len(result)} values, expected {n * n}"
        )
    return np.array(result, dtype=np.float64)
