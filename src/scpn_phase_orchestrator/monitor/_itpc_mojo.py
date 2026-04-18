# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for inter-trial phase coherence

"""Mojo backend for ``monitor/itpc.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["_ensure_exe", "compute_itpc_mojo", "itpc_persistence_mojo"]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "itpc_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/itpc.mojo "
            f"-o mojo/itpc_mojo -Xlinker -lm"
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
            f"Mojo itpc returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [
        float(line)
        for line in proc.stdout.strip().splitlines()
        if line
    ]


def compute_itpc_mojo(
    phases_flat: NDArray, n_trials: int, n_tp: int
) -> NDArray:
    if n_trials == 0 or n_tp == 0:
        return np.zeros(n_tp, dtype=np.float64)
    tokens: list[str] = ["ITPC", str(n_trials), str(n_tp)]
    tokens.extend(repr(float(x)) for x in phases_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != n_tp:
        raise ValueError(
            f"Mojo ITPC returned {len(result)} values, expected {n_tp}"
        )
    return np.array(result, dtype=np.float64)


def itpc_persistence_mojo(
    phases_flat: NDArray,
    n_trials: int,
    n_tp: int,
    pause_indices: NDArray,
) -> float:
    idx = np.ascontiguousarray(pause_indices.ravel(), dtype=np.int64)
    if idx.size == 0:
        return 0.0
    tokens: list[str] = [
        "PERS", str(n_trials), str(n_tp), str(int(idx.size)),
    ]
    tokens.extend(str(int(x)) for x in idx.tolist())
    tokens.extend(repr(float(x)) for x in phases_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    return float(result[0])
