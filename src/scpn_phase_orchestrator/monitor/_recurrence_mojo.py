# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for recurrence kernels

"""Mojo backend for ``monitor/recurrence.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_ensure_exe",
    "cross_recurrence_matrix_mojo",
    "recurrence_matrix_mojo",
]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "recurrence_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/recurrence.mojo "
            f"-o mojo/recurrence_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[int]:
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
            f"Mojo recurrence returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [
        int(line)
        for line in proc.stdout.strip().splitlines()
        if line
    ]


def recurrence_matrix_mojo(
    traj_flat: NDArray, t: int, d: int, epsilon: float, angular: bool,
) -> NDArray:
    tokens: list[str] = [
        "REC",
        str(int(t)), str(int(d)), str(int(angular)),
        repr(float(epsilon)),
    ]
    tokens.extend(repr(float(x)) for x in traj_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != t * t:
        raise ValueError(
            f"Mojo REC returned {len(result)} values, expected {t * t}"
        )
    return np.array(result, dtype=np.uint8)


def cross_recurrence_matrix_mojo(
    traj_a_flat: NDArray,
    traj_b_flat: NDArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> NDArray:
    tokens: list[str] = [
        "CROSS",
        str(int(t)), str(int(d)), str(int(angular)),
        repr(float(epsilon)),
    ]
    tokens.extend(repr(float(x)) for x in traj_a_flat.ravel().tolist())
    tokens.extend(repr(float(x)) for x in traj_b_flat.ravel().tolist())
    result = _run(" ".join(tokens) + "\n")
    if len(result) != t * t:
        raise ValueError(
            f"Mojo CROSS returned {len(result)} values, expected {t * t}"
        )
    return np.array(result, dtype=np.uint8)
