# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Poincaré kernels

"""Mojo backend for ``monitor/poincare.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "_ensure_exe",
    "phase_poincare_mojo",
    "poincare_section_mojo",
]

_EXE_PATH = (
    Path(__file__).resolve().parents[3] / "mojo" / "poincare_mojo"
)


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/poincare.mojo "
            f"-o mojo/poincare_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[str]:
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
            f"Mojo poincare returned exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return [line for line in proc.stdout.strip().splitlines() if line]


def _parse(lines: list[str], dim: int, t: int) -> tuple[NDArray, NDArray, int]:
    if not lines:
        return (
            np.zeros(t * dim, dtype=np.float64),
            np.zeros(t, dtype=np.float64),
            0,
        )
    n_cr = int(lines[0])
    crossings_flat = np.zeros(t * dim, dtype=np.float64)
    times = np.zeros(t, dtype=np.float64)
    pos = 1
    for k in range(n_cr * dim):
        crossings_flat[k] = float(lines[pos + k])
    pos += n_cr * dim
    for k in range(n_cr):
        times[k] = float(lines[pos + k])
    return crossings_flat, times, n_cr


def poincare_section_mojo(
    traj_flat: NDArray, t: int, d: int,
    normal: NDArray, offset: float, direction_id: int,
) -> tuple[NDArray, NDArray, int]:
    tokens: list[str] = [
        "SEC", str(int(t)), str(int(d)), str(int(direction_id)),
        repr(float(offset)),
    ]
    tokens.extend(repr(float(x)) for x in normal.ravel().tolist())
    tokens.extend(repr(float(x)) for x in traj_flat.ravel().tolist())
    return _parse(_run(" ".join(tokens) + "\n"), d, t)


def phase_poincare_mojo(
    phases_flat: NDArray, t: int, n: int,
    oscillator_idx: int, section_phase: float,
) -> tuple[NDArray, NDArray, int]:
    tokens: list[str] = [
        "PHASE", str(int(t)), str(int(n)),
        str(int(oscillator_idx)), repr(float(section_phase)),
    ]
    tokens.extend(repr(float(x)) for x in phases_flat.ravel().tolist())
    return _parse(_run(" ".join(tokens) + "\n"), n, t)
