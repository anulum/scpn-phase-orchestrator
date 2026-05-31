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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._poincare_validation import (
    validate_phase_poincare_backend_inputs,
    validate_poincare_backend_outputs,
    validate_poincare_section_backend_inputs,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "_ensure_exe",
    "phase_poincare_mojo",
    "poincare_section_mojo",
]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "poincare_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/poincare.mojo "
            f"-o mojo/poincare_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str) -> list[str]:
    exe = _ensure_exe()
    proc = subprocess.run(  # nosec B603
        [str(exe)],
        input=payload,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo poincare returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    return proc.stdout.splitlines()


def _parse(lines: list[str], dim: int, t: int) -> tuple[FloatArray, FloatArray, int]:
    if not lines:
        raise ValueError("Mojo Poincare output missing crossing count header")
    try:
        n_cr = int(lines[0])
    except ValueError as exc:
        raise ValueError("Mojo Poincare crossing count must be an integer") from exc
    if n_cr < 0:
        raise ValueError("Mojo Poincare crossing count must be non-negative")
    expected_lines = 1 + n_cr * dim + n_cr
    if len(lines) != expected_lines:
        raise ValueError(
            f"Mojo Poincare returned {len(lines)} lines, expected {expected_lines}"
        )
    crossings_flat = np.zeros(t * dim, dtype=np.float64)
    times = np.zeros(t, dtype=np.float64)
    pos = 1
    try:
        for k in range(n_cr * dim):
            crossings_flat[k] = float(lines[pos + k])
    except ValueError as exc:
        raise ValueError("Mojo Poincare crossings must be finite real values") from exc
    pos += n_cr * dim
    try:
        for k in range(n_cr):
            times[k] = float(lines[pos + k])
    except ValueError as exc:
        raise ValueError("Mojo Poincare times must be finite real values") from exc
    return validate_poincare_backend_outputs(
        crossings_flat,
        times,
        n_cr,
        t=t,
        dim=dim,
    )


def poincare_section_mojo(
    traj_flat: FloatArray,
    t: int,
    d: int,
    normal: FloatArray,
    offset: float,
    direction_id: int,
) -> tuple[FloatArray, FloatArray, int]:
    """Extract Poincare section crossings through the Mojo backend."""

    traj, t, d, nrm, offset, direction_id = validate_poincare_section_backend_inputs(
        traj_flat,
        t,
        d,
        normal,
        offset,
        direction_id,
    )
    tokens: list[str] = [
        "SEC",
        str(t),
        str(d),
        str(direction_id),
        repr(offset),
    ]
    tokens.extend(repr(float(x)) for x in nrm.tolist())
    tokens.extend(repr(float(x)) for x in traj.tolist())
    return _parse(_run(" ".join(tokens) + "\n"), d, t)


def phase_poincare_mojo(
    phases_flat: FloatArray,
    t: int,
    n: int,
    oscillator_idx: int,
    section_phase: float,
) -> tuple[FloatArray, FloatArray, int]:
    """Compute phase-space Poincare diagnostics through the Mojo backend."""

    phases, t, n, oscillator_idx, section_phase = (
        validate_phase_poincare_backend_inputs(
            phases_flat,
            t,
            n,
            oscillator_idx,
            section_phase,
        )
    )
    tokens: list[str] = [
        "PHASE",
        str(t),
        str(n),
        str(oscillator_idx),
        repr(section_phase),
    ]
    tokens.extend(repr(float(x)) for x in phases.tolist())
    return _parse(_run(" ".join(tokens) + "\n"), n, t)
