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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._npe_validation import (
    expected_npe_backend_output,
    expected_phase_distance_backend_output,
    validate_npe_backend_inputs,
    validate_npe_backend_output,
    validate_phase_distance_backend_input,
    validate_phase_distance_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_distance_matrix_mojo", "compute_npe_mojo"]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "npe_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/npe.mojo "
            f"-o mojo/npe_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str, *, expected_count: int, label: str) -> list[float]:
    """Call the backend kernel with the prepared inputs and return its result."""
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
            f"Mojo npe returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo {label} must emit exactly {expected_count} scalar line(s), "
            f"got {len(lines)}"
        )
    values: list[float] = []
    for line in lines:
        try:
            values.append(float(line))
        except ValueError as exc:
            raise ValueError(
                f"Mojo {label} emitted a non-scalar NPE value: {line!r}"
            ) from exc
    return values


def phase_distance_matrix_mojo(phases: FloatArray) -> FloatArray:
    """Compute pairwise wrapped phase distances through the Mojo backend."""
    p = validate_phase_distance_backend_input(phases)
    n = int(p.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    tokens = ["PDM", str(n)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n * n, label="PDM")
    return validate_phase_distance_backend_output(
        result,
        n_phases=n,
        expected=expected_phase_distance_backend_output(p),
    )


def compute_npe_mojo(phases: FloatArray, max_radius: float) -> float:
    """Compute normalised phase entropy through the Mojo backend."""
    p, radius = validate_npe_backend_inputs(phases, max_radius)
    n = int(p.size)
    if n < 2:
        return 0.0
    tokens = ["NPE", str(n), repr(radius)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=1, label="NPE")
    return validate_npe_backend_output(
        result[0],
        expected=expected_npe_backend_output(p, radius),
        atol=1.0e-9,
    )
