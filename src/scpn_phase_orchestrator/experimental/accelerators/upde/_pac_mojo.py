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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._pac_validation import (
    validate_modulation_index_inputs,
    validate_modulation_index_output,
    validate_pac_matrix_inputs,
    validate_pac_matrix_output,
)

from .._mojo_runtime import require_mojo_executable, run_mojo_executable

__all__ = ["modulation_index_mojo", "pac_matrix_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "pac_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/pac.mojo "
            f"-o mojo/pac_mojo -Xlinker -lm"
        )
    return require_mojo_executable(_EXE_PATH)


def _run(payload: str, *, expected_count: int, label: str) -> list[float]:
    """Call the backend kernel with the prepared inputs and return its result."""
    exe = _ensure_exe()
    proc = run_mojo_executable(exe, payload, runner=subprocess.run)
    if proc.returncode != 0:
        raise ValueError(
            f"Mojo pac returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo PAC {label} returned {len(lines)} lines, expected {expected_count}"
        )
    values: list[float] = []
    for line in lines:
        try:
            value = float(line)
        except ValueError as exc:
            raise ValueError(
                f"Mojo PAC {label} output must be finite real values"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Mojo PAC {label} output must be finite real values")
        values.append(value)
    return values


def modulation_index_mojo(
    theta_low: FloatArray, amp_high: FloatArray, n_bins: int
) -> float:
    """Compute phase-amplitude coupling modulation index.

    The calculation is delegated to the Mojo backend.
    """
    t, a, bins = validate_modulation_index_inputs(theta_low, amp_high, n_bins)
    n = int(t.size)
    if n == 0:
        return 0.0
    tokens = ["MI", str(n), str(bins)]
    tokens.extend(repr(float(x)) for x in t.tolist())
    tokens.extend(repr(float(x)) for x in a.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=1, label="MI")
    return validate_modulation_index_output(result[0])


def pac_matrix_mojo(
    phases_flat: FloatArray,
    amplitudes_flat: FloatArray,
    t: int,
    n: int,
    n_bins: int,
) -> FloatArray:
    """Compute the phase-amplitude coupling matrix.

    The calculation is delegated to the Mojo backend.
    """
    p, a, t_i, n_i, bins = validate_pac_matrix_inputs(
        phases_flat,
        amplitudes_flat,
        t,
        n,
        n_bins,
    )
    tokens = ["MAT", str(t_i), str(n_i), str(bins)]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in a.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n_i * n_i, label="matrix")
    return validate_pac_matrix_output(result, n=n_i)
