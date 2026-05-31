# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for envelope kernels

"""Mojo backend for ``upde/envelope.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._envelope_validation import (
    validate_envelope_modulation_input,
    validate_envelope_modulation_output,
    validate_extract_envelope_input,
    validate_extract_envelope_output,
)

__all__ = [
    "_ensure_exe",
    "envelope_modulation_depth_mojo",
    "extract_envelope_mojo",
]
FloatArray: TypeAlias = NDArray[np.float64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "envelope_mojo"


def _ensure_exe() -> Path:
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/envelope.mojo "
            f"-o mojo/envelope_mojo -Xlinker -lm"
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
        raise ValueError(f"Mojo envelope exit {proc.returncode}: {proc.stderr.strip()}")
    return [line for line in proc.stdout.strip().splitlines() if line]


def extract_envelope_mojo(amps: FloatArray, window: int) -> FloatArray:
    """Extract the analytic phase envelope.

    The calculation is delegated to the Mojo backend.
    """

    a, window_i = validate_extract_envelope_input(amps, window)
    if a.size == 0:
        return np.zeros(0, dtype=np.float64)
    if window_i >= a.size:
        rms = float(np.sqrt(np.mean(a * a)))
        return np.full(a.size, rms, dtype=np.float64)
    tokens: list[str] = [
        "RMS",
        str(int(a.size)),
        str(window_i),
    ]
    tokens.extend(repr(float(x)) for x in a.tolist())
    lines = _run(" ".join(tokens) + "\n")
    return validate_extract_envelope_output(
        np.array([float(x) for x in lines], dtype=np.float64),
        n=int(a.size),
    )


def envelope_modulation_depth_mojo(env: FloatArray) -> float:
    """Compute envelope modulation depth.

    The calculation is delegated to the Mojo backend.
    """

    e = validate_envelope_modulation_input(env)
    if e.size == 0:
        return 0.0
    vmax = float(np.max(e))
    vmin = float(np.min(e))
    if vmax + vmin <= 0.0:
        return 0.0
    tokens: list[str] = ["MOD", str(int(e.size))]
    tokens.extend(repr(float(x)) for x in e.tolist())
    lines = _run(" ".join(tokens) + "\n")
    if not lines:
        raise ValueError("Mojo MOD returned empty output")
    return validate_envelope_modulation_output(float(lines[0]))
