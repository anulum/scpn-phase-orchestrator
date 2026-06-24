# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for winding numbers

"""Mojo backend for ``monitor/winding.py``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._winding_validation import (
    expected_winding_backend_output,
    validate_winding_backend_inputs,
    validate_winding_backend_output,
)

__all__ = ["_ensure_exe", "winding_numbers_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_EXE_PATH = Path(__file__).resolve().parents[5] / "mojo" / "winding_mojo"


def _ensure_exe() -> Path:
    """Build the Mojo backend executable if it is missing, else raise."""
    if not _EXE_PATH.exists():
        raise ImportError(
            f"{_EXE_PATH} not built. Run: mojo build mojo/winding.mojo "
            f"-o mojo/winding_mojo -Xlinker -lm"
        )
    return _EXE_PATH


def _run(payload: str, *, expected_count: int, label: str) -> list[int]:
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
            f"Mojo winding returned exit {proc.returncode}: {proc.stderr.strip()}"
        )
    lines = proc.stdout.splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"Mojo {label} must emit exactly {expected_count} integer line(s), "
            f"got {len(lines)}"
        )
    values: list[int] = []
    for line in lines:
        try:
            values.append(int(line))
        except ValueError as exc:
            raise ValueError(
                f"Mojo {label} emitted a non-integer winding value: {line!r}"
            ) from exc
    return values


def winding_numbers_mojo(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> IntArray:
    """Compute oscillator winding numbers through the Mojo backend."""
    phases, t, n = validate_winding_backend_inputs(phases_flat, t, n)
    tokens: list[str] = ["WIND", str(t), str(n)]
    tokens.extend(repr(float(x)) for x in phases.tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n, label="WIND")
    expected = expected_winding_backend_output(phases, t, n)
    return validate_winding_backend_output(result, t=t, n=n, expected=expected)
