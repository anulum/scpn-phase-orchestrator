# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for winding numbers

"""Go backend for ``monitor/winding.py`` via ``libwinding.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._winding_validation import (
    expected_winding_backend_output,
    validate_winding_backend_inputs,
    validate_winding_backend_output,
)

__all__ = ["winding_numbers_go"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libwinding.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libwinding.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libwinding.so winding.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.WindingNumbers.restype = ctypes.c_int
    lib.WindingNumbers.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    _LIB = lib
    return lib


def winding_numbers_go(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> IntArray:
    """Compute oscillator winding numbers through the Go backend."""
    p, t, n = validate_winding_backend_inputs(phases_flat, t, n)
    lib = _load_lib()
    out = np.zeros(n, dtype=np.int64)
    rc = lib.WindingNumbers(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(n)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )
    if rc != 0:
        raise ValueError(f"Go WindingNumbers rc={rc}")
    expected = expected_winding_backend_output(p, t, n)
    return validate_winding_backend_output(out, t=t, n=n, expected=expected)
