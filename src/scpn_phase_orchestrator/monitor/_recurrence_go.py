# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for recurrence kernels

"""Go backend for ``monitor/recurrence.py`` via ``librecurrence.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["cross_recurrence_matrix_go", "recurrence_matrix_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "librecurrence.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"librecurrence.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o librecurrence.so "
            f"recurrence.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.RecurrenceMatrix.restype = ctypes.c_int
    lib.RecurrenceMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    lib.CrossRecurrenceMatrix.restype = ctypes.c_int
    lib.CrossRecurrenceMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    _LIB = lib
    return lib


def recurrence_matrix_go(
    traj_flat: NDArray, t: int, d: int, epsilon: float, angular: bool,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(traj_flat.ravel(), dtype=np.float64)
    out = np.zeros(t * t, dtype=np.uint8)
    rc = lib.RecurrenceMatrix(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)), ctypes.c_int(int(d)),
        ctypes.c_double(float(epsilon)), ctypes.c_int(int(angular)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    if rc != 0:
        raise ValueError(f"Go RecurrenceMatrix rc={rc}")
    return out


def cross_recurrence_matrix_go(
    traj_a_flat: NDArray,
    traj_b_flat: NDArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> NDArray:
    lib = _load_lib()
    a = np.ascontiguousarray(traj_a_flat.ravel(), dtype=np.float64)
    b = np.ascontiguousarray(traj_b_flat.ravel(), dtype=np.float64)
    out = np.zeros(t * t, dtype=np.uint8)
    rc = lib.CrossRecurrenceMatrix(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)), ctypes.c_int(int(d)),
        ctypes.c_double(float(epsilon)), ctypes.c_int(int(angular)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    if rc != 0:
        raise ValueError(f"Go CrossRecurrenceMatrix rc={rc}")
    return out
