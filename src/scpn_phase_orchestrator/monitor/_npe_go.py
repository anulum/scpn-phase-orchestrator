# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for NPE

"""Go backend for ``monitor/npe.py``. Loads ``libnpe.so`` via ctypes."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_distance_matrix_go", "compute_npe_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libnpe.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libnpe.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libnpe.so npe.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.PhaseDistanceMatrix.restype = ctypes.c_int
    lib.PhaseDistanceMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ComputeNPE.restype = ctypes.c_int
    lib.ComputeNPE.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def phase_distance_matrix_go(phases: FloatArray) -> FloatArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    n = p.size
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.PhaseDistanceMatrix(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go PhaseDistanceMatrix rc={rc}")
    return out


def compute_npe_go(phases: FloatArray, max_radius: float) -> float:
    lib = _load_lib()
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    out = ctypes.c_double(0.0)
    rc = lib.ComputeNPE(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(p.size),
        ctypes.c_double(max_radius),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go ComputeNPE rc={rc}")
    return float(out.value)
