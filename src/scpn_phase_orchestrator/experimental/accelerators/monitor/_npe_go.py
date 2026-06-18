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

from ._npe_validation import (
    expected_npe_backend_output,
    expected_phase_distance_backend_output,
    validate_npe_backend_inputs,
    validate_npe_backend_output,
    validate_phase_distance_backend_input,
    validate_phase_distance_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["phase_distance_matrix_go", "compute_npe_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libnpe.so"
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
    """Compute pairwise wrapped phase distances through the Go backend."""
    p = validate_phase_distance_backend_input(phases)
    lib = _load_lib()
    n = p.size
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.PhaseDistanceMatrix(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go PhaseDistanceMatrix rc={rc}")
    return validate_phase_distance_backend_output(
        out,
        n_phases=n,
        expected=expected_phase_distance_backend_output(p),
    )


def compute_npe_go(phases: FloatArray, max_radius: float) -> float:
    """Compute normalised phase entropy through the Go backend."""
    p, radius = validate_npe_backend_inputs(phases, max_radius)
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.ComputeNPE(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(p.size),
        ctypes.c_double(radius),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go ComputeNPE rc={rc}")
    return validate_npe_backend_output(
        out.value,
        expected=expected_npe_backend_output(p, radius),
    )
