# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for dimension kernels

"""Go backend for ``monitor/dimension.py`` via ``libdimension.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._dimension_validation import (
    expected_correlation_integral_backend_output,
    expected_kaplan_yorke_backend_output,
    validate_correlation_integral_backend_inputs,
    validate_correlation_integral_backend_output,
    validate_kaplan_yorke_backend_input,
    validate_kaplan_yorke_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["correlation_integral_go", "kaplan_yorke_dimension_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libdimension.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libdimension.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libdimension.so "
            f"dimension.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.CorrelationIntegral.restype = ctypes.c_int
    lib.CorrelationIntegral.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.KaplanYorkeDimension.restype = ctypes.c_int
    lib.KaplanYorkeDimension.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def correlation_integral_go(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    """Compute the phase-space correlation integral through the Go backend."""
    traj, t_int, d_int, ii, jj, eps = validate_correlation_integral_backend_inputs(
        traj_flat,
        t,
        d,
        idx_i,
        idx_j,
        epsilons,
    )
    lib = _load_lib()
    out = np.zeros(eps.size, dtype=np.float64)
    rc = lib.CorrelationIntegral(
        traj.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_int),
        ctypes.c_int(d_int),
        ii.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        jj.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(ii.size)),
        eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(eps.size)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go CorrelationIntegral rc={rc}")
    expected = expected_correlation_integral_backend_output(
        traj,
        t_int,
        d_int,
        ii,
        jj,
        eps,
    )
    return validate_correlation_integral_backend_output(out, eps, expected=expected)


def kaplan_yorke_dimension_go(lyapunov_exponents: FloatArray) -> float:
    """Estimate the Kaplan-Yorke dimension through the Go backend."""
    le = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.KaplanYorkeDimension(
        le.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(le.size)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go KaplanYorkeDimension rc={rc}")
    expected = expected_kaplan_yorke_backend_output(le)
    return validate_kaplan_yorke_backend_output(out.value, le, expected=expected)
