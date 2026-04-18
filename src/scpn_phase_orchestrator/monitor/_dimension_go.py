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

import numpy as np
from numpy.typing import NDArray

__all__ = ["correlation_integral_go", "kaplan_yorke_dimension_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libdimension.so"
)
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
        ctypes.c_int, ctypes.c_int,
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
    traj_flat: NDArray,
    t: int,
    d: int,
    idx_i: NDArray,
    idx_j: NDArray,
    epsilons: NDArray,
) -> NDArray:
    lib = _load_lib()
    traj = np.ascontiguousarray(traj_flat, dtype=np.float64)
    ii = np.ascontiguousarray(idx_i.ravel(), dtype=np.int64)
    jj = np.ascontiguousarray(idx_j.ravel(), dtype=np.int64)
    eps = np.ascontiguousarray(epsilons.ravel(), dtype=np.float64)
    out = np.zeros(eps.size, dtype=np.float64)
    rc = lib.CorrelationIntegral(
        traj.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)), ctypes.c_int(int(d)),
        ii.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        jj.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(ii.size)),
        eps.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(eps.size)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go CorrelationIntegral rc={rc}")
    return out


def kaplan_yorke_dimension_go(lyapunov_exponents: NDArray) -> float:
    lib = _load_lib()
    le = np.ascontiguousarray(lyapunov_exponents.ravel(), dtype=np.float64)
    out = ctypes.c_double(0.0)
    rc = lib.KaplanYorkeDimension(
        le.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(le.size)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go KaplanYorkeDimension rc={rc}")
    return float(out.value)
