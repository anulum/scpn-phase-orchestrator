# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for entropy production rate

"""Go backend for ``monitor/entropy_prod.py`` via ``libentropy_prod.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["entropy_production_rate_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libentropy_prod.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libentropy_prod.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libentropy_prod.so "
            f"entropy_prod.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.EntropyProductionRate.restype = ctypes.c_int
    lib.EntropyProductionRate.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def entropy_production_rate_go(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: float,
    dt: float,
) -> float:
    lib = _load_lib()
    n = int(phases.size)
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    o = np.ascontiguousarray(omegas.ravel(), dtype=np.float64)
    k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    out = ctypes.c_double(0.0)
    rc = lib.EntropyProductionRate(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(float(alpha)),
        ctypes.c_double(float(dt)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go EntropyProductionRate rc={rc}")
    return float(out.value)
