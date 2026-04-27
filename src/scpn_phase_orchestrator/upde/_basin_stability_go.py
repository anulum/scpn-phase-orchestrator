# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for steady-state R

"""Go backend for ``upde/basin_stability.py`` via ``libbasin_stability.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["steady_state_r_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libbasin_stability.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libbasin_stability.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libbasin_stability.so basin_stability.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SteadyStateR.restype = ctypes.c_double
    lib.SteadyStateR.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _LIB = lib
    return lib


def steady_state_r_go(
    phases_init: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    lib = _load_lib()
    p = np.ascontiguousarray(phases_init, dtype=np.float64)
    o = np.ascontiguousarray(omegas, dtype=np.float64)
    k = np.ascontiguousarray(knm_flat, dtype=np.float64)
    a = np.ascontiguousarray(alpha_flat, dtype=np.float64)
    r = lib.SteadyStateR(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        ctypes.c_double(float(k_scale)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_transient)),
        ctypes.c_int(int(n_measure)),
    )
    return float(r)
