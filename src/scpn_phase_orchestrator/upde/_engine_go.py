# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for UPDE engine

"""Go backend for ``upde/engine.py`` via ``libupde_engine.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["upde_run_go"]

_METHOD_IDS = {"euler": 0, "rk4": 1, "rk45": 2}

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libupde_engine.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libupde_engine.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libupde_engine.so "
            f"upde_engine.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.UPDERun.restype = ctypes.c_int
    lib.UPDERun.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # phases (in/out)
        ctypes.POINTER(ctypes.c_double),  # omegas
        ctypes.POINTER(ctypes.c_double),  # knm
        ctypes.POINTER(ctypes.c_double),  # alpha
        ctypes.c_int,  # n
        ctypes.c_double,  # zeta
        ctypes.c_double,  # psi
        ctypes.c_double,  # dt
        ctypes.c_int,  # n_steps
        ctypes.c_int,  # method (0/1/2)
        ctypes.c_int,  # n_substeps
        ctypes.c_double,  # atol
        ctypes.c_double,  # rtol
    ]
    _LIB = lib
    return lib


def upde_run_go(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> NDArray:
    lib = _load_lib()
    if method not in _METHOD_IDS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {list(_METHOD_IDS)}"
        )
    n = int(phases.size)
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64).copy()
    o = np.ascontiguousarray(omegas.ravel(), dtype=np.float64)
    k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
    rc = lib.UPDERun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_steps)),
        ctypes.c_int(int(_METHOD_IDS[method])),
        ctypes.c_int(int(n_substeps)),
        ctypes.c_double(float(atol)),
        ctypes.c_double(float(rtol)),
    )
    if rc != 0:
        raise ValueError(f"Go UPDERun rc={rc}")
    return p
