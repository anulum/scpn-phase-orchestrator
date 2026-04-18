# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for simplicial Kuramoto

"""Go backend for ``upde/simplicial.py`` via ``libsimplicial.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["simplicial_run_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libsimplicial.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libsimplicial.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libsimplicial.so simplicial.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SimplicialRun.restype = ctypes.c_int
    lib.SimplicialRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def simplicial_run_go(
    phases: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    zeta: float,
    psi: float,
    sigma2: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases, dtype=np.float64)
    o = np.ascontiguousarray(omegas, dtype=np.float64)
    k = np.ascontiguousarray(knm_flat, dtype=np.float64)
    a = np.ascontiguousarray(alpha_flat, dtype=np.float64)
    out = np.zeros(int(n), dtype=np.float64)
    rc = lib.SimplicialRun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        ctypes.c_double(float(sigma2)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_steps)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SimplicialRun rc={rc}")
    return out
