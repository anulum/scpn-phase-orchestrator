# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Hodge decomposition

"""Go backend for ``coupling/hodge.py`` via ``libhodge.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["hodge_decomposition_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libhodge.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libhodge.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libhodge.so hodge.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.HodgeDecomposition.restype = ctypes.c_int
    lib.HodgeDecomposition.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def hodge_decomposition_go(
    knm_flat: NDArray, phases: NDArray, n: int,
) -> tuple[NDArray, NDArray, NDArray]:
    lib = _load_lib()
    k = np.ascontiguousarray(knm_flat.ravel(), dtype=np.float64)
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    gradient = np.zeros(n, dtype=np.float64)
    curl = np.zeros(n, dtype=np.float64)
    harmonic = np.zeros(n, dtype=np.float64)
    rc = lib.HodgeDecomposition(
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        curl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        harmonic.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go HodgeDecomposition rc={rc}")
    return gradient, curl, harmonic
