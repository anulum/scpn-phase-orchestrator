# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for chimera local-R kernel

"""Go backend for ``monitor/chimera.py`` via ``libchimera.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["local_order_parameter_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libchimera.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libchimera.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libchimera.so chimera.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.LocalOrderParameter.restype = ctypes.c_int
    lib.LocalOrderParameter.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def local_order_parameter_go(
    phases: NDArray, knm_flat: NDArray, n: int,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    k = np.ascontiguousarray(knm_flat.ravel(), dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    rc = lib.LocalOrderParameter(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go LocalOrderParameter rc={rc}")
    return out
