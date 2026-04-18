# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for inter-trial phase coherence

"""Go backend for ``monitor/itpc.py`` via ``libitpc.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["compute_itpc_go", "itpc_persistence_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libitpc.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libitpc.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libitpc.so itpc.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ComputeITPC.restype = ctypes.c_int
    lib.ComputeITPC.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ITPCPersistence.restype = ctypes.c_int
    lib.ITPCPersistence.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def compute_itpc_go(
    phases_flat: NDArray, n_trials: int, n_tp: int
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64)
    out = np.zeros(n_tp, dtype=np.float64)
    rc = lib.ComputeITPC(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n_trials)),
        ctypes.c_int(int(n_tp)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go ComputeITPC rc={rc}")
    return out


def itpc_persistence_go(
    phases_flat: NDArray,
    n_trials: int,
    n_tp: int,
    pause_indices: NDArray,
) -> float:
    lib = _load_lib()
    p = np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64)
    idx = np.ascontiguousarray(pause_indices.ravel(), dtype=np.int64)
    out = ctypes.c_double(0.0)
    rc = lib.ITPCPersistence(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n_trials)),
        ctypes.c_int(int(n_tp)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(idx.size)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go ITPCPersistence rc={rc}")
    return float(out.value)
