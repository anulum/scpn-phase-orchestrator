# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for PAC

"""Go backend for ``upde/pac.py``. Calls ``libpac.so`` via ctypes."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["modulation_index_go", "pac_matrix_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libpac.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libpac.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libpac.so pac.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ModulationIndex.restype = ctypes.c_int
    lib.ModulationIndex.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.PACMatrix.restype = ctypes.c_int
    lib.PACMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def modulation_index_go(
    theta_low: NDArray, amp_high: NDArray, n_bins: int
) -> float:
    lib = _load_lib()
    t = np.ascontiguousarray(theta_low.ravel(), dtype=np.float64)
    a = np.ascontiguousarray(amp_high.ravel(), dtype=np.float64)
    n = min(t.size, a.size)
    out = ctypes.c_double(0.0)
    rc = lib.ModulationIndex(
        t[:n].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a[:n].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(n_bins),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go ModulationIndex rc={rc}")
    return float(out.value)


def pac_matrix_go(
    phases_flat: NDArray,
    amplitudes_flat: NDArray,
    t: int,
    n: int,
    n_bins: int,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases_flat, dtype=np.float64)
    a = np.ascontiguousarray(amplitudes_flat, dtype=np.float64)
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.PACMatrix(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t),
        ctypes.c_int(n),
        ctypes.c_int(n_bins),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go PACMatrix rc={rc}")
    return out
