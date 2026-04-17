# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for order parameters

"""Go backend for ``upde/order_params.py``.

Calls ``liborder_params.so`` (built from ``go/order_params.go``) via
ctypes. Raises ``ImportError`` when the library is missing.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "order_parameter_go",
    "plv_go",
    "layer_coherence_go",
]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "liborder_params.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"liborder_params.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o liborder_params.so "
            f"order_params.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.OrderParameter.restype = ctypes.c_int
    lib.OrderParameter.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.PLV.restype = ctypes.c_int
    lib.PLV.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.LayerCoherence.restype = ctypes.c_int
    lib.LayerCoherence.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def order_parameter_go(phases: NDArray) -> tuple[float, float]:
    lib = _load_lib()
    phases64 = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    out_r = ctypes.c_double(0.0)
    out_psi = ctypes.c_double(0.0)
    rc = lib.OrderParameter(
        phases64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(phases64.size),
        ctypes.byref(out_r),
        ctypes.byref(out_psi),
    )
    if rc != 0:
        raise ValueError(f"Go OrderParameter rc={rc}")
    return float(out_r.value), float(out_psi.value)


def plv_go(phases_a: NDArray, phases_b: NDArray) -> float:
    lib = _load_lib()
    if phases_a.size != phases_b.size:
        raise ValueError(
            f"PLV requires equal-length arrays, got "
            f"{phases_a.size} vs {phases_b.size}"
        )
    a64 = np.ascontiguousarray(phases_a.ravel(), dtype=np.float64)
    b64 = np.ascontiguousarray(phases_b.ravel(), dtype=np.float64)
    out = ctypes.c_double(0.0)
    rc = lib.PLV(
        a64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(a64.size),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go PLV rc={rc}")
    return float(out.value)


def layer_coherence_go(phases: NDArray, indices: NDArray) -> float:
    lib = _load_lib()
    p64 = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    i64 = np.ascontiguousarray(indices.ravel(), dtype=np.int64)
    out = ctypes.c_double(0.0)
    rc = lib.LayerCoherence(
        p64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(p64.size),
        i64.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(i64.size),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go LayerCoherence rc={rc}")
    return float(out.value)
