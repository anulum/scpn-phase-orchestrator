# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for winding numbers

"""Go backend for ``monitor/winding.py`` via ``libwinding.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["winding_numbers_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libwinding.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libwinding.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libwinding.so winding.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.WindingNumbers.restype = ctypes.c_int
    lib.WindingNumbers.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
    ]
    _LIB = lib
    return lib


def winding_numbers_go(
    phases_flat: NDArray,
    t: int,
    n: int,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases_flat.ravel(), dtype=np.float64)
    out = np.zeros(n, dtype=np.int64)
    rc = lib.WindingNumbers(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(n)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
    )
    if rc != 0:
        raise ValueError(f"Go WindingNumbers rc={rc}")
    return out
