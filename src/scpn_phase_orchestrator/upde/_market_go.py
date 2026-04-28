# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for market PLV / R(t)

"""Go backend for ``upde/market.py`` via ``libmarket.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["market_order_parameter_go", "market_plv_go"]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libmarket.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libmarket.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libmarket.so market.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.MarketOrderParameter.restype = ctypes.c_int
    lib.MarketOrderParameter.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.MarketPLV.restype = ctypes.c_int
    lib.MarketPLV.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def market_order_parameter_go(
    phases_flat: NDArray,
    t: int,
    n: int,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases_flat, dtype=np.float64)
    out = np.zeros(int(t), dtype=np.float64)
    rc = lib.MarketOrderParameter(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(n)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go MarketOrderParameter rc={rc}")
    return out


def market_plv_go(
    phases_flat: NDArray,
    t: int,
    n: int,
    window: int,
) -> NDArray:
    lib = _load_lib()
    n_windows = int(t) - int(window) + 1
    p = np.ascontiguousarray(phases_flat, dtype=np.float64)
    out = np.zeros(n_windows * int(n) * int(n), dtype=np.float64)
    rc = lib.MarketPLV(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(n)),
        ctypes.c_int(int(window)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go MarketPLV rc={rc}")
    return out
