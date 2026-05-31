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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._market_validation import (
    validate_market_order_inputs,
    validate_market_order_output,
    validate_market_plv_inputs,
    validate_market_plv_output,
)

__all__ = ["market_order_parameter_go", "market_plv_go"]

FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libmarket.so"
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
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> FloatArray:
    """Compute market phase order parameter.

    The calculation is delegated to the Go backend.
    """

    p, t_i, n_i = validate_market_order_inputs(phases_flat, t, n)
    lib = _load_lib()
    out = np.zeros(t_i, dtype=np.float64)
    rc = lib.MarketOrderParameter(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_i),
        ctypes.c_int(n_i),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go MarketOrderParameter rc={rc}")
    return validate_market_order_output(out, t=t_i)


def market_plv_go(
    phases_flat: FloatArray,
    t: int,
    n: int,
    window: int,
) -> FloatArray:
    """Compute market phase-locking value.

    The calculation is delegated to the Go backend.
    """

    p, t_i, n_i, window_i = validate_market_plv_inputs(
        phases_flat,
        t,
        n,
        window,
    )
    lib = _load_lib()
    n_windows = t_i - window_i + 1
    out = np.zeros(n_windows * n_i * n_i, dtype=np.float64)
    rc = lib.MarketPLV(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_i),
        ctypes.c_int(n_i),
        ctypes.c_int(window_i),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go MarketPLV rc={rc}")
    return validate_market_plv_output(out, t=t_i, n=n_i, window=window_i)
