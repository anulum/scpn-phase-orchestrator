# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for multi-head AttnRes

"""Go backend for the multi-head AttnRes dispatcher.

Calls ``libattnres.so`` (built from ``go/attnres.go``) via ctypes.
Raises ``ImportError`` when the compiled library is missing — the
dispatcher then falls through to the next backend.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._attnres_validation import (
    validate_attnres_backend_inputs,
    validate_attnres_backend_output,
)

__all__ = ["attnres_modulate_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libattnres.so"

_LIB: ctypes.CDLL | None = None
FloatArray: TypeAlias = NDArray[np.float64]


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libattnres.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libattnres.so attnres.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.AttnResModulate.restype = ctypes.c_int
    lib.AttnResModulate.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # knm
        ctypes.POINTER(ctypes.c_double),  # theta
        ctypes.POINTER(ctypes.c_double),  # w_q
        ctypes.POINTER(ctypes.c_double),  # w_k
        ctypes.POINTER(ctypes.c_double),  # w_v
        ctypes.POINTER(ctypes.c_double),  # w_o
        ctypes.c_int,  # n
        ctypes.c_int,  # n_heads
        ctypes.c_int,  # block_size
        ctypes.c_double,  # temperature
        ctypes.c_double,  # lambda
        ctypes.POINTER(ctypes.c_double),  # out
    ]
    _LIB = lib
    return lib


def attnres_modulate_go(
    knm_flat: FloatArray,
    theta: FloatArray,
    w_q: FloatArray,
    w_k: FloatArray,
    w_v: FloatArray,
    w_o: FloatArray,
    n: int,
    n_heads: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> FloatArray:
    """Go-backed multi-head AttnRes. Signature matches the Rust FFI."""
    (
        knm64,
        theta64,
        wq64,
        wk64,
        wv64,
        wo64,
        n,
        n_heads,
        block_size,
        temperature,
        lambda_,
    ) = validate_attnres_backend_inputs(
        knm_flat,
        theta,
        w_q,
        w_k,
        w_v,
        w_o,
        n,
        n_heads,
        block_size,
        temperature,
        lambda_,
    )
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    lib = _load_lib()
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.AttnResModulate(
        knm64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        theta64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wq64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wk64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wv64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        wo64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(n_heads),
        ctypes.c_int(block_size),
        ctypes.c_double(temperature),
        ctypes.c_double(lambda_),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go AttnResModulate returned error code {rc}")
    return validate_attnres_backend_output(out, n=n)
