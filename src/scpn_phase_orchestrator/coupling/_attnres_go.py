# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for AttnRes coupling modulation

"""Go backend for the AttnRes fallback chain.

Calls the c-shared ``libattnres.so`` built from ``go/attnres.go`` via
ctypes. The Go implementation is deliberately identical to the Rust
and Python reference algorithms so the backends are bit-compatible.
Raises ``ImportError`` when the compiled library is missing — the
dispatcher then falls through to the NumPy terminal fallback.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["attnres_modulate_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libattnres.so"
)

_LIB: ctypes.CDLL | None = None


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
        ctypes.c_int,  # n
        ctypes.c_int,  # block_size
        ctypes.c_double,  # temperature
        ctypes.c_double,  # lambda
        ctypes.POINTER(ctypes.c_double),  # out
    ]
    _LIB = lib
    return lib


def attnres_modulate_go(
    knm_flat: NDArray,
    theta: NDArray,
    n: int,
    block_size: int,
    temperature: float,
    lambda_: float,
) -> NDArray:
    """Go-backed AttnRes modulation. Signature matches the Rust FFI."""
    lib = _load_lib()
    knm_arr = np.ascontiguousarray(knm_flat, dtype=np.float64)
    theta_arr = np.ascontiguousarray(theta, dtype=np.float64)
    out = np.zeros(n * n, dtype=np.float64)
    rc = lib.AttnResModulate(
        knm_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        theta_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(block_size),
        ctypes.c_double(temperature),
        ctypes.c_double(lambda_),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(
            f"Go AttnResModulate returned error code {rc} "
            f"(invalid shape or hyperparameter)"
        )
    return out
