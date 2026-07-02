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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._hodge_validation import (
    validate_hodge_backend_inputs,
    validate_hodge_backend_output,
)

__all__ = ["hodge_decomposition_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libhodge.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libhodge.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libhodge.so hodge.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.HodgeDecomposition.restype = ctypes.c_int
    lib.HodgeDecomposition.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def hodge_decomposition_go(
    knm_flat: FloatArray,
    phases: FloatArray,
    n: int,
    edges_flat: NDArray[np.int64],
    n_edges: int,
    tris_flat: NDArray[np.int64],
    n_tris: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the Hodge gradient, curl, and harmonic flow matrices with Go."""
    k, p, n, edges, n_edges, tris, n_tris = validate_hodge_backend_inputs(
        knm_flat,
        phases,
        n,
        edges_flat,
        n_edges,
        tris_flat,
        n_tris,
    )
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    lib = _load_lib()
    gradient = np.zeros(n * n, dtype=np.float64)
    curl = np.zeros(n * n, dtype=np.float64)
    harmonic = np.zeros(n * n, dtype=np.float64)
    rc = lib.HodgeDecomposition(
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        edges.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(n_edges)),
        tris.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(n_tris)),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        curl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        harmonic.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go HodgeDecomposition rc={rc}")
    return validate_hodge_backend_output(
        (gradient.reshape(n, n), curl.reshape(n, n), harmonic.reshape(n, n)),
        n=n,
    )
