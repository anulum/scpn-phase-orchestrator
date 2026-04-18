# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for hypergraph Kuramoto

"""Go backend for ``upde/hypergraph.py`` via ``libhypergraph.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["hypergraph_run_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libhypergraph.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libhypergraph.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libhypergraph.so hypergraph.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.HypergraphRun.restype = ctypes.c_int
    lib.HypergraphRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def hypergraph_run_go(
    phases: NDArray,
    omegas: NDArray,
    n: int,
    edge_nodes: NDArray,
    edge_offsets: NDArray,
    edge_strengths: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    lib = _load_lib()
    p = np.ascontiguousarray(phases, dtype=np.float64)
    o = np.ascontiguousarray(omegas, dtype=np.float64)
    en = np.ascontiguousarray(edge_nodes, dtype=np.int64)
    eo = np.ascontiguousarray(edge_offsets, dtype=np.int64)
    es = np.ascontiguousarray(edge_strengths, dtype=np.float64)
    knm = np.ascontiguousarray(knm_flat, dtype=np.float64)
    alpha = np.ascontiguousarray(alpha_flat, dtype=np.float64)
    out = np.zeros(int(n), dtype=np.float64)
    knm_ptr = (
        knm.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if knm.size else ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
    )
    alpha_ptr = (
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if alpha.size else ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
    )
    rc = lib.HypergraphRun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        en.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        eo.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        es.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(en.size)), ctypes.c_int(int(eo.size)),
        knm_ptr, ctypes.c_int(int(knm.size)),
        alpha_ptr, ctypes.c_int(int(alpha.size)),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_steps)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go HypergraphRun rc={rc}")
    return out
