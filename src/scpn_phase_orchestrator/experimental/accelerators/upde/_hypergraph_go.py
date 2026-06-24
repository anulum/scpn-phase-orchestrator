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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._hypergraph_validation import (
    TWO_PI,
    validate_hypergraph_inputs,
    validate_hypergraph_output,
)

__all__ = ["hypergraph_run_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libhypergraph.so"
_LIB: ctypes.CDLL | None = None
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
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
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def hypergraph_run_go(
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    edge_nodes: IntArray,
    edge_offsets: IntArray,
    edge_strengths: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate hypergraph Kuramoto dynamics.

    The calculation is delegated to the Go backend.
    """
    (
        p,
        o,
        n_i,
        en,
        eo,
        es,
        knm,
        alpha,
        zeta_f,
        psi_f,
        dt_f,
        steps_i,
    ) = validate_hypergraph_inputs(
        phases,
        omegas,
        n,
        edge_nodes,
        edge_offsets,
        edge_strengths,
        knm_flat,
        alpha_flat,
        zeta,
        psi,
        dt,
        n_steps,
    )
    if steps_i == 0:
        return np.mod(p, TWO_PI)
    lib = _load_lib()
    out = np.zeros(n_i, dtype=np.float64)
    knm_ptr = (
        knm.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if knm.size
        else ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
    )
    alpha_ptr = (
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if alpha.size
        else ctypes.cast(0, ctypes.POINTER(ctypes.c_double))
    )
    rc = lib.HypergraphRun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_i),
        en.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        eo.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        es.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(en.size)),
        ctypes.c_int(int(eo.size)),
        knm_ptr,
        ctypes.c_int(int(knm.size)),
        alpha_ptr,
        ctypes.c_int(int(alpha.size)),
        ctypes.c_double(zeta_f),
        ctypes.c_double(psi_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(steps_i),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go HypergraphRun rc={rc}")
    return validate_hypergraph_output(out, n=n_i)
