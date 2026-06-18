# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for simplicial Kuramoto

"""Go backend for ``upde/simplicial.py`` via ``libsimplicial.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._simplicial_validation import (
    validate_simplicial_inputs,
    validate_simplicial_output,
)

__all__ = ["simplicial_run_go"]

FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libsimplicial.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libsimplicial.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libsimplicial.so simplicial.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SimplicialRun.restype = ctypes.c_int
    lib.SimplicialRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def simplicial_run_go(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    sigma2: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate pairwise-plus-simplicial Kuramoto dynamics.

    The calculation is delegated to the Go backend.
    """
    phases, omegas, knm_flat, alpha_flat, n, zeta, psi, sigma2, dt, n_steps = (
        validate_simplicial_inputs(
            phases,
            omegas,
            knm_flat,
            alpha_flat,
            n,
            zeta,
            psi,
            sigma2,
            dt,
            n_steps,
        )
    )
    if n_steps == 0:
        return phases.copy()
    lib = _load_lib()
    out: FloatArray = np.zeros(int(n), dtype=np.float64)
    rc = lib.SimplicialRun(
        phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        omegas.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        knm_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        alpha_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        ctypes.c_double(float(sigma2)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(n_steps)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SimplicialRun rc={rc}")
    return validate_simplicial_output(out, n=n)
