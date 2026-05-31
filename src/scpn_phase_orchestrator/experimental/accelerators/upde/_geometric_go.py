# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for torus geometric integrator

"""Go backend for ``upde/geometric.py`` via ``libgeometric.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._geometric_validation import (
    TWO_PI,
    validate_torus_inputs,
    validate_torus_output,
)

__all__ = ["torus_run_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libgeometric.so"
_LIB: ctypes.CDLL | None = None
FloatArray: TypeAlias = NDArray[np.float64]


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libgeometric.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libgeometric.so geometric.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.TorusRun.restype = ctypes.c_int
    lib.TorusRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
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


def torus_run_go(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate torus phase dynamics.

    The calculation is delegated to the Go backend.
    """

    (
        p,
        o,
        k,
        a,
        n_i,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
    ) = validate_torus_inputs(
        phases,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        zeta,
        psi,
        dt,
        n_steps,
    )
    if n_steps_i == 0:
        return p % TWO_PI
    lib = _load_lib()
    out = np.zeros(n_i, dtype=np.float64)
    rc = lib.TorusRun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_i),
        ctypes.c_double(zeta_f),
        ctypes.c_double(psi_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(n_steps_i),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go TorusRun rc={rc}")
    return validate_torus_output(out, n=n_i)
