# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for swarmalator stepper

"""Go backend for ``upde/swarmalator.py`` via ``libswarmalator.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._swarmalator_validation import (
    validate_swarmalator_inputs,
    validate_swarmalator_output,
)

__all__ = ["swarmalator_step_go"]

FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libswarmalator.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libswarmalator.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libswarmalator.so "
            f"swarmalator.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SwarmalatorStep.restype = ctypes.c_int
    lib.SwarmalatorStep.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def swarmalator_step_go(
    pos: FloatArray,
    phases: FloatArray,
    omegas: FloatArray,
    n: int,
    dim: int,
    a: float,
    b: float,
    j: float,
    k: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance one swarmalator position-phase step.

    The calculation is delegated to the Go backend.
    """
    p, ph, om, n_i, dim_i, a_f, b_f, j_f, k_f, dt_f = validate_swarmalator_inputs(
        pos,
        phases,
        omegas,
        n,
        dim,
        a,
        b,
        j,
        k,
        dt,
    )
    lib = _load_lib()
    pos_flat = np.ascontiguousarray(p.ravel(), dtype=np.float64)
    new_pos: FloatArray = np.zeros(n_i * dim_i, dtype=np.float64)
    new_phases: FloatArray = np.zeros(n_i, dtype=np.float64)
    rc = lib.SwarmalatorStep(
        pos_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        om.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_i),
        ctypes.c_int(dim_i),
        ctypes.c_double(a_f),
        ctypes.c_double(b_f),
        ctypes.c_double(j_f),
        ctypes.c_double(k_f),
        ctypes.c_double(dt_f),
        new_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        new_phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SwarmalatorStep rc={rc}")
    return validate_swarmalator_output(new_pos, new_phases, n=n_i, dim=dim_i)
