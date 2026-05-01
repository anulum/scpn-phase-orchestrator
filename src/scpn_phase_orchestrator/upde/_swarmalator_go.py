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

__all__ = ["swarmalator_step_go"]

FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[3] / "go" / "libswarmalator.so"
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
    lib = _load_lib()
    p = np.ascontiguousarray(pos.ravel(), dtype=np.float64)
    ph = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    om = np.ascontiguousarray(omegas.ravel(), dtype=np.float64)
    new_pos: FloatArray = np.zeros(n * dim, dtype=np.float64)
    new_phases: FloatArray = np.zeros(n, dtype=np.float64)
    rc = lib.SwarmalatorStep(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        om.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        ctypes.c_int(int(dim)),
        ctypes.c_double(float(a)),
        ctypes.c_double(float(b)),
        ctypes.c_double(float(j)),
        ctypes.c_double(float(k)),
        ctypes.c_double(float(dt)),
        new_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        new_phases.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SwarmalatorStep rc={rc}")
    return new_pos.reshape(n, dim), new_phases
