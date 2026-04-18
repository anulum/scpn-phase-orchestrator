# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for inertial stepper

"""Go backend for ``upde/inertial.py`` via ``libinertial.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["inertial_step_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libinertial.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libinertial.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libinertial.so inertial.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.InertialStep.restype = ctypes.c_int
    lib.InertialStep.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def inertial_step_go(
    theta: NDArray,
    omega_dot: NDArray,
    power: NDArray,
    knm_flat: NDArray,
    inertia: NDArray,
    damping: NDArray,
    n: int,
    dt: float,
) -> tuple[NDArray, NDArray]:
    lib = _load_lib()
    th = np.ascontiguousarray(theta, dtype=np.float64)
    od = np.ascontiguousarray(omega_dot, dtype=np.float64)
    pw = np.ascontiguousarray(power, dtype=np.float64)
    km = np.ascontiguousarray(knm_flat, dtype=np.float64)
    ine = np.ascontiguousarray(inertia, dtype=np.float64)
    dmp = np.ascontiguousarray(damping, dtype=np.float64)
    new_theta = np.zeros(n, dtype=np.float64)
    new_omega_dot = np.zeros(n, dtype=np.float64)
    rc = lib.InertialStep(
        th.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        od.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        km.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ine.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)), ctypes.c_double(float(dt)),
        new_theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        new_omega_dot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go InertialStep rc={rc}")
    return new_theta, new_omega_dot
