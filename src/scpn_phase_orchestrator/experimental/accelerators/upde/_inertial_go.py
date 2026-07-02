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
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._inertial_validation import (
    validate_inertial_inputs,
    validate_inertial_output,
)

__all__ = ["inertial_step_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libinertial.so"
_LIB: ctypes.CDLL | None = None
Float64Array: TypeAlias = NDArray[np.float64]


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libinertial.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libinertial.so inertial.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.InertialStep.restype = ctypes.c_int
    lib.InertialStep.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def inertial_step_go(
    theta: Float64Array,
    omega_dot: Float64Array,
    power: Float64Array,
    knm_flat: Float64Array,
    inertia: Float64Array,
    damping: Float64Array,
    n: int,
    dt: float,
) -> tuple[Float64Array, Float64Array]:
    """Advance one inertial Kuramoto step.

    The calculation is delegated to the Go backend.
    """
    th, od, pw, km, ine, dmp, n_i, dt_f = validate_inertial_inputs(
        theta,
        omega_dot,
        power,
        knm_flat,
        inertia,
        damping,
        n,
        dt,
    )
    lib = _load_lib()
    new_theta = np.zeros(n_i, dtype=np.float64)
    new_omega_dot = np.zeros(n_i, dtype=np.float64)
    rc = lib.InertialStep(
        th.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        od.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        km.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ine.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dmp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_i),
        ctypes.c_double(dt_f),
        new_theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        new_omega_dot.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go InertialStep rc={rc}")
    return validate_inertial_output(new_theta, new_omega_dot, n=n_i)
