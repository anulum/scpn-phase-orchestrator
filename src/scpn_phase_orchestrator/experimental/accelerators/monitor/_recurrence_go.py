# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for recurrence kernels

"""Go backend for ``monitor/recurrence.py`` via ``librecurrence.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._recurrence_validation import (
    expected_recurrence_backend_output,
    validate_cross_recurrence_backend_inputs,
    validate_recurrence_backend_inputs,
    validate_recurrence_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
ByteArray: TypeAlias = NDArray[np.uint8]

__all__ = ["cross_recurrence_matrix_go", "recurrence_matrix_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "librecurrence.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"librecurrence.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o librecurrence.so "
            f"recurrence.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.RecurrenceMatrix.restype = ctypes.c_int
    lib.RecurrenceMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    lib.CrossRecurrenceMatrix.restype = ctypes.c_int
    lib.CrossRecurrenceMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ubyte),
    ]
    _LIB = lib
    return lib


def recurrence_matrix_go(
    traj_flat: FloatArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> ByteArray:
    """Compute the recurrence matrix through the Go backend."""
    p, t_int, d_int, radius, angular_flag = validate_recurrence_backend_inputs(
        traj_flat,
        t,
        d,
        epsilon,
        angular,
    )
    lib = _load_lib()
    out = np.zeros(t_int * t_int, dtype=np.uint8)
    rc = lib.RecurrenceMatrix(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_int),
        ctypes.c_int(d_int),
        ctypes.c_double(radius),
        ctypes.c_int(int(angular_flag)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    if rc != 0:
        raise ValueError(f"Go RecurrenceMatrix rc={rc}")
    return validate_recurrence_backend_output(
        out,
        t=t_int,
        name="recurrence_matrix",
        expected=expected_recurrence_backend_output(
            p,
            p,
            t=t_int,
            d=d_int,
            epsilon=radius,
            angular=angular_flag,
        ),
    )


def cross_recurrence_matrix_go(
    traj_a_flat: FloatArray,
    traj_b_flat: FloatArray,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> ByteArray:
    """Compute the cross-recurrence matrix through the Go backend."""
    (
        a,
        b,
        t_int,
        d_int,
        radius,
        angular_flag,
    ) = validate_cross_recurrence_backend_inputs(
        traj_a_flat,
        traj_b_flat,
        t,
        d,
        epsilon,
        angular,
    )
    lib = _load_lib()
    out = np.zeros(t_int * t_int, dtype=np.uint8)
    rc = lib.CrossRecurrenceMatrix(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_int),
        ctypes.c_int(d_int),
        ctypes.c_double(radius),
        ctypes.c_int(int(angular_flag)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    if rc != 0:
        raise ValueError(f"Go CrossRecurrenceMatrix rc={rc}")
    return validate_recurrence_backend_output(
        out,
        t=t_int,
        name="cross_recurrence_matrix",
        expected=expected_recurrence_backend_output(
            a,
            b,
            t=t_int,
            d=d_int,
            epsilon=radius,
            angular=angular_flag,
        ),
    )
