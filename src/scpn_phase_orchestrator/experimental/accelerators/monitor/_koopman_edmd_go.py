# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Koopman EDMD

"""Go backend for ``monitor/koopman_edmd.py``. Loads ``libkoopman_edmd.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._koopman_edmd_validation import (
    validate_edmd_backend_inputs,
    validate_edmd_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["koopman_edmd_solve_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libkoopman_edmd.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libkoopman_edmd.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libkoopman_edmd.so "
            f"koopman_edmd.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    double_ptr = ctypes.POINTER(ctypes.c_double)
    lib.KoopmanEdmdSolve.restype = ctypes.c_int
    lib.KoopmanEdmdSolve.argtypes = [
        double_ptr,
        double_ptr,
        double_ptr,
        double_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        double_ptr,
        double_ptr,
        double_ptr,
    ]
    _LIB = lib
    return lib


def _as_double_ptr(array: FloatArray) -> ctypes._Pointer[ctypes.c_double]:
    """Return a C double-pointer view of the array for the FFI call."""
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def koopman_edmd_solve_go(
    x_lift: FloatArray,
    inputs: FloatArray,
    y_lift: FloatArray,
    states: FloatArray,
    regularisation: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Solve the EDMD-with-control least squares through the Go backend."""
    dims = validate_edmd_backend_inputs(x_lift, inputs, y_lift, states)
    lib = _load_lib()
    xl = np.ascontiguousarray(x_lift.ravel(), dtype=np.float64)
    ui = np.ascontiguousarray(inputs.ravel(), dtype=np.float64)
    yl = np.ascontiguousarray(y_lift.ravel(), dtype=np.float64)
    st = np.ascontiguousarray(states.ravel(), dtype=np.float64)
    a = np.zeros(dims.lift_dim * dims.lift_dim, dtype=np.float64)
    b = np.zeros(dims.lift_dim * dims.input_dim, dtype=np.float64)
    c = np.zeros(dims.state_dim * dims.lift_dim, dtype=np.float64)
    rc = lib.KoopmanEdmdSolve(
        _as_double_ptr(xl),
        _as_double_ptr(ui),
        _as_double_ptr(yl),
        _as_double_ptr(st),
        ctypes.c_int(dims.samples),
        ctypes.c_int(dims.lift_dim),
        ctypes.c_int(dims.input_dim),
        ctypes.c_int(dims.state_dim),
        ctypes.c_double(float(regularisation)),
        _as_double_ptr(a),
        _as_double_ptr(b),
        _as_double_ptr(c),
    )
    if rc != 0:
        raise ValueError(f"Go KoopmanEdmdSolve rc={rc}")
    return validate_edmd_backend_output(
        a.reshape(dims.lift_dim, dims.lift_dim),
        b.reshape(dims.lift_dim, dims.input_dim),
        c.reshape(dims.state_dim, dims.lift_dim),
        dims,
    )
