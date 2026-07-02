# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for inter-trial phase coherence

"""Go backend for ``monitor/itpc.py`` via ``libitpc.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._itpc_validation import (
    expected_compute_itpc_backend_output,
    expected_itpc_persistence_backend_output,
    validate_compute_itpc_backend_inputs,
    validate_compute_itpc_backend_output,
    validate_itpc_persistence_backend_inputs,
    validate_itpc_persistence_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = ["compute_itpc_go", "itpc_persistence_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libitpc.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libitpc.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libitpc.so itpc.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.ComputeITPC.restype = ctypes.c_int
    lib.ComputeITPC.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ITPCPersistence.restype = ctypes.c_int
    lib.ITPCPersistence.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def compute_itpc_go(phases_flat: FloatArray, n_trials: int, n_tp: int) -> FloatArray:
    """Compute inter-trial phase coherence through the Go backend."""
    p, n_trials, n_tp = validate_compute_itpc_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
    )
    if n_trials == 0 or n_tp == 0:
        return np.zeros(n_tp, dtype=np.float64)
    lib = _load_lib()
    out = np.zeros(n_tp, dtype=np.float64)
    rc = lib.ComputeITPC(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n_trials)),
        ctypes.c_int(int(n_tp)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go ComputeITPC rc={rc}")
    expected = expected_compute_itpc_backend_output(p, n_trials, n_tp)
    return validate_compute_itpc_backend_output(out, n_tp, expected=expected)


def itpc_persistence_go(
    phases_flat: FloatArray,
    n_trials: int,
    n_tp: int,
    pause_indices: IntArray,
) -> float:
    """Compute inter-trial phase-coherence persistence through the Go backend."""
    p, n_trials, n_tp, idx = validate_itpc_persistence_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
        pause_indices,
    )
    if idx.size == 0 or n_trials == 0 or n_tp == 0:
        return 0.0
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.ITPCPersistence(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n_trials)),
        ctypes.c_int(int(n_tp)),
        idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(idx.size)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go ITPCPersistence rc={rc}")
    expected = expected_itpc_persistence_backend_output(p, n_trials, n_tp, idx)
    return validate_itpc_persistence_backend_output(out.value, expected=expected)
