# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for partial information decomposition

"""Go backend for ``monitor/pid.py`` via ``libpid.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._pid_validation import (
    validate_pid_backend_inputs,
    validate_pid_scalar_output,
)

__all__ = ["pid_decomposition_go"]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libpid.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libpid.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libpid.so pid.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.PidDecomposition.restype = ctypes.c_int
    lib.PidDecomposition.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_longlong),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def pid_decomposition_go(
    phase_history_flat: FloatArray,
    t: int,
    n: int,
    group_a: IntArray,
    group_b: IntArray,
    n_bins: int,
) -> tuple[float, float]:
    """Compute (redundancy, synergy) through the Go backend."""
    history, t, n, group_a_idx, group_b_idx, bins = validate_pid_backend_inputs(
        phase_history_flat, t, n, group_a, group_b, n_bins
    )
    lib = _load_lib()
    red = ctypes.c_double(0.0)
    syn = ctypes.c_double(0.0)
    rc = lib.PidDecomposition(
        history.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(t)),
        ctypes.c_int(int(n)),
        group_a_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(group_a_idx.size)),
        group_b_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        ctypes.c_int(int(group_b_idx.size)),
        ctypes.c_int(int(bins)),
        ctypes.byref(red),
        ctypes.byref(syn),
    )
    if rc != 0:
        raise ValueError(f"Go PidDecomposition rc={rc}")
    return (
        validate_pid_scalar_output(red.value, name="redundancy"),
        validate_pid_scalar_output(syn.value, name="synergy"),
    )
