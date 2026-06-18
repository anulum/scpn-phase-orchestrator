# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for PAC

"""Go backend for ``upde/pac.py``. Calls ``libpac.so`` via ctypes."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._pac_validation import (
    validate_modulation_index_inputs,
    validate_modulation_index_output,
    validate_pac_matrix_inputs,
    validate_pac_matrix_output,
)

__all__ = ["modulation_index_go", "pac_matrix_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libpac.so"
_LIB: ctypes.CDLL | None = None

FloatArray: TypeAlias = NDArray[np.float64]


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libpac.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libpac.so pac.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ModulationIndex.restype = ctypes.c_int
    lib.ModulationIndex.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.PACMatrix.restype = ctypes.c_int
    lib.PACMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def modulation_index_go(
    theta_low: FloatArray, amp_high: FloatArray, n_bins: int
) -> float:
    """Compute phase-amplitude coupling modulation index.

    The calculation is delegated to the Go backend.
    """
    t, a, bins = validate_modulation_index_inputs(theta_low, amp_high, n_bins)
    n = t.size
    if n == 0:
        return 0.0
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.ModulationIndex(
        t.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(bins),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go ModulationIndex rc={rc}")
    return validate_modulation_index_output(out.value)


def pac_matrix_go(
    phases_flat: FloatArray,
    amplitudes_flat: FloatArray,
    t: int,
    n: int,
    n_bins: int,
) -> FloatArray:
    """Compute the phase-amplitude coupling matrix.

    The calculation is delegated to the Go backend.
    """
    p, a, t_i, n_i, bins = validate_pac_matrix_inputs(
        phases_flat,
        amplitudes_flat,
        t,
        n,
        n_bins,
    )
    lib = _load_lib()
    out = np.zeros(n_i * n_i, dtype=np.float64)
    rc = lib.PACMatrix(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(t_i),
        ctypes.c_int(n_i),
        ctypes.c_int(bins),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go PACMatrix rc={rc}")
    return validate_pac_matrix_output(out, n=n_i)
