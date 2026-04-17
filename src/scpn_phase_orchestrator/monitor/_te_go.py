# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for transfer entropy

"""Go backend for ``monitor/transfer_entropy.py``. ``libtransfer_entropy.so``
via ctypes."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["phase_te_go", "te_matrix_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libtransfer_entropy.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libtransfer_entropy.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libtransfer_entropy.so "
            f"transfer_entropy.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.PhaseTransferEntropy.restype = ctypes.c_int
    lib.PhaseTransferEntropy.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.TransferEntropyMatrix.restype = ctypes.c_int
    lib.TransferEntropyMatrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def phase_te_go(
    source: NDArray, target: NDArray, n_bins: int
) -> float:
    lib = _load_lib()
    s = np.ascontiguousarray(source.ravel(), dtype=np.float64)
    t = np.ascontiguousarray(target.ravel(), dtype=np.float64)
    n = min(s.size, t.size)
    out = ctypes.c_double(0.0)
    rc = lib.PhaseTransferEntropy(
        s[:n].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        t[:n].ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_int(n_bins),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go PhaseTransferEntropy rc={rc}")
    return float(out.value)


def te_matrix_go(
    phase_series: NDArray,
    n_osc: int,
    n_time: int,
    n_bins: int,
) -> NDArray:
    lib = _load_lib()
    series = np.ascontiguousarray(phase_series, dtype=np.float64)
    out = np.zeros(n_osc * n_osc, dtype=np.float64)
    rc = lib.TransferEntropyMatrix(
        series.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_osc),
        ctypes.c_int(n_time),
        ctypes.c_int(n_bins),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go TransferEntropyMatrix rc={rc}")
    return out
