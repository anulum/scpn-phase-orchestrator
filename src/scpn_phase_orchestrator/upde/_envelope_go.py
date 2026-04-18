# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for envelope kernels

"""Go backend for ``upde/envelope.py`` via ``libenvelope.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["envelope_modulation_depth_go", "extract_envelope_go"]

_LIB_PATH = (
    Path(__file__).resolve().parents[3] / "go" / "libenvelope.so"
)
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libenvelope.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libenvelope.so "
            f"envelope.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ExtractEnvelope.restype = ctypes.c_int
    lib.ExtractEnvelope.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.EnvelopeModulationDepth.restype = ctypes.c_int
    lib.EnvelopeModulationDepth.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def extract_envelope_go(amps: NDArray, window: int) -> NDArray:
    lib = _load_lib()
    a = np.ascontiguousarray(amps.ravel(), dtype=np.float64)
    if a.size == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(a.size, dtype=np.float64)
    rc = lib.ExtractEnvelope(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(a.size)),
        ctypes.c_int(int(window)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go ExtractEnvelope rc={rc}")
    return out


def envelope_modulation_depth_go(env: NDArray) -> float:
    lib = _load_lib()
    e = np.ascontiguousarray(env.ravel(), dtype=np.float64)
    out = ctypes.c_double(0.0)
    rc = lib.EnvelopeModulationDepth(
        e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(e.size)),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go EnvelopeModulationDepth rc={rc}")
    return float(out.value)
