# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for psychedelic observables

"""Go backend for ``monitor/psychedelic.py`` via ``libpsychedelic.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["entropy_from_phases_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libpsychedelic.so"
_LIB: ctypes.CDLL | None = None


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _validated_backend_inputs(phases: object, n_bins: object) -> tuple[FloatArray, int]:
    if _contains_boolean_alias(phases):
        raise ValueError("phases must not contain boolean values")
    raw = np.asarray(phases)
    if np.iscomplexobj(raw):
        raise ValueError("phases must contain real-valued samples")
    try:
        phase_values = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite real-valued vector") from exc
    if not np.all(np.isfinite(phase_values)):
        raise ValueError("phases must contain only finite values")
    if isinstance(n_bins, (bool, np.bool_)) or not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    bin_count = int(n_bins)
    if bin_count < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return np.ascontiguousarray(phase_values, dtype=np.float64), bin_count


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libpsychedelic.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libpsychedelic.so "
            f"psychedelic.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.EntropyFromPhases.restype = ctypes.c_int
    lib.EntropyFromPhases.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def entropy_from_phases_go(phases: FloatArray, n_bins: int) -> float:
    """Compute phase-distribution entropy through the Go backend."""

    p, bin_count = _validated_backend_inputs(phases, n_bins)
    lib = _load_lib()
    out = ctypes.c_double(0.0)
    rc = lib.EntropyFromPhases(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(p.size)),
        ctypes.c_int(bin_count),
        ctypes.byref(out),
    )
    if rc != 0:
        raise ValueError(f"Go EntropyFromPhases rc={rc}")
    return float(out.value)
