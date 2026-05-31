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

from ._psychedelic_validation import (
    validate_psychedelic_backend_inputs,
    validate_psychedelic_entropy_backend_output,
)

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["entropy_from_phases_go"]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libpsychedelic.so"
_LIB: ctypes.CDLL | None = None


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

    p, bin_count = validate_psychedelic_backend_inputs(phases, n_bins)
    if p.size == 0:
        return 0.0
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
    return validate_psychedelic_entropy_backend_output(out.value, bin_count)
