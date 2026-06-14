# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for delayed Kuramoto

"""Go backend for ``upde/delay.py`` via ``libdelay.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._delay_validation import validate_delay_backend_inputs

__all__ = ["delayed_kuramoto_run_go"]
FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libdelay.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libdelay.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared -o libdelay.so delay.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.DelayedKuramotoRun.restype = ctypes.c_int
    lib.DelayedKuramotoRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def delayed_kuramoto_run_go(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    delay_steps: int,
    n_steps: int,
) -> FloatArray:
    """Integrate delayed Kuramoto dynamics through the Go backend."""
    ph, om, knm, alpha, n, zeta, psi, dt, delay_steps, n_steps = (
        validate_delay_backend_inputs(
            phases, omegas, knm_flat, alpha_flat, n, zeta, psi, dt, delay_steps, n_steps
        )
    )
    lib = _load_lib()
    out = np.zeros(n, dtype=np.float64)
    rc = lib.DelayedKuramotoRun(
        ph.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        om.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        knm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        alpha.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(int(n)),
        ctypes.c_double(float(zeta)),
        ctypes.c_double(float(psi)),
        ctypes.c_double(float(dt)),
        ctypes.c_int(int(delay_steps)),
        ctypes.c_int(int(n_steps)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go DelayedKuramotoRun rc={rc}")
    return out
