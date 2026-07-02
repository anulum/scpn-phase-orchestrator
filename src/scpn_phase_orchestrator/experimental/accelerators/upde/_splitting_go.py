# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Strang splitting

"""Go backend for ``upde/splitting.py`` via ``libsplitting.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .._go_runtime import load_go_library
from ._splitting_validation import (
    validate_splitting_inputs,
    validate_splitting_output,
)

__all__ = ["splitting_run_go"]

FloatArray: TypeAlias = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libsplitting.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libsplitting.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libsplitting.so splitting.go"
        )
    lib = load_go_library(_LIB_PATH)
    lib.SplittingRun.restype = ctypes.c_int
    lib.SplittingRun.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
    ]
    _LIB = lib
    return lib


def splitting_run_go(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate Strang-split phase dynamics.

    The calculation is delegated to the Go backend.
    """
    p, o, k, a, n_i, zeta_f, psi_f, dt_f, n_steps_i = validate_splitting_inputs(
        phases,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        zeta,
        psi,
        dt,
        n_steps,
    )
    lib = _load_lib()
    out: FloatArray = np.zeros(n_i, dtype=np.float64)
    rc = lib.SplittingRun(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_i),
        ctypes.c_double(zeta_f),
        ctypes.c_double(psi_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(n_steps_i),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if rc != 0:
        raise ValueError(f"Go SplittingRun rc={rc}")
    return validate_splitting_output(out, n=n_i)
