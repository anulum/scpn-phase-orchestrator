# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for steady-state R

"""Go backend for ``upde/basin_stability.py`` via ``libbasin_stability.so``."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._basin_stability_validation import (
    validate_basin_stability_inputs,
    validate_basin_stability_output,
)

__all__ = ["steady_state_r_go"]

FloatArray = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libbasin_stability.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libbasin_stability.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libbasin_stability.so basin_stability.go"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.SteadyStateR.restype = ctypes.c_double
    lib.SteadyStateR.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _LIB = lib
    return lib


def steady_state_r_go(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Compute steady-state order parameter for basin-stability trials.

    The calculation is delegated to the Go backend.
    """

    (
        p,
        o,
        k,
        a,
        n_i,
        k_scale_f,
        dt_f,
        n_transient_i,
        n_measure_i,
    ) = validate_basin_stability_inputs(
        phases_init,
        omegas,
        knm_flat,
        alpha_flat,
        n,
        k_scale,
        dt,
        n_transient,
        n_measure,
    )
    if n_measure_i == 0:
        return 0.0
    lib = _load_lib()
    r = lib.SteadyStateR(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_i),
        ctypes.c_double(k_scale_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(n_transient_i),
        ctypes.c_int(n_measure_i),
    )
    return validate_basin_stability_output(r)
