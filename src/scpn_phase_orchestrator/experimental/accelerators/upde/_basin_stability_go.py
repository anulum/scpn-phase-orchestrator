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

from scpn_phase_orchestrator.upde._basin_stability_validation import (
    validate_basin_stability_inputs,
    validate_basin_stability_output,
)

from .._go_runtime import load_go_library

__all__ = ["steady_state_r_go"]

FloatArray = NDArray[np.float64]

_LIB_PATH = Path(__file__).resolve().parents[5] / "go" / "libbasin_stability.so"
_LIB: ctypes.CDLL | None = None


def _load_lib() -> ctypes.CDLL:
    """Load the compiled Go backend shared library, else raise."""
    global _LIB
    if _LIB is not None:
        return _LIB
    if not _LIB_PATH.exists():
        raise ImportError(
            f"libbasin_stability.so not found at {_LIB_PATH}. Build with: "
            f"cd go && go build -buildmode=c-shared "
            f"-o libbasin_stability.so basin_stability.go"
        )
    lib = load_go_library(_LIB_PATH)
    if not hasattr(lib, "SteadyStateRV2"):
        raise ImportError(
            "Go basin library lacks the checked SteadyStateRV2 ABI; rebuild it"
        )
    lib.SteadyStateRV2.restype = ctypes.c_double
    lib.SteadyStateRV2.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
        ctypes.c_longlong,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_longlong,
        ctypes.c_longlong,
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
    """Measure one finite-window Kuramoto trial through the original Go runtime.

    Parameters
    ----------
    phases_init, omegas : numpy.ndarray
        Finite real phases in radians and frequencies in rad/s, N entries.
    knm_flat, alpha_flat : numpy.ndarray
        Row-major N*N target/source rate coupling and radian phase lags.
    n : int
        Positive population count matching every supplied buffer.
    k_scale : float
        Finite coupling multiplier, without implicit population normalization.
    dt : float
        Finite positive timestep in seconds.
    n_transient, n_measure : int
        Nonnegative discarded and post-step measurement counts.

    Returns
    -------
    float
        Mean post-step R in [0,1]; the zero-window identity is zero.

    Raises
    ------
    ImportError
        If a required runtime artifact is unavailable.
    TypeError
        If input or output payloads contain unsupported numerical aliases.
    ValueError
        If shapes, finite domains, metadata or native arithmetic fail.
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
    if any(value > 2**63 - 1 for value in (n_i, n_transient_i, n_measure_i)):
        raise ValueError("Go metadata must fit signed 64-bit integers")
    lib = _load_lib()
    r = lib.SteadyStateRV2(
        p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        o.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_ulonglong(p.size),
        ctypes.c_ulonglong(o.size),
        ctypes.c_ulonglong(k.size),
        ctypes.c_ulonglong(a.size),
        ctypes.c_longlong(n_i),
        ctypes.c_double(k_scale_f),
        ctypes.c_double(dt_f),
        ctypes.c_longlong(n_transient_i),
        ctypes.c_longlong(n_measure_i),
    )
    return validate_basin_stability_output(r)
