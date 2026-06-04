# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for Doppler UPDE

"""Go backend for Doppler-corrected UPDE schedule runs."""

from __future__ import annotations

import ctypes
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_go import (
    _METHOD_IDS,
    _load_lib,
)
from scpn_phase_orchestrator.upde.doppler import validate_doppler_backend_inputs

__all__ = ["doppler_run_go"]
FloatArray: TypeAlias = NDArray[np.float64]


def _configure_symbol(lib: ctypes.CDLL) -> None:
    if not hasattr(lib, "UPDERunDopplerSchedule"):
        raise ImportError("Go UPDERunDopplerSchedule symbol is not available")
    lib.UPDERunDopplerSchedule.restype = ctypes.c_int
    lib.UPDERunDopplerSchedule.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
    ]


def doppler_run_go(
    phases: FloatArray,
    omega_schedule: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    velocity_schedule: FloatArray,
    doppler_strength: float,
    doppler_epsilon: float,
    zeta: float,
    psi: float,
    dt: float,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> FloatArray:
    """Run the Doppler schedule through the Go shared library."""

    (
        p,
        omega,
        k,
        a,
        velocities,
        strength,
        epsilon,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_doppler_backend_inputs(
        phases,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        doppler_strength,
        doppler_epsilon,
        zeta,
        psi,
        dt,
        method,
        n_substeps,
        atol,
        rtol,
    )
    lib = _load_lib()
    _configure_symbol(lib)
    n = int(p.size)
    p_work = p.copy()
    rc = lib.UPDERunDopplerSchedule(
        p_work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        omega.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(strength),
        ctypes.c_double(epsilon),
        ctypes.c_double(zeta_f),
        ctypes.c_double(psi_f),
        ctypes.c_double(dt_f),
        ctypes.c_int(n_steps_i),
        ctypes.c_int(_METHOD_IDS[method_s]),
        ctypes.c_int(n_substeps_i),
        ctypes.c_double(atol_f),
        ctypes.c_double(rtol_f),
    )
    if rc != 0:
        raise ValueError(f"Go UPDERunDopplerSchedule rc={rc}")
    return np.ascontiguousarray(p_work, dtype=np.float64)
