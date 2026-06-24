# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Go bridge for moving-frame UPDE

"""Go backend for moving-frame UPDE schedule runs."""

from __future__ import annotations

import ctypes
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_go import (
    _METHOD_IDS,
    _load_lib,
)
from scpn_phase_orchestrator.upde.moving_frame import (
    validate_moving_frame_backend_inputs,
)

__all__ = ["moving_frame_run_go"]
FloatArray: TypeAlias = NDArray[np.float64]


def _configure_symbol(lib: ctypes.CDLL) -> None:
    """Configure the loaded backend symbol's argument and return types."""
    if not hasattr(lib, "UPDERunMovingFrameSchedule"):
        raise ImportError("Go UPDERunMovingFrameSchedule symbol is not available")
    lib.UPDERunMovingFrameSchedule.restype = ctypes.c_int
    lib.UPDERunMovingFrameSchedule.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
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


def moving_frame_run_go(
    phases: FloatArray,
    positions: FloatArray,
    omega_schedule: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    velocity_schedule: FloatArray,
    spatial_k_base: float,
    spatial_decay_form: int,
    spatial_decay_exponent: float,
    spatial_decay_length_scale: float,
    spatial_epsilon: float,
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
    """Run the moving-frame schedule through the Go shared library."""
    (
        p,
        z,
        omega,
        k,
        a,
        velocities,
        k_base,
        decay_code,
        decay_exponent,
        decay_length_scale,
        spatial_eps,
        strength,
        doppler_eps,
        zeta_f,
        psi_f,
        dt_f,
        n_steps_i,
        method_s,
        n_substeps_i,
        atol_f,
        rtol_f,
    ) = validate_moving_frame_backend_inputs(
        phases,
        positions,
        omega_schedule,
        knm,
        alpha,
        velocity_schedule,
        spatial_k_base,
        spatial_decay_form,
        spatial_decay_exponent,
        spatial_decay_length_scale,
        spatial_epsilon,
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
    z_work = z.copy()
    rc = lib.UPDERunMovingFrameSchedule(
        p_work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z_work.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        omega.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        k.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        velocities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n),
        ctypes.c_double(k_base),
        ctypes.c_int(decay_code),
        ctypes.c_double(decay_exponent),
        ctypes.c_double(decay_length_scale),
        ctypes.c_double(spatial_eps),
        ctypes.c_double(strength),
        ctypes.c_double(doppler_eps),
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
        raise ValueError(f"Go UPDERunMovingFrameSchedule rc={rc}")
    return np.ascontiguousarray(np.concatenate([p_work, z_work]), dtype=np.float64)
