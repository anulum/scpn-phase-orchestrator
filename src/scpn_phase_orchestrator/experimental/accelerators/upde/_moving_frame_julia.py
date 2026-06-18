# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for moving-frame UPDE

"""Julia backend for moving-frame UPDE schedule runs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia import _ensure
from scpn_phase_orchestrator.upde.moving_frame import (
    validate_moving_frame_backend_inputs,
)

__all__ = ["moving_frame_run_julia"]
FloatArray: TypeAlias = NDArray[np.float64]


def moving_frame_run_julia(
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
    """Run the moving-frame schedule through Julia."""
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
    jl = _ensure()
    if not hasattr(jl, "upde_run_moving_frame_schedule"):
        raise ImportError("Julia upde_run_moving_frame_schedule is not available")
    return np.ascontiguousarray(
        np.asarray(
            jl.upde_run_moving_frame_schedule(
                p,
                z,
                omega,
                k.ravel(),
                a.ravel(),
                velocities,
                int(p.size),
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
            ),
            dtype=np.float64,
        )
    )
