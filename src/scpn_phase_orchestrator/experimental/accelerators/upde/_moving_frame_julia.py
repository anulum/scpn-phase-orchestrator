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

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia import (
    _ensure,
    _invoke,
)
from scpn_phase_orchestrator.upde.moving_frame import (
    _expected_positions_from_schedule,
    validate_moving_frame_backend_inputs,
    validate_moving_frame_backend_output,
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
    """Run the admitted UPDE integration through the actual Julia backend.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    positions : object
        Absolute axial coordinates per oscillator, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Finite phase lag in radians, scalar or shape ``(N, N)``; use zero for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
    spatial_k_base : float
        Base coupling strength before spatial modulation.
    spatial_decay_form : object
        Spatial decay law name (e.g. ``exponential`` or ``power``).
    spatial_decay_exponent : float
        Exponent of the spatial decay law.
    spatial_decay_length_scale : float
        Characteristic length scale of the spatial decay.
    spatial_epsilon : float
        Numerical floor guarding the spatial-decay denominator.
    doppler_strength : float
        Doppler coupling-correction strength.
    doppler_epsilon : float
        Numerical floor guarding the Doppler denominator.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    dt : float
        Integration step size.
    method : str
        Integration method (``euler``, ``rk4``, or ``rk45``).
    n_substeps : int
        Number of inner substeps per outer step.
    atol : float
        Absolute tolerance for the adaptive (rk45) integrator.
    rtol : float
        Relative tolerance for the adaptive (rk45) integrator.

    Returns
    -------
    FloatArray
        Flat final phases followed by positions from the Julia moving-frame schedule.

    Notes
    -----
    Native Julia DomainError from numerical divergence is translated to
    ValueError. Other Julia exceptions retain their native identity.
    """
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
    expected_positions = _expected_positions_from_schedule(z, velocities, dt_f)
    return validate_moving_frame_backend_output(
        np.asarray(
            _invoke(
                jl.upde_run_moving_frame_schedule,
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
        ),
        n=int(p.size),
        expected_positions=expected_positions,
    )
