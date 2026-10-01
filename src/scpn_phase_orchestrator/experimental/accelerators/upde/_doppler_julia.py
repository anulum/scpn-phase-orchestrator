# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Julia bridge for Doppler UPDE

"""Julia backend for Doppler-corrected UPDE schedule runs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia import (
    _ensure,
    _invoke,
)
from scpn_phase_orchestrator.upde.doppler import (
    validate_doppler_backend_inputs,
    validate_doppler_backend_output,
)

__all__ = ["doppler_run_julia"]
FloatArray: TypeAlias = NDArray[np.float64]


def doppler_run_julia(
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
    """Run the admitted UPDE integration through the actual Julia backend.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    omega_schedule : object
        Per-step natural-frequency vectors, shape ``(n_steps, N)``.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Finite phase lag in radians, scalar or shape ``(N, N)``; use zero for no lag.
    velocity_schedule : object
        Per-step axial velocity vectors, shape ``(n_steps, N)``.
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
        The final phases after running the Doppler schedule in Julia.

    Notes
    -----
    Native Julia DomainError from numerical divergence is translated to
    ValueError. Other Julia exceptions retain their native identity.
    """
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
    jl = _ensure()
    if not hasattr(jl, "upde_run_doppler_schedule"):
        raise ImportError("Julia upde_run_doppler_schedule is not available")
    return validate_doppler_backend_output(
        np.asarray(
            _invoke(
                jl.upde_run_doppler_schedule,
                p,
                omega,
                k.ravel(),
                a.ravel(),
                velocities,
                int(p.size),
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
            ),
            dtype=np.float64,
        ),
        n=int(p.size),
    )
