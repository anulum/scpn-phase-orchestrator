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

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_julia import _ensure
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
    """Run the Doppler schedule through Julia."""
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
            jl.upde_run_doppler_schedule(
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
