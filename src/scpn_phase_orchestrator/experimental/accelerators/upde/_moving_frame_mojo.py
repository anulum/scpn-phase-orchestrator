# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for moving-frame UPDE

"""Mojo backend for moving-frame UPDE schedule runs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo import (
    _METHOD_IDS,
    _run,
)
from scpn_phase_orchestrator.upde.moving_frame import (
    _expected_positions_from_schedule,
    validate_moving_frame_backend_inputs,
    validate_moving_frame_backend_output,
)

__all__ = ["moving_frame_run_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]


def moving_frame_run_mojo(
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
    """Run the moving-frame schedule through the Mojo executable."""
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
    n = int(p.size)
    tokens: list[str] = [
        "RUN_MOVING_FRAME",
        str(n),
        repr(k_base),
        str(decay_code),
        repr(decay_exponent),
        repr(decay_length_scale),
        repr(spatial_eps),
        repr(strength),
        repr(doppler_eps),
        repr(zeta_f),
        repr(psi_f),
        repr(dt_f),
        str(n_steps_i),
        str(_METHOD_IDS[method_s]),
        str(n_substeps_i),
        repr(atol_f),
        repr(rtol_f),
    ]
    tokens.extend(repr(float(x)) for x in p.tolist())
    tokens.extend(repr(float(x)) for x in z.tolist())
    tokens.extend(repr(float(x)) for x in omega.ravel().tolist())
    tokens.extend(repr(float(x)) for x in k.ravel().tolist())
    tokens.extend(repr(float(x)) for x in a.ravel().tolist())
    tokens.extend(repr(float(x)) for x in velocities.ravel().tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=2 * n)
    expected_positions = _expected_positions_from_schedule(z, velocities, dt_f)
    return validate_moving_frame_backend_output(
        np.array(result, dtype=np.float64),
        n=n,
        expected_positions=expected_positions,
    )
