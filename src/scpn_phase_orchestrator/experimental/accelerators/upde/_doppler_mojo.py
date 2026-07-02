# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Mojo bridge for Doppler UPDE

"""Mojo backend for Doppler-corrected UPDE schedule runs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._engine_mojo import (
    _METHOD_IDS,
    _run,
)
from scpn_phase_orchestrator.upde.doppler import (
    validate_doppler_backend_inputs,
    validate_doppler_backend_output,
)

__all__ = ["doppler_run_mojo"]
FloatArray: TypeAlias = NDArray[np.float64]


def doppler_run_mojo(
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
    """Run the Doppler schedule through the Mojo executable."""
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
    n = int(p.size)
    tokens: list[str] = [
        "RUN_DOPPLER",
        str(n),
        repr(strength),
        repr(epsilon),
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
    tokens.extend(repr(float(x)) for x in omega.ravel().tolist())
    tokens.extend(repr(float(x)) for x in k.ravel().tolist())
    tokens.extend(repr(float(x)) for x in a.ravel().tolist())
    tokens.extend(repr(float(x)) for x in velocities.ravel().tolist())
    result = _run(" ".join(tokens) + "\n", expected_count=n)
    return validate_doppler_backend_output(np.array(result, dtype=np.float64), n=n)
