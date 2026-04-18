# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Python reference kernel for the UPDE engine

"""NumPy reference implementation of the UPDE batched integrator.

Mirrors ``spo-kernel/crates/spo-engine/src/upde.rs`` line-for-line —
forward Euler, classic RK4, Dormand-Prince RK45 with PI step-size
control, a single ``% 2π`` wrap per outer step, and ``n_substeps``
support for the fixed-step methods.

The module is private to :mod:`scpn_phase_orchestrator.upde`. The
module-level dispatcher in :mod:`scpn_phase_orchestrator.upde._run`
invokes :func:`upde_run_python` when every non-Python backend loader
fails.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["upde_run_python"]


# Dormand-Prince (1980) Butcher tableau — shared semantics with
# `spo-engine/src/dp_tableau.rs`.
_DP_A21 = 1 / 5
_DP_A31, _DP_A32 = 3 / 40, 9 / 40
_DP_A41, _DP_A42, _DP_A43 = 44 / 45, -56 / 15, 32 / 9
_DP_A51, _DP_A52 = 19372 / 6561, -25360 / 2187
_DP_A53, _DP_A54 = 64448 / 6561, -212 / 729
_DP_A61, _DP_A62, _DP_A63, _DP_A64, _DP_A65 = (
    9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656,
)
_DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_DP_B4 = (
    5179 / 57600, 0.0, 7571 / 16695, 393 / 640,
    -92097 / 339200, 187 / 2100, 1 / 40,
)


def _compute_derivative(
    theta: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
) -> NDArray:
    diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
    coupling = np.sum(knm * np.sin(diff), axis=1)
    driving = zeta * np.sin(psi - theta) if zeta != 0.0 else 0.0
    return cast("NDArray", omegas + coupling + driving)


def _dp_stages(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
) -> tuple[NDArray, NDArray]:
    """Seven Dormand-Prince stages; returns ``(y5, y4)``."""
    k1 = _compute_derivative(phases, omegas, knm, alpha, zeta, psi)
    k2 = _compute_derivative(
        phases + dt * _DP_A21 * k1, omegas, knm, alpha, zeta, psi,
    )
    k3 = _compute_derivative(
        phases + dt * (_DP_A31 * k1 + _DP_A32 * k2),
        omegas, knm, alpha, zeta, psi,
    )
    k4 = _compute_derivative(
        phases + dt * (_DP_A41 * k1 + _DP_A42 * k2 + _DP_A43 * k3),
        omegas, knm, alpha, zeta, psi,
    )
    k5 = _compute_derivative(
        phases + dt * (
            _DP_A51 * k1 + _DP_A52 * k2 + _DP_A53 * k3 + _DP_A54 * k4
        ),
        omegas, knm, alpha, zeta, psi,
    )
    k6 = _compute_derivative(
        phases + dt * (
            _DP_A61 * k1 + _DP_A62 * k2 + _DP_A63 * k3
            + _DP_A64 * k4 + _DP_A65 * k5
        ),
        omegas, knm, alpha, zeta, psi,
    )
    y5 = phases + dt * (
        _DP_B5[0] * k1 + _DP_B5[2] * k3 + _DP_B5[3] * k4
        + _DP_B5[4] * k5 + _DP_B5[5] * k6
    )
    k7 = _compute_derivative(y5, omegas, knm, alpha, zeta, psi)
    y4 = phases + dt * (
        _DP_B4[0] * k1 + _DP_B4[2] * k3 + _DP_B4[3] * k4
        + _DP_B4[4] * k5 + _DP_B4[5] * k6 + _DP_B4[6] * k7
    )
    return y5, y4


def _rk45_step(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    atol: float,
    rtol: float,
    dt_config: float,
    last_dt: float,
) -> tuple[NDArray, float]:
    """One Dormand-Prince step with PI step-size control."""
    dt = last_dt
    for _ in range(4):
        y5, y4 = _dp_stages(phases, omegas, knm, alpha, zeta, psi, dt)
        err = np.abs(y5 - y4)
        scale = atol + rtol * np.maximum(np.abs(phases), np.abs(y5))
        err_norm = float(np.max(err / scale))
        if err_norm <= 1.0:
            factor = (
                min(0.9 * err_norm ** (-0.2), 5.0) if err_norm > 0.0 else 5.0
            )
            return y5, min(dt * factor, dt_config * 10.0)
        dt = dt * max(0.9 * err_norm ** (-0.25), 0.2)
    return y5, dt


def _rk4_substep(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
) -> NDArray:
    k1 = _compute_derivative(phases, omegas, knm, alpha, zeta, psi)
    k2 = _compute_derivative(
        phases + 0.5 * dt * k1, omegas, knm, alpha, zeta, psi
    )
    k3 = _compute_derivative(
        phases + 0.5 * dt * k2, omegas, knm, alpha, zeta, psi
    )
    k4 = _compute_derivative(
        phases + dt * k3, omegas, knm, alpha, zeta, psi
    )
    return cast(
        "NDArray", phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    )


def upde_run_python(
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> NDArray:
    """Python fallback matching the Rust kernel exactly."""
    if method not in ("euler", "rk4", "rk45"):
        raise ValueError(f"unknown method {method!r}")
    if n_substeps < 1:
        raise ValueError("n_substeps must be ≥ 1")
    phases = phases.copy()
    last_dt = dt
    sub_dt = dt / n_substeps
    for _ in range(n_steps):
        if method == "rk45":
            phases, last_dt = _rk45_step(
                phases, omegas, knm, alpha, zeta, psi,
                atol, rtol, dt, last_dt,
            )
        elif method == "rk4":
            for _ in range(n_substeps):
                phases = _rk4_substep(
                    phases, omegas, knm, alpha, zeta, psi, sub_dt
                )
        else:  # euler
            for _ in range(n_substeps):
                deriv = _compute_derivative(
                    phases, omegas, knm, alpha, zeta, psi
                )
                phases = phases + sub_dt * deriv
        phases = phases % TWO_PI
    return phases
