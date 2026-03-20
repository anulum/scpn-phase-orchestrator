# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adjoint sensitivity for SSGF gradient
#
# The correct implementation uses diffrax (JAX) autodiff through the ODE
# solver for exact gradients at O(1) memory cost. This module provides a
# numpy finite-difference fallback for environments without JAX.

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = ["cost_R", "gradient_knm_fd"]


def cost_R(phases: NDArray) -> float:
    """Cost = 1 - R (minimize to maximize synchronization)."""
    R, _ = compute_order_parameter(phases)
    return 1.0 - R


def gradient_knm_fd(
    engine: UPDEEngine,
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    n_steps: int = 100,
    epsilon: float = 1e-4,
    zeta: float = 0.0,
    psi: float = 0.0,
) -> NDArray:
    """Finite-difference gradient of cost_R w.r.t. knm entries.

    For each K_ij (i≠j), perturbs by ±ε and measures the effect on R
    after n_steps. Returns gradient matrix ∂(1-R)/∂K_ij.

    This is O(N² · n_steps) — the adjoint method via diffrax reduces
    this to O(n_steps) but requires JAX. This fallback is correct but
    slow for large N.
    """
    n = knm.shape[0]
    grad = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            knm_plus = knm.copy()
            knm_plus[i, j] += epsilon
            p_plus = engine.run(
                phases_init, omegas, knm_plus, zeta, psi, alpha, n_steps
            )
            c_plus = cost_R(p_plus)

            knm_minus = knm.copy()
            knm_minus[i, j] -= epsilon
            p_minus = engine.run(
                phases_init, omegas, knm_minus, zeta, psi, alpha, n_steps
            )
            c_minus = cost_R(p_minus)

            grad[i, j] = (c_plus - c_minus) / (2 * epsilon)

    return grad
