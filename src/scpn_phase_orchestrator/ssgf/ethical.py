# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — C15_sec ethical cost term
#
# Implements the ethical Lagrangian from R5 Insight 19:
# L_ethical = U_total + w_c15 · C15_sec
# C15_sec = (1 - J_sec) + κ · Φ_ethics
#
# J_sec = α·R + β·K + γ·Q - ν·S_dev  (SEC functional)
# Φ_ethics = Σ max(0, g_k)²           (CBF constraint penalties)
#
# Grounded in: Harsanyi aggregation, MacAskill ECW,
# Lyapunov/CBF safety, Wiener cybernetic ethics.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.spectral import fiedler_value
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

try:
    from spo_kernel import (
        compute_ethical_cost_rust as _rust_ethical_cost,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["EthicalCost", "compute_ethical_cost"]


@dataclass
class EthicalCost:
    """C15_sec ethical cost: SEC functional, CBF penalties, violations."""

    J_sec: float
    phi_ethics: float
    c15_sec: float
    constraints_violated: int


def compute_ethical_cost(
    phases: NDArray,
    knm: NDArray,
    *,
    alpha_R: float = 0.4,
    beta_K: float = 0.3,
    gamma_Q: float = 0.2,
    nu_S: float = 0.1,
    kappa: float = 1.0,
    R_min: float = 0.2,
    connectivity_min: float = 0.1,
    max_coupling: float = 5.0,
) -> EthicalCost:
    """Compute C15_sec ethical cost term.

    J_sec = α·R + β·K_norm + γ·Q - ν·S_dev
    where:
      R = Kuramoto order parameter (coherence)
      K_norm = λ₂(L) / max(λ₂) (normalized connectivity, Wiener)
      Q = 1 - sparsity (coupling quality)
      S_dev = std(phases) / π (phase deviation from uniform)

    Φ_ethics = Σ max(0, g_k)² where g_k are CBF constraint violations:
      g_1: R_min - R                   (non-harm: minimum coherence)
      g_2: connectivity_min - λ₂       (Wiener: maintain connectivity)
      g_3: max(K_ij) - max_coupling    (boundary: coupling limits)
    """
    n = len(phases)
    if n == 0:
        return EthicalCost(
            J_sec=0.0, phi_ethics=0.0, c15_sec=1.0, constraints_violated=0
        )

    if _HAS_RUST:
        p = np.ascontiguousarray(phases, dtype=np.float64)
        k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        j, phi, c15, nv = _rust_ethical_cost(
            p,
            k,
            n,
            alpha_R,
            beta_K,
            gamma_Q,
            nu_S,
            kappa,
            R_min,
            connectivity_min,
            max_coupling,
        )
        return EthicalCost(
            J_sec=j,
            phi_ethics=phi,
            c15_sec=c15,
            constraints_violated=nv,
        )

    R, _ = compute_order_parameter(phases)
    lam2 = fiedler_value(knm)
    lam2_max = float(n)  # max λ₂ for complete graph with unit weights
    K_norm = lam2 / lam2_max if lam2_max > 0 else 0.0

    n_nonzero = np.count_nonzero(knm)
    n_possible = n * (n - 1)
    Q = n_nonzero / n_possible if n_possible > 0 else 0.0

    S_dev = float(np.std(phases)) / np.pi

    J_sec = alpha_R * R + beta_K * K_norm + gamma_Q * Q - nu_S * S_dev

    # CBF constraint violations
    g = [
        R_min - R,
        connectivity_min - lam2,
        float(np.max(knm)) - max_coupling if np.any(knm > 0) else 0.0,
    ]
    violations = [max(0.0, gi) ** 2 for gi in g]
    phi_ethics = kappa * sum(violations)
    n_violated = sum(1 for gi in g if gi > 0)

    c15_sec = (1.0 - J_sec) + phi_ethics

    return EthicalCost(
        J_sec=J_sec,
        phi_ethics=phi_ethics,
        c15_sec=c15_sec,
        constraints_violated=n_violated,
    )
