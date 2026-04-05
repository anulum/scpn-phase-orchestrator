# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF cost terms

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.spectral import fiedler_value
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

try:
    from spo_kernel import (  # type: ignore[import-untyped]
        compute_ssgf_costs_rust as _rust_costs,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["SSGFCosts", "compute_ssgf_costs"]


@dataclass
class SSGFCosts:
    c1_sync: float
    c2_spectral_gap: float
    c3_sparsity: float
    c4_symmetry: float
    u_total: float


def compute_ssgf_costs(
    W: NDArray,
    phases: NDArray,
    weights: tuple[float, ...] = (1.0, 0.5, 0.1, 0.1),
) -> SSGFCosts:
    """Compute SSGF cost terms for geometry W given current phases.

    C1: 1 - R (synchronization deficit)
    C2: -λ₂(L(W)) (negative algebraic connectivity — maximize λ₂)
    C3: ||W||₁ / N² (sparsity regularizer — prevent dense coupling)
    C4: ||W - W^T||_F / N (symmetry deviation)

    U_total = w1·C1 + w2·C2 + w3·C3 + w4·C4
    """
    w1, w2, w3, w4 = weights
    W = np.asarray(W, dtype=np.float64)
    n = W.shape[0]

    if _HAS_RUST:
        w_flat = np.ascontiguousarray(W.ravel())
        p = np.ascontiguousarray(phases, dtype=np.float64)
        c1, c2, c3, c4, ut = _rust_costs(w_flat, p, n, w1, w2, w3, w4)
        return SSGFCosts(
            c1_sync=c1,
            c2_spectral_gap=c2,
            c3_sparsity=c3,
            c4_symmetry=c4,
            u_total=ut,
        )

    R, _ = compute_order_parameter(phases)
    c1 = 1.0 - R

    lam2 = fiedler_value(W)
    c2 = -lam2  # minimize → maximize algebraic connectivity

    c3 = float(np.sum(np.abs(W))) / (n * n) if n > 0 else 0.0

    c4 = float(np.linalg.norm(W - W.T, "fro")) / n if n > 0 else 0.0

    u_total = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4

    return SSGFCosts(
        c1_sync=c1,
        c2_spectral_gap=c2,
        c3_sparsity=c3,
        c4_symmetry=c4,
        u_total=u_total,
    )
