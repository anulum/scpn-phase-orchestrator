# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition of coupling dynamics

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        hodge_decomposition_rust as _rust_hodge,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["HodgeResult", "hodge_decomposition"]


@dataclass
class HodgeResult:
    """Hodge decomposition of coupling flow into three orthogonal parts.

    gradient: phase-locking component (conservative, derives from a potential)
    curl:     circulation component (antisymmetric coupling flows)
    harmonic: topological residual (neither gradient nor curl)
    """

    gradient: NDArray
    curl: NDArray
    harmonic: NDArray


def hodge_decomposition(knm: NDArray, phases: NDArray) -> HodgeResult:
    """Decompose coupling dynamics into gradient, curl, and harmonic parts.

    For each oscillator i:
      gradient_i = Σ_j K_ij^sym cos(θ_j - θ_i)   [symmetric part]
      curl_i     = Σ_j K_ij^anti cos(θ_j - θ_i)   [antisymmetric part]
      harmonic   = total - gradient - curl           [topological residual]

    The symmetric part of K drives gradient (phase-locking) flow;
    the antisymmetric part drives rotational (curl) flow.

    Jiang et al. 2011, Math. Program. 127(1):203-244.
    """
    phases = np.asarray(phases, dtype=np.float64)
    n = len(phases)
    if n == 0:
        empty = np.array([], dtype=np.float64)
        return HodgeResult(gradient=empty, curl=empty, harmonic=empty)

    if _HAS_RUST:
        knm = np.asarray(knm, dtype=np.float64)
        k_flat = np.ascontiguousarray(knm.ravel())
        p_flat = np.ascontiguousarray(phases)
        g, c, h = _rust_hodge(k_flat, p_flat, n)
        return HodgeResult(
            gradient=np.asarray(g),
            curl=np.asarray(c),
            harmonic=np.asarray(h),
        )

    diff = phases[np.newaxis, :] - phases[:, np.newaxis]  # θ_j - θ_i
    cos_diff = np.cos(diff)

    k_sym = 0.5 * (knm + knm.T)
    k_anti = 0.5 * (knm - knm.T)

    # Total coupling force on each oscillator
    total = np.sum(knm * cos_diff, axis=1)

    gradient = np.sum(k_sym * cos_diff, axis=1)
    curl = np.sum(k_anti * cos_diff, axis=1)

    # Harmonic = whatever the sym/anti split doesn't capture
    # For exact matrix decomposition this is zero, but numerical residuals remain
    harmonic = total - gradient - curl

    return HodgeResult(
        gradient=gradient.astype(np.float64),
        curl=curl.astype(np.float64),
        harmonic=harmonic.astype(np.float64),
    )
