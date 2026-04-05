# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral graph analysis for coupling networks

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (  # type: ignore[import-untyped]
        critical_coupling_rust as _rust_kc,
    )
    from spo_kernel import (
        fiedler_value_rust as _rust_fv,
    )
    from spo_kernel import (
        fiedler_vector_rust as _rust_fvec,
    )
    from spo_kernel import (
        spectral_gap_rust as _rust_sg,
    )
    from spo_kernel import (
        sync_convergence_rate_rust as _rust_scr,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = [
    "graph_laplacian",
    "fiedler_value",
    "fiedler_vector",
    "critical_coupling",
    "fiedler_partition",
    "spectral_gap",
    "sync_convergence_rate",
]


def graph_laplacian(knm: NDArray) -> NDArray:
    """Combinatorial graph Laplacian L = D - W where D = diag(row sums)."""
    w = np.abs(knm)
    np.fill_diagonal(w, 0.0)
    degrees = w.sum(axis=1)
    L: NDArray = np.diag(degrees) - w
    return L


def fiedler_value(knm: NDArray) -> float:
    """Algebraic connectivity λ₂(L) — second smallest eigenvalue of L.

    Dörfler & Bullo 2014, Automatica 50(6):1539–1564.
    """
    if _HAS_RUST:
        n = knm.shape[0]
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        return float(_rust_fv(flat, n))

    L = graph_laplacian(knm)
    eigs = np.linalg.eigvalsh(L)
    # λ₁ ≈ 0 (connected graph), λ₂ is algebraic connectivity
    return float(eigs[1]) if len(eigs) > 1 else 0.0


def fiedler_vector(knm: NDArray) -> NDArray:
    """Eigenvector corresponding to λ₂(L) — partitions network into sync clusters."""
    if _HAS_RUST:
        n = knm.shape[0]
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        return np.asarray(_rust_fvec(flat, n))

    L = graph_laplacian(knm)
    _, vecs = np.linalg.eigh(L)
    return vecs[:, 1]


def critical_coupling(omegas: NDArray, knm: NDArray) -> float:
    """Dörfler-Bullo critical coupling: K_c = max|ω_i - ω_j| / λ₂(L).

    Synchronization requires K > K_c. Returns inf if graph is disconnected (λ₂ = 0).

    Dörfler & Bullo 2013, IEEE Proc. 102(10):1539–1564.
    """
    if _HAS_RUST:
        n = knm.shape[0]
        return float(
            _rust_kc(
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                n,
            )
        )

    lambda2 = fiedler_value(knm)
    if lambda2 < 1e-12:
        return float("inf")
    omega_spread = float(np.max(omegas) - np.min(omegas))
    return omega_spread / lambda2


def fiedler_partition(knm: NDArray) -> tuple[list[int], list[int]]:
    """Bisect network using Fiedler vector sign.

    Returns (group_positive, group_negative) — indices of oscillators
    in each partition. Oscillators with v₂ > 0 tend to synchronize together.
    """
    v2 = fiedler_vector(knm)
    pos = [i for i, val in enumerate(v2) if val >= 0]
    neg = [i for i, val in enumerate(v2) if val < 0]
    return pos, neg


def spectral_gap(knm: NDArray) -> float:
    """Gap between λ₂ and λ₃ — larger gap means cleaner two-cluster structure."""
    if _HAS_RUST:
        n = knm.shape[0]
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        return float(_rust_sg(flat, n))

    L = graph_laplacian(knm)
    eigs = np.linalg.eigvalsh(L)
    if len(eigs) < 3:
        return 0.0
    return float(eigs[2] - eigs[1])


def sync_convergence_rate(
    knm: NDArray, omegas: NDArray, gamma_max: float = 0.0
) -> float:
    """Estimated convergence rate μ = K_eff · λ₂ · cos(γ_max) / N.

    gamma_max: maximum phase difference between connected pairs (radians).
    Dörfler & Bullo 2014, §III.B.
    """
    n = len(omegas)
    if n == 0:
        return 0.0
    if _HAS_RUST:
        return float(
            _rust_scr(
                np.ascontiguousarray(knm.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                n,
                gamma_max,
            )
        )

    lambda2 = fiedler_value(knm)
    k_eff = float(np.mean(knm[knm > 0])) if np.any(knm > 0) else 0.0
    return float(k_eff * lambda2 * np.cos(gamma_max) / n)
