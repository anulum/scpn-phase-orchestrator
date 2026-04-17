# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Normalized Persistent Entropy (NPE)

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = ["compute_npe", "phase_distance_matrix"]


def phase_distance_matrix(phases: NDArray) -> NDArray:
    """Pairwise circular distance matrix in [0, pi]."""
    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import phase_distance_matrix as _rust_pdm

        flat = _rust_pdm(np.ascontiguousarray(phases.ravel()))
        n = len(phases)
        return np.asarray(flat).reshape(n, n)

    diff = phases[:, np.newaxis] - phases[np.newaxis, :]
    dist: NDArray = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    return dist


def compute_npe(phases: NDArray, max_radius: float | None = None) -> float:
    """Normalized Persistent Entropy from H0 persistence diagram.

    NPE = -Sigma p_i log(p_i) / log(n-1) where p_i = lifetime_i / Sigma lifetimes.

    More sensitive than R for synchronization detection.
    NPE ~ 1 = uniform (incoherent), NPE ~ 0 = one dominant component (synchronized).

    Scientific Reports 2025 — NPE outperforms Kuramoto R for sync detection.

    Uses single-linkage clustering on circular distance matrix as a
    lightweight substitute for full Vietoris-Rips persistence (ripser).
    """
    n = len(phases)
    if n < 2:
        return 0.0

    if max_radius is None:
        max_radius = np.pi

    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import compute_npe as _rust_npe

        return float(_rust_npe(np.ascontiguousarray(phases.ravel()), max_radius))

    dist = phase_distance_matrix(phases)

    triu_idx = np.triu_indices(n, k=1)
    edges = dist[triu_idx]
    sorted_idx = np.argsort(edges)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lifetimes: list[float] = []
    for idx in sorted_idx:
        i, j = int(triu_idx[0][idx]), int(triu_idx[1][idx])
        d = float(edges[idx])
        if d > max_radius:
            break
        ri, rj = find(i), find(j)
        if ri != rj:
            lifetimes.append(d)
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1

    if not lifetimes:
        return 0.0

    total = sum(lifetimes)
    if total < 1e-15:
        return 0.0

    probs = np.array(lifetimes) / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0

    if max_entropy < 1e-15:
        return 0.0

    return entropy / max_entropy
