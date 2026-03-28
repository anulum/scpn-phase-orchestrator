# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anylum.li
# SCPN Phase Orchestrator — Normalized Persistent Entropy (NPE)

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["compute_npe", "phase_distance_matrix"]


def phase_distance_matrix(phases: NDArray) -> NDArray:
    """Pairwise circular distance matrix in [0, π]."""
    diff = phases[:, np.newaxis] - phases[np.newaxis, :]
    # Geodesic distance on S¹ via atan2 (handles wraparound correctly)
    dist: NDArray = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    return dist


def compute_npe(phases: NDArray, max_radius: float | None = None) -> float:
    """Normalized Persistent Entropy from H0 persistence diagram.

    NPE = -Σ p_i log(p_i) / log(n-1) where p_i = lifetime_i / Σ lifetimes.

    More sensitive than R for synchronization detection.
    NPE ≈ 1 → uniform (incoherent), NPE ≈ 0 → one dominant component (synchronized).

    Scientific Reports 2025 — NPE outperforms Kuramoto R for sync detection.

    Uses single-linkage clustering on circular distance matrix as a
    lightweight substitute for full Vietoris-Rips persistence (ripser).
    """
    n = len(phases)
    if n < 2:
        return 0.0

    dist = phase_distance_matrix(phases)

    # H0 persistence via MST: each edge merges two components, so
    # birth=0, death=edge_weight for the dying component.
    # The n-1 MST edge weights ARE the H0 barcode lifetimes.
    if max_radius is None:
        max_radius = np.pi

    # Upper triangle distances, sorted
    triu_idx = np.triu_indices(n, k=1)
    edges = dist[triu_idx]
    sorted_idx = np.argsort(edges)

    # Union-Find for Kruskal
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        # Union-Find with path compression (halving variant)
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
            # Kruskal step: merge two components; H0 bar dies at d
            lifetimes.append(d)
            # Union by rank keeps tree balanced → near O(α(n)) find
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

    # p_i = lifetime_i / Σ lifetimes (persistence probability)
    probs = np.array(lifetimes) / total
    probs = probs[probs > 0]
    # Shannon entropy of the persistence diagram
    entropy = -float(np.sum(probs * np.log(probs)))
    # Normalise by log(n-1) so NPE ∈ [0,1]; max when all bars equal
    max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0

    if max_entropy < 1e-15:  # pragma: no cover — log(n) > 0 for n >= 2
        return 0.0

    return entropy / max_entropy
