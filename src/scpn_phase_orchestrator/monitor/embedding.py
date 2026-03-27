# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding and attractor reconstruction

"""Time-delay embedding for attractor reconstruction from scalar time series.

Implements Takens' embedding theorem: a scalar observable of a deterministic
system can be embedded in m dimensions using time-delayed copies to recover
the attractor topology (up to diffeomorphism).

References:
    Takens 1981, "Detecting strange attractors in turbulence",
        Lecture Notes in Mathematics 898:366-381.
    Fraser & Swinney 1986, Phys. Rev. A 33:1134-1140.
        (optimal delay via mutual information)
    Kennel, Brown & Abarbanel 1992, Phys. Rev. A 45:3403-3411.
        (false nearest neighbors for embedding dimension)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "EmbeddingResult",
    "delay_embed",
    "optimal_delay",
    "optimal_dimension",
]


@dataclass
class EmbeddingResult:
    """Result of delay embedding procedure.

    Attributes:
        trajectory: (T', m) embedded trajectory.
        delay: time delay τ used.
        dimension: embedding dimension m used.
        T_effective: number of embedded points (T - (m-1)*τ).
    """

    trajectory: NDArray
    delay: int
    dimension: int
    T_effective: int


def delay_embed(
    signal: NDArray,
    delay: int,
    dimension: int,
) -> NDArray:
    """Construct time-delay embedding matrix.

    Given scalar signal x(t), constructs vectors:
        v(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]

    Args:
        signal: (T,) scalar time series.
        delay: Time delay τ in samples.
        dimension: Embedding dimension m.

    Returns:
        (T', m) embedded trajectory where T' = T - (m-1)*τ.
    """
    s = np.asarray(signal).ravel()
    T = len(s)
    T_eff = T - (dimension - 1) * delay
    if T_eff <= 0:
        msg = (
            f"Signal too short (T={T}) for delay={delay}, "
            f"dimension={dimension}: need T > {(dimension - 1) * delay}"
        )
        raise ValueError(msg)

    indices = np.arange(dimension) * delay
    rows = np.arange(T_eff)[:, np.newaxis] + indices[np.newaxis, :]
    result: NDArray[np.floating] = s[rows]
    return result


def _mutual_information(signal: NDArray, lag: int, n_bins: int = 32) -> float:
    """Average mutual information between x(t) and x(t+lag).

    Uses histogram-based estimation (Fraser & Swinney 1986).
    """
    s = np.asarray(signal).ravel()
    T = len(s) - lag
    if T <= 0:
        return 0.0

    x = s[:T]
    y = s[lag : lag + T]

    # Joint histogram
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)

    # Normalize
    p_xy = hist_xy / hist_xy.sum()
    p_x = hist_x / hist_x.sum()
    p_y = hist_y / hist_y.sum()

    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

    return float(mi)


def optimal_delay(
    signal: NDArray,
    max_lag: int = 100,
    n_bins: int = 32,
) -> int:
    """Find optimal delay τ as first minimum of mutual information.

    Fraser & Swinney 1986: the first local minimum of the average
    mutual information I(τ) provides a good embedding delay — it
    balances independence (large τ) against loss of dynamical
    correlation (too large τ).

    Args:
        signal: (T,) scalar time series.
        max_lag: Maximum lag to search.
        n_bins: Histogram bins for MI estimation.

    Returns:
        Optimal delay τ (samples). Returns 1 if no minimum found.
    """
    s = np.asarray(signal).ravel()
    max_lag = min(max_lag, len(s) // 2)

    mi_values = np.array(
        [_mutual_information(s, lag, n_bins) for lag in range(max_lag)]
    )

    # First local minimum
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            return i

    return 1


def _nearest_neighbor_distances(embedded: NDArray) -> tuple[NDArray, NDArray]:
    """Find nearest neighbor for each point and return (distances, indices)."""
    T, m = embedded.shape
    nn_dist = np.full(T, np.inf)
    nn_idx = np.zeros(T, dtype=int)

    for i in range(T):
        diffs = embedded - embedded[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf  # exclude self
        j = np.argmin(dists)
        nn_dist[i] = dists[j]
        nn_idx[i] = j

    return nn_dist, nn_idx


def optimal_dimension(
    signal: NDArray,
    delay: int,
    max_dim: int = 10,
    rtol: float = 15.0,
    atol: float = 2.0,
) -> int:
    """Find optimal embedding dimension via False Nearest Neighbors.

    Kennel, Brown & Abarbanel 1992: a false nearest neighbor is one
    whose distance increases drastically when the embedding dimension
    is increased by 1. When FNN fraction drops below threshold, the
    attractor is unfolded.

    Args:
        signal: (T,) scalar time series.
        delay: Time delay τ.
        max_dim: Maximum dimension to test.
        rtol: Relative distance threshold for FNN criterion 1.
        atol: Absolute distance threshold (fraction of attractor size)
            for FNN criterion 2.

    Returns:
        Optimal embedding dimension m.
    """
    s = np.asarray(signal).ravel()
    sigma = np.std(s)
    if sigma == 0:
        return 1

    for m in range(1, max_dim + 1):
        T_eff = len(s) - m * delay
        if T_eff <= 1:
            return m

        emb_m = delay_embed(s, delay, m)
        T_m = emb_m.shape[0]

        T_next = len(s) - m * delay
        if T_next <= 1:
            return m

        nn_dist, nn_idx = _nearest_neighbor_distances(emb_m)

        n_false = 0
        n_valid = 0

        for i in range(T_m):
            j = int(nn_idx[i])
            d = nn_dist[i]
            if d == 0 or d == np.inf:
                continue

            # Check if the (m+1)-th coordinate changes the distance
            i_next = i + m * delay
            j_next = j + m * delay
            if i_next >= len(s) or j_next >= len(s):
                continue

            n_valid += 1
            extra_dist = abs(s[i_next] - s[j_next])

            # Criterion 1: relative increase
            if extra_dist / d > rtol:
                n_false += 1
                continue

            # Criterion 2: absolute size
            new_dist = np.sqrt(d**2 + extra_dist**2)
            if new_dist / sigma > atol:
                n_false += 1

        fnn_frac = n_false / n_valid if n_valid > 0 else 0.0
        if fnn_frac < 0.01:
            return m

    return max_dim


def auto_embed(
    signal: NDArray,
    max_lag: int = 100,
    max_dim: int = 10,
) -> EmbeddingResult:
    """Automatically determine delay and dimension, then embed.

    Combines optimal_delay (MI first minimum) and optimal_dimension
    (FNN) to produce the embedding.

    Args:
        signal: (T,) scalar time series.
        max_lag: Maximum lag for MI search.
        max_dim: Maximum dimension for FNN search.

    Returns:
        EmbeddingResult with embedded trajectory and parameters used.
    """
    tau = optimal_delay(signal, max_lag)
    m = optimal_dimension(signal, tau, max_dim)
    traj = delay_embed(signal, tau, m)
    return EmbeddingResult(
        trajectory=traj,
        delay=tau,
        dimension=m,
        T_effective=traj.shape[0],
    )
