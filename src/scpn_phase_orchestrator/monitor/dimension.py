# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal dimension estimates

"""Fractal dimension estimation for phase-space trajectories.

Implements:
    - Correlation dimension D2 via Grassberger-Procaccia algorithm
    - Kaplan-Yorke dimension D_KY from Lyapunov spectrum

References:
    Grassberger & Procaccia 1983, Phys. Rev. Lett. 50:346-349.
    Kaplan & Yorke 1979, Lecture Notes in Mathematics 730:228-237.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        correlation_integral_rust as _rust_ci,
    )
    from spo_kernel import (
        kaplan_yorke_dimension_rust as _rust_ky,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = [
    "CorrelationDimensionResult",
    "correlation_integral",
    "correlation_dimension",
    "kaplan_yorke_dimension",
]


@dataclass
class CorrelationDimensionResult:
    """Result of correlation dimension estimation.

    Attributes:
        D2: Estimated correlation dimension.
        epsilons: (K,) array of distance thresholds used.
        C_eps: (K,) correlation integral values C(ε).
        slope: (K-1,) local log-log slopes.
        scaling_range: (ε_lo, ε_hi) range where power law holds.
    """

    D2: float
    epsilons: NDArray
    C_eps: NDArray
    slope: NDArray
    scaling_range: tuple[float, float]


def correlation_integral(
    trajectory: NDArray,
    epsilons: NDArray,
    max_pairs: int = 50000,
    seed: int = 42,
) -> NDArray:
    """Compute correlation integral C(ε) = fraction of pairs within distance ε.

    Grassberger-Procaccia 1983: C(ε) ~ ε^D2 in the scaling region.

    For large T, subsamples pairs randomly to keep computation tractable.

    Args:
        trajectory: (T, d) embedded trajectory.
        epsilons: (K,) array of distance thresholds.
        max_pairs: Maximum number of pairs to evaluate.
        seed: RNG seed for pair subsampling.

    Returns:
        (K,) array of C(ε) values.
    """
    traj = np.atleast_2d(trajectory)
    T = traj.shape[0]

    if _HAS_RUST:
        d = traj.shape[1]
        flat = np.ascontiguousarray(traj.ravel(), dtype=np.float64)
        eps_arr = np.ascontiguousarray(np.sort(epsilons), dtype=np.float64)
        return np.asarray(_rust_ci(flat, T, d, eps_arr, max_pairs, seed))

    total_pairs = T * (T - 1) // 2
    epsilons = np.sort(epsilons)

    if total_pairs <= max_pairs:
        # All T(T-1)/2 unique pairs; Heaviside counting below
        idx_i, idx_j = np.triu_indices(T, k=1)
        diffs = traj[idx_i] - traj[idx_j]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
    else:
        # Subsample pairs
        rng = np.random.default_rng(seed)
        n_pairs = max_pairs
        idx_i = rng.integers(0, T, n_pairs)
        idx_j = rng.integers(0, T, n_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        diffs = traj[idx_i] - traj[idx_j]
        dists = np.sqrt(np.sum(diffs**2, axis=1))

    # C(ε) = (2 / N(N-1)) Σ Θ(ε - ||x_i - x_j||)
    # Grassberger & Procaccia 1983, Eq. 1
    n_pairs_actual = len(dists)
    return np.array([np.sum(dists < eps) / n_pairs_actual for eps in epsilons])


def correlation_dimension(
    trajectory: NDArray,
    n_epsilons: int = 30,
    max_pairs: int = 50000,
    seed: int = 42,
) -> CorrelationDimensionResult:
    """Estimate correlation dimension D2 from embedded trajectory.

    Computes C(ε) for logarithmically spaced ε, then estimates D2
    as the slope of log C(ε) vs log ε in the scaling region.

    The scaling region is identified as the range where local slopes
    are most stable (lowest variance over a sliding window).

    Args:
        trajectory: (T, d) embedded trajectory.
        n_epsilons: Number of ε values to sample.
        max_pairs: Max pairs for correlation integral.
        seed: RNG seed.

    Returns:
        CorrelationDimensionResult with D2, C(ε), slopes, and scaling range.
    """
    traj = np.atleast_2d(trajectory)

    # Distance range: from 1% to 100% of attractor diameter
    diam = _attractor_diameter(traj)
    if diam == 0:
        return CorrelationDimensionResult(
            D2=0.0,
            epsilons=np.array([1.0]),
            C_eps=np.array([1.0]),
            slope=np.array([0.0]),
            scaling_range=(1.0, 1.0),
        )

    epsilons = np.logspace(np.log10(diam * 0.01), np.log10(diam), n_epsilons)

    C_eps = correlation_integral(traj, epsilons, max_pairs, seed)

    # Local slopes: d log C / d log ε
    valid = C_eps > 0
    if valid.sum() < 3:
        return CorrelationDimensionResult(
            D2=0.0,
            epsilons=epsilons,
            C_eps=C_eps,
            slope=np.zeros(len(epsilons) - 1),
            scaling_range=(float(epsilons[0]), float(epsilons[-1])),
        )

    # D2 = lim_{ε→0} d log C(ε) / d log ε (GP83 power-law scaling)
    log_eps = np.log(epsilons[valid])
    log_C = np.log(C_eps[valid])
    slopes = np.diff(log_C) / np.diff(log_eps)

    # Find most stable region (sliding window of 5)
    window = min(5, len(slopes))
    if window < 2:
        D2 = float(slopes[0]) if len(slopes) > 0 else 0.0
        return CorrelationDimensionResult(
            D2=D2,
            epsilons=epsilons,
            C_eps=C_eps,
            slope=slopes,
            scaling_range=(float(epsilons[0]), float(epsilons[-1])),
        )

    # Scaling region: window with lowest slope variance (plateau)
    best_var = np.inf
    best_start = 0
    for i in range(len(slopes) - window + 1):
        v = np.var(slopes[i : i + window])
        if v < best_var:
            best_var = v
            best_start = i

    # D2 is the mean slope in the identified scaling region
    D2 = float(np.mean(slopes[best_start : best_start + window]))
    eps_valid = epsilons[valid]
    scaling_lo = float(eps_valid[best_start])
    scaling_hi = float(eps_valid[min(best_start + window, len(eps_valid) - 1)])

    return CorrelationDimensionResult(
        D2=D2,
        epsilons=epsilons,
        C_eps=C_eps,
        slope=slopes,
        scaling_range=(scaling_lo, scaling_hi),
    )


def _attractor_diameter(trajectory: NDArray) -> float:
    """Estimate attractor diameter as max distance between sampled points."""
    T = trajectory.shape[0]
    if T <= 1:
        return 0.0
    # Sample up to 200 points for speed
    if T > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(T, 200, replace=False)
        sample = trajectory[idx]
    else:
        sample = trajectory
    maxd = 0.0
    for i in range(len(sample)):
        dists = np.sqrt(np.sum((sample[i] - sample) ** 2, axis=1))
        d = float(np.max(dists))
        if d > maxd:
            maxd = d
    return maxd


def kaplan_yorke_dimension(lyapunov_exponents: NDArray) -> float:
    """Kaplan-Yorke dimension from Lyapunov spectrum.

    D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|

    where j is the largest index such that the cumulative sum of
    the first j exponents is non-negative.

    Kaplan & Yorke 1979. The Kaplan-Yorke conjecture equates this
    to the information dimension D1.

    Args:
        lyapunov_exponents: (N,) Lyapunov exponents, sorted descending.

    Returns:
        D_KY. Returns 0.0 if the largest exponent is negative (stable
        fixed point, zero-dimensional attractor).
    """
    if _HAS_RUST:
        le_sorted = np.sort(lyapunov_exponents)[::-1]
        return float(_rust_ky(np.ascontiguousarray(le_sorted, dtype=np.float64)))

    # Sort descending: λ_1 ≥ λ_2 ≥ ... ≥ λ_N
    le = np.sort(lyapunov_exponents)[::-1]
    cumsum = np.cumsum(le)

    # All exponents negative → stable fixed point → D_KY = 0
    if cumsum[0] < 0:
        return 0.0

    # Find j: largest index where Σ_{i=1}^{j} λ_i ≥ 0
    j = 0
    for i in range(len(cumsum)):
        if cumsum[i] >= 0:
            j = i
        else:
            break

    # All exponents non-negative → volume-expanding in all directions
    if j + 1 >= len(le):
        return float(len(le))

    denom = abs(le[j + 1])
    if denom == 0:
        return float(j + 1)

    # D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|
    # Kaplan & Yorke 1979; interpolates between integer dimensions
    return float(j + 1) + float(cumsum[j]) / float(denom)
