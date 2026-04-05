# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability analysis

"""Basin stability for Kuramoto synchronization.

Monte Carlo estimation of the volume of the basin of attraction for
the synchronized state. Basin stability S_B is the probability that
a random initial condition converges to the synchronized attractor.

References:
    Menck et al. 2013, Nature Physics 9:89-92.
    Ji et al. 2014, Sci. Reports 4:4783.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (  # type: ignore[import-untyped]
        basin_stability_rust as _rust_basin,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = [
    "BasinStabilityResult",
    "basin_stability",
    "multi_basin_stability",
]


@dataclass
class BasinStabilityResult:
    """Basin stability estimation result.

    Attributes:
        S_B: Basin stability (fraction of ICs converging to sync).
        n_samples: Total number of initial conditions tested.
        n_converged: Number that converged to synchronized state.
        R_final: (n_samples,) final order parameter for each trial.
        R_threshold: Threshold used for sync classification.
    """

    S_B: float
    n_samples: int
    n_converged: int
    R_final: NDArray
    R_threshold: float


def _run_kuramoto_to_steady(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Integrate Kuramoto and return time-averaged R."""
    phases = phases_init.copy()

    for _ in range(n_transient):
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        phases += dt * (omegas + coupling)

    R_sum = 0.0
    for _ in range(n_measure):
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        phases += dt * (omegas + coupling)
        z = np.mean(np.exp(1j * phases))
        R_sum += float(np.abs(z))

    return R_sum / n_measure


def basin_stability(
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray | None = None,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    n_samples: int = 100,
    R_threshold: float = 0.8,
    seed: int = 42,
) -> BasinStabilityResult:
    """Estimate basin stability of the synchronized state.

    Draws n_samples random initial phase configurations from [0, 2π)^N,
    integrates each to steady state, and checks if R_final > R_threshold.

    Args:
        omegas: (N,) natural frequencies.
        knm: (N, N) coupling matrix.
        alpha: (N, N) phase lags (default: zeros).
        dt: Integration timestep.
        n_transient: Transient steps to discard.
        n_measure: Steps to average R over.
        n_samples: Number of random initial conditions.
        R_threshold: Threshold for classifying as "synchronized".
        seed: RNG seed.

    Returns:
        BasinStabilityResult with S_B, R_final array, and counts.
    """
    N = len(omegas)
    if alpha is None:
        alpha = np.zeros((N, N))

    if _HAS_RUST:
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        s_b, r_finals_arr, n_conv = _rust_basin(
            o, k, a, N, dt, n_transient, n_measure,
            n_samples, R_threshold, seed,
        )
        return BasinStabilityResult(
            S_B=float(s_b),
            n_samples=n_samples,
            n_converged=int(n_conv),
            R_final=np.asarray(r_finals_arr),
            R_threshold=R_threshold,
        )

    rng = np.random.default_rng(seed)
    R_finals = np.zeros(n_samples)

    for i in range(n_samples):
        phases_init = rng.uniform(0, 2 * np.pi, N)
        R_finals[i] = _run_kuramoto_to_steady(
            phases_init, omegas, knm, alpha, dt, n_transient, n_measure,
        )

    n_converged = int(np.sum(R_finals >= R_threshold))

    return BasinStabilityResult(
        S_B=n_converged / n_samples,
        n_samples=n_samples,
        n_converged=n_converged,
        R_final=R_finals,
        R_threshold=R_threshold,
    )


def multi_basin_stability(
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray | None = None,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    n_samples: int = 100,
    R_thresholds: tuple[float, ...] = (0.3, 0.6, 0.8),
    seed: int = 42,
) -> dict[str, BasinStabilityResult]:
    """Basin stability at multiple synchronization thresholds.

    Classifies outcomes into basins:
        - "desynchronized": R < R_thresholds[0]
        - "partial": R_thresholds[0] <= R < R_thresholds[-1]
        - "synchronized": R >= R_thresholds[-1]

    Useful for detecting multi-stability (chimera states, partial sync).

    Returns:
        Dict mapping threshold labels to BasinStabilityResult.
    """
    N = len(omegas)
    if alpha is None:
        alpha = np.zeros((N, N))

    if _HAS_RUST:
        # Run once at lowest threshold to get R_finals, then threshold locally
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        _, r_finals_arr, _ = _rust_basin(
            o, k, a, N, dt, n_transient, n_measure,
            n_samples, 0.0, seed,
        )
        R_finals = np.asarray(r_finals_arr)
        results = {}
        for thresh in R_thresholds:
            n_above = int(np.sum(R_finals >= thresh))
            results[f"R>={thresh:.2f}"] = BasinStabilityResult(
                S_B=n_above / n_samples,
                n_samples=n_samples,
                n_converged=n_above,
                R_final=R_finals,
                R_threshold=thresh,
            )
        return results

    rng = np.random.default_rng(seed)
    R_finals = np.zeros(n_samples)

    for i in range(n_samples):
        phases_init = rng.uniform(0, 2 * np.pi, N)
        R_finals[i] = _run_kuramoto_to_steady(
            phases_init, omegas, knm, alpha, dt, n_transient, n_measure,
        )

    results = {}
    for thresh in R_thresholds:
        n_above = int(np.sum(R_finals >= thresh))
        results[f"R>={thresh:.2f}"] = BasinStabilityResult(
            S_B=n_above / n_samples,
            n_samples=n_samples,
            n_converged=n_above,
            R_final=R_finals,
            R_threshold=thresh,
        )

    return results
