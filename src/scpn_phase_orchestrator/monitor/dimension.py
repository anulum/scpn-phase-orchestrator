# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal dimension estimates

"""Fractal dimension estimation for phase-space trajectories with a
5-backend fallback chain per ``feedback_module_standard_attnres.md``.

Implements:

* :func:`correlation_integral` — Grassberger-Procaccia 1983 ``C(ε)``.
* :func:`correlation_dimension` — ``D2`` via log-log slope on ``C(ε)``.
* :func:`kaplan_yorke_dimension` — ``D_KY`` from a Lyapunov spectrum
  (Kaplan & Yorke 1979).

For parity across backends the RNG that picks subsampled pairs is
owned by the Python dispatcher and its seeded indices are passed to
every non-Rust backend. The Rust path keeps its own internal RNG for
backward compatibility; full-pairs mode is bit-exact across all five.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "CorrelationDimensionResult",
    "correlation_integral",
    "correlation_dimension",
    "kaplan_yorke_dimension",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        correlation_integral_rust,
        kaplan_yorke_dimension_rust,
    )

    return {
        "ci": correlation_integral_rust,
        "ky": kaplan_yorke_dimension_rust,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._dimension_mojo import (
        _ensure_exe,
        correlation_integral_mojo,
        kaplan_yorke_dimension_mojo,
    )

    _ensure_exe()
    return {"ci": correlation_integral_mojo, "ky": kaplan_yorke_dimension_mojo}


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._dimension_julia import (
        correlation_integral_julia,
        kaplan_yorke_dimension_julia,
    )

    return {
        "ci": correlation_integral_julia,
        "ky": kaplan_yorke_dimension_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._dimension_go import (
        _load_lib,
        correlation_integral_go,
        kaplan_yorke_dimension_go,
    )

    _load_lib()
    return {"ci": correlation_integral_go, "ky": kaplan_yorke_dimension_go}


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()[fn_name]


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


def _prepare_pair_indices(
    total_t: int,
    max_pairs: int,
    seed: int,
) -> tuple[NDArray, NDArray] | None:
    """Pre-select pair indices for non-Rust correlation-integral paths.

    Returns ``(idx_i, idx_j)`` — either the full upper-triangle or a
    seed-deterministic subsample — or ``None`` when the trajectory is
    too short for any pair to exist.
    """
    if total_t <= 1:
        return None
    total_pairs = total_t * (total_t - 1) // 2
    if total_pairs <= max_pairs:
        idx_i, idx_j = np.triu_indices(total_t, k=1)
        return (
            np.ascontiguousarray(idx_i, dtype=np.int64),
            np.ascontiguousarray(idx_j, dtype=np.int64),
        )
    rng = np.random.default_rng(seed)
    idx_i = rng.integers(0, total_t, max_pairs).astype(np.int64)
    idx_j = rng.integers(0, total_t, max_pairs).astype(np.int64)
    mask = idx_i != idx_j
    return idx_i[mask], idx_j[mask]


def correlation_integral(
    trajectory: NDArray,
    epsilons: NDArray,
    max_pairs: int = 50000,
    seed: int = 42,
) -> NDArray:
    """Correlation integral ``C(ε) = fraction of pairs within ε``.

    Grassberger-Procaccia 1983: ``C(ε) ∝ ε^{D₂}`` in the scaling
    region.

    Dispatches to the active backend. For ``T · (T−1)/2 ≤ max_pairs``
    all pairs are evaluated and every backend returns bit-exact
    agreement; when subsampling is needed the Python dispatcher owns
    the RNG and passes deterministic indices to every non-Rust
    backend, while the Rust path keeps its in-kernel RNG for API
    stability.

    Args:
        trajectory: ``(T, d)`` embedded trajectory.
        epsilons: ``(K,)`` distance thresholds.
        max_pairs: maximum number of pairs to evaluate.
        seed: RNG seed for pair subsampling.

    Returns:
        ``(K,)`` array of ``C(ε)`` values.
    """
    traj = np.atleast_2d(trajectory)
    t, d = int(traj.shape[0]), int(traj.shape[1])
    eps_sorted = np.ascontiguousarray(np.sort(epsilons), dtype=np.float64)

    backend_fn = _dispatch("ci")
    if backend_fn is not None and ACTIVE_BACKEND == "rust":
        fn_rust = cast(
            "Callable[[NDArray, int, int, NDArray, int, int], NDArray]",
            backend_fn,
        )
        return np.asarray(
            fn_rust(
                np.ascontiguousarray(traj.ravel(), dtype=np.float64),
                t,
                d,
                eps_sorted,
                int(max_pairs),
                int(seed),
            ),
            dtype=np.float64,
        )

    pair_result = _prepare_pair_indices(t, int(max_pairs), int(seed))
    if pair_result is None:
        return np.zeros(eps_sorted.size, dtype=np.float64)
    idx_i, idx_j = pair_result

    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray, int, int, NDArray, NDArray, NDArray], NDArray]",
            backend_fn,
        )
        return np.asarray(
            fn(
                np.ascontiguousarray(traj.ravel(), dtype=np.float64),
                t,
                d,
                idx_i,
                idx_j,
                eps_sorted,
            ),
            dtype=np.float64,
        )

    diffs = traj[idx_i] - traj[idx_j]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    n_pairs_actual = len(dists)
    if n_pairs_actual == 0:
        return np.zeros(eps_sorted.size, dtype=np.float64)
    return np.array([np.sum(dists < eps) / n_pairs_actual for eps in eps_sorted])


def correlation_dimension(
    trajectory: NDArray,
    n_epsilons: int = 30,
    max_pairs: int = 50000,
    seed: int = 42,
) -> CorrelationDimensionResult:
    """Estimate ``D₂`` via a log-log plateau over ``C(ε)``."""
    traj = np.atleast_2d(trajectory)
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

    valid = C_eps > 0
    if valid.sum() < 3:
        return CorrelationDimensionResult(
            D2=0.0,
            epsilons=epsilons,
            C_eps=C_eps,
            slope=np.zeros(len(epsilons) - 1),
            scaling_range=(float(epsilons[0]), float(epsilons[-1])),
        )

    log_eps = np.log(epsilons[valid])
    log_C = np.log(C_eps[valid])
    slopes = np.diff(log_C) / np.diff(log_eps)

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

    best_var = np.inf
    best_start = 0
    for i in range(len(slopes) - window + 1):
        v = np.var(slopes[i : i + window])
        if v < best_var:
            best_var = v
            best_start = i

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
    t = trajectory.shape[0]
    if t <= 1:
        return 0.0
    if t > 200:
        rng = np.random.default_rng(0)
        idx = rng.choice(t, 200, replace=False)
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
    """Kaplan-Yorke / information dimension from a Lyapunov spectrum.

    ``D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|`` where ``j`` is the
    largest index such that the cumulative sum of the first ``j``
    exponents is non-negative.

    Kaplan & Yorke 1979. The Kaplan-Yorke conjecture equates this to
    the information dimension ``D₁``.

    Args:
        lyapunov_exponents: ``(N,)`` Lyapunov exponents.

    Returns:
        ``D_KY``. Returns ``0.0`` if the largest exponent is negative
        (stable fixed point, zero-dimensional attractor).
    """
    le = np.asarray(lyapunov_exponents, dtype=np.float64)
    if le.size == 0:
        return 0.0
    backend_fn = _dispatch("ky")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray], float]", backend_fn)
        le_sorted = np.sort(le)[::-1]
        return float(fn(np.ascontiguousarray(le_sorted, dtype=np.float64)))

    le_sorted = np.sort(le)[::-1]
    cumsum = np.cumsum(le_sorted)

    if cumsum[0] < 0:
        return 0.0

    j = 0
    for i in range(len(cumsum)):
        if cumsum[i] >= 0:
            j = i
        else:
            break

    if j + 1 >= len(le_sorted):
        return float(len(le_sorted))

    denom = abs(le_sorted[j + 1])
    if denom == 0:
        return float(j + 1)

    return float(j + 1) + float(cumsum[j]) / float(denom)
