# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal dimension estimates

"""Fractal dimension estimation with a 5-backend fallback chain.

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
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

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


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._dimension_mojo import (
        _ensure_exe,
        correlation_integral_mojo,
        kaplan_yorke_dimension_mojo,
    )

    _ensure_exe()
    return {"ci": correlation_integral_mojo, "ky": kaplan_yorke_dimension_mojo}


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._dimension_julia import (
        correlation_integral_julia,
        kaplan_yorke_dimension_julia,
    )

    return {
        "ci": correlation_integral_julia,
        "ky": kaplan_yorke_dimension_julia,
    }


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._dimension_go import (
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
_BACKEND_FN_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    cached = _BACKEND_FN_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_FN_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object | None:
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    deduped: list[str] = []
    for backend in ordered_backends:
        if backend in deduped:
            continue
        deduped.append(backend)
    for backend in deduped:
        if backend == "python":
            return None
        try:
            backend_cache = _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        fn = backend_cache.get(fn_name)
        if fn is None:
            continue
        return fn
    return None


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


def _has_complex_payload(value: object) -> bool:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return _contains_complex_alias(value)
    return bool(np.iscomplexobj(raw) or _contains_complex_alias(value))


def _validate_trajectory(trajectory: object) -> FloatArray:
    raw = np.asarray(trajectory)
    if _contains_boolean_alias(trajectory):
        raise ValueError("trajectory must not contain boolean values")
    if _has_complex_payload(trajectory):
        raise ValueError("trajectory must contain real-valued phase-space samples")
    try:
        traj = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory must be a finite 1D or 2D float array") from exc
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]
    elif traj.ndim != 2:
        raise ValueError(f"trajectory must be 1D or 2D, got shape {traj.shape}")
    if not np.all(np.isfinite(traj)):
        raise ValueError("trajectory must contain only finite values")
    return np.ascontiguousarray(traj, dtype=np.float64)


def _validate_epsilons(epsilons: object) -> FloatArray:
    raw = np.asarray(epsilons)
    if _contains_boolean_alias(epsilons):
        raise ValueError("epsilons must not contain boolean values")
    if _has_complex_payload(epsilons):
        raise ValueError("epsilons must contain real-valued distance thresholds")
    try:
        eps = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilons must be a finite one-dimensional array") from exc
    if eps.ndim != 1:
        raise ValueError(f"epsilons must be one-dimensional, got shape {eps.shape}")
    if not np.all(np.isfinite(eps)) or np.any(eps < 0.0):
        raise ValueError("epsilons must contain only finite non-negative values")
    return np.ascontiguousarray(np.sort(eps), dtype=np.float64)


def _validate_dimension_result_epsilons(epsilons: object) -> FloatArray:
    eps = _validate_epsilons(epsilons)
    if np.any(eps <= 0.0):
        raise ValueError("epsilons must be positive for log-log scaling")
    if eps.size > 1 and np.any(np.diff(eps) <= 0.0):
        raise ValueError("epsilons must be strictly increasing for log-log scaling")
    return eps


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_spectrum(lyapunov_exponents: object) -> FloatArray:
    raw = np.asarray(lyapunov_exponents)
    if _contains_boolean_alias(lyapunov_exponents):
        raise ValueError("lyapunov_exponents must not contain boolean values")
    if _has_complex_payload(lyapunov_exponents):
        raise ValueError("lyapunov_exponents must contain real-valued exponents")
    try:
        le = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("lyapunov_exponents must be a finite 1D float array") from exc
    if le.ndim != 1:
        raise ValueError(
            f"lyapunov_exponents must be one-dimensional, got shape {le.shape}"
        )
    if not np.all(np.isfinite(le)):
        raise ValueError("lyapunov_exponents must contain only finite values")
    return np.ascontiguousarray(le, dtype=np.float64)


def _validate_non_negative_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def _validate_ci_values(value: object, *, expected_size: int) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("correlation integral output must not contain boolean values")
    raw = np.asarray(value)
    if _has_complex_payload(value):
        raise ValueError("correlation integral output must contain real values")
    try:
        ci = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("correlation integral output must be numeric") from exc
    if ci.shape != (expected_size,):
        raise ValueError(
            f"correlation integral output shape {ci.shape} does not match "
            f"({expected_size},)"
        )
    if not np.all(np.isfinite(ci)):
        raise ValueError("correlation integral output must contain only finite values")
    if np.any((ci < -1e-12) | (ci > 1.0 + 1e-12)):
        raise ValueError("correlation integral output must lie in [0, 1]")
    if ci.size > 1 and np.any(np.diff(ci) < -1e-12):
        raise ValueError("correlation integral output must be monotonic in epsilon")
    return np.ascontiguousarray(np.clip(ci, 0.0, 1.0), dtype=np.float64)


def _validate_ci_exact_reference(
    value: object,
    *,
    expected: FloatArray,
    atol: float,
) -> FloatArray:
    result = _validate_ci_values(value, expected_size=int(expected.size))
    if not np.allclose(result, expected, rtol=0.0, atol=atol):
        raise ValueError(
            "correlation integral backend output diverged from exact reference"
        )
    return result


def _validate_ky_dimension(
    value: object,
    *,
    n_exponents: int,
    expected: float | None = None,
    atol: float = 1e-12,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("kaplan_yorke_dimension must not be a boolean value")
    if _has_complex_payload(value):
        raise ValueError("kaplan_yorke_dimension must be real-valued")
    dimension = _validate_non_negative_float(value, name="kaplan_yorke_dimension")
    if dimension > n_exponents + 1e-12:
        raise ValueError("kaplan_yorke_dimension must not exceed spectrum length")
    clipped = min(dimension, float(n_exponents))
    if expected is not None and abs(clipped - expected) > atol:
        raise ValueError(
            "kaplan_yorke_dimension backend output diverged from exact reference"
        )
    return clipped


@dataclass
class CorrelationDimensionResult:
    """Result of correlation dimension estimation.

    Attributes
    ----------
        D2: Estimated correlation dimension.
        epsilons: (K,) array of distance thresholds used.
        C_eps: (K,) correlation integral values C(ε).
        slope: (K-1,) local log-log slopes.
        scaling_range: (ε_lo, ε_hi) range where power law holds.
    """

    D2: float
    epsilons: FloatArray
    C_eps: FloatArray
    slope: FloatArray
    scaling_range: tuple[float, float]

    def __post_init__(self) -> None:
        d2 = _validate_non_negative_float(self.D2, name="D2")
        epsilons = _validate_dimension_result_epsilons(self.epsilons)
        try:
            c_eps = _validate_ci_values(self.C_eps, expected_size=int(epsilons.size))
        except ValueError as exc:
            raise ValueError(f"C_eps {exc}") from exc
        if _contains_boolean_alias(self.slope):
            raise ValueError("slope must not contain boolean values")
        if _has_complex_payload(self.slope):
            raise ValueError("slope must contain real values")
        try:
            slope = np.asarray(self.slope, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("slope must be a finite one-dimensional array") from exc
        if slope.ndim != 1:
            raise ValueError("slope must be one-dimensional")
        if not np.all(np.isfinite(slope)):
            raise ValueError("slope must contain only finite values")
        if slope.size not in {max(int(epsilons.size) - 1, 0), 1}:
            raise ValueError("slope length must match epsilon intervals")
        if not isinstance(self.scaling_range, tuple) or len(self.scaling_range) != 2:
            raise ValueError("scaling_range must contain two finite values")
        lo = _validate_non_negative_float(self.scaling_range[0], name="scaling_range")
        hi = _validate_non_negative_float(self.scaling_range[1], name="scaling_range")
        if hi < lo:
            raise ValueError("scaling_range upper bound must be >= lower bound")

        self.D2 = d2
        self.epsilons = epsilons
        self.C_eps = c_eps
        self.slope = np.ascontiguousarray(slope, dtype=np.float64)
        self.scaling_range = (lo, hi)


def _prepare_pair_indices(
    total_t: int,
    max_pairs: int,
    seed: int,
) -> tuple[IntArray, IntArray] | None:
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


def _correlation_integral_exact_reference(
    trajectory: FloatArray,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    if idx_i.size == 0:
        return np.zeros(epsilons.size, dtype=np.float64)
    diffs = trajectory[idx_i] - trajectory[idx_j]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return _validate_ci_values(
        np.array([np.sum(dists < eps) / dists.size for eps in epsilons]),
        expected_size=int(epsilons.size),
    )


def _kaplan_yorke_exact_reference(lyapunov_exponents: FloatArray) -> float:
    if lyapunov_exponents.size == 0:
        return 0.0
    le_sorted = np.sort(lyapunov_exponents)[::-1]
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


def correlation_integral(
    trajectory: object,
    epsilons: object,
    max_pairs: object = 50000,
    seed: object = 42,
) -> FloatArray:
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

    Returns
    -------
        ``(K,)`` array of ``C(ε)`` values.

    Parameters
    ----------
    trajectory : object
        Phase-space trajectory, shape ``(T, d)``.
    epsilons : object
        Radii at which to evaluate the correlation integral.
    max_pairs : object
        Maximum number of point pairs sampled, or ``None`` for all.
    seed : object
        Seed for the deterministic RNG.
    """
    traj = _validate_trajectory(trajectory)
    t, d = int(traj.shape[0]), int(traj.shape[1])
    eps_sorted = _validate_epsilons(epsilons)
    max_pairs = _validate_int_at_least(max_pairs, name="max_pairs", minimum=1)
    seed = _validate_int_at_least(seed, name="seed", minimum=0)
    pair_result = _prepare_pair_indices(t, max_pairs, seed)
    if pair_result is None:
        return np.zeros(eps_sorted.size, dtype=np.float64)
    idx_i, idx_j = pair_result
    expected = _correlation_integral_exact_reference(traj, idx_i, idx_j, eps_sorted)
    full_pairs = idx_i.size == t * (t - 1) // 2

    backend_fn = _dispatch("ci")
    if backend_fn is not None and ACTIVE_BACKEND == "rust":
        fn_rust = cast(
            "Callable[[FloatArray, int, int, FloatArray, int, int], FloatArray]",
            backend_fn,
        )
        try:
            rust_output = fn_rust(
                np.ascontiguousarray(traj.ravel(), dtype=np.float64),
                t,
                d,
                eps_sorted,
                max_pairs,
                seed,
            )
            if full_pairs:
                return _validate_ci_exact_reference(
                    rust_output,
                    expected=expected,
                    atol=1e-12,
                )
            return _validate_ci_values(
                rust_output,
                expected_size=int(eps_sorted.size),
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int, IntArray, IntArray, FloatArray], "
            "FloatArray]",
            backend_fn,
        )
        try:
            return _validate_ci_exact_reference(
                fn(
                    np.ascontiguousarray(traj.ravel(), dtype=np.float64),
                    t,
                    d,
                    idx_i,
                    idx_j,
                    eps_sorted,
                ),
                expected=expected,
                atol=1e-9 if ACTIVE_BACKEND == "mojo" else 1e-12,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    return expected


def correlation_dimension(
    trajectory: object,
    n_epsilons: object = 30,
    max_pairs: object = 50000,
    seed: object = 42,
) -> CorrelationDimensionResult:
    """Estimate ``D₂`` via a log-log plateau over ``C(ε)``.

    Parameters
    ----------
    trajectory : object
        Phase-space trajectory, shape ``(T, d)``.
    n_epsilons : object
        Number of radii sampled across the log-log range.
    max_pairs : object
        Maximum number of point pairs sampled, or ``None`` for all.
    seed : object
        Seed for the deterministic RNG.

    Returns
    -------
    CorrelationDimensionResult
        The estimated correlation dimension ``D₂`` result.
    """
    traj = _validate_trajectory(trajectory)
    n_epsilons = _validate_int_at_least(
        n_epsilons,
        name="n_epsilons",
        minimum=2,
    )
    max_pairs = _validate_int_at_least(max_pairs, name="max_pairs", minimum=1)
    seed = _validate_int_at_least(seed, name="seed", minimum=0)
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

    valid_indices = np.flatnonzero(valid)
    first_valid = int(valid_indices[0])
    log_eps = np.log(epsilons[valid])
    log_C = np.log(C_eps[valid])
    slopes = np.diff(log_C) / np.diff(log_eps)
    full_slopes = np.zeros(len(epsilons) - 1, dtype=np.float64)
    full_slopes[first_valid : first_valid + len(slopes)] = slopes

    window = min(5, len(slopes))
    if window < 2:
        D2 = max(0.0, float(slopes[0])) if len(slopes) > 0 else 0.0
        return CorrelationDimensionResult(
            D2=D2,
            epsilons=epsilons,
            C_eps=C_eps,
            slope=full_slopes,
            scaling_range=(float(epsilons[0]), float(epsilons[-1])),
        )

    best_var = float(np.inf)
    best_start = 0
    for i in range(len(slopes) - window + 1):
        v = float(np.var(slopes[i : i + window]))
        if v < best_var:
            best_var = v
            best_start = i

    D2 = max(0.0, float(np.mean(slopes[best_start : best_start + window])))
    eps_valid = epsilons[valid]
    scaling_lo = float(eps_valid[best_start])
    scaling_hi = float(eps_valid[min(best_start + window, len(eps_valid) - 1)])

    return CorrelationDimensionResult(
        D2=D2,
        epsilons=epsilons,
        C_eps=C_eps,
        slope=full_slopes,
        scaling_range=(scaling_lo, scaling_hi),
    )


def _attractor_diameter(trajectory: FloatArray) -> float:
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


def kaplan_yorke_dimension(lyapunov_exponents: FloatArray) -> float:
    """Kaplan-Yorke / information dimension from a Lyapunov spectrum.

    ``D_KY = j + (Σ_{i=1}^{j} λ_i) / |λ_{j+1}|`` where ``j`` is the
    largest index such that the cumulative sum of the first ``j``
    exponents is non-negative.

    Kaplan & Yorke 1979. The Kaplan-Yorke conjecture equates this to
    the information dimension ``D₁``.

    Args:
        lyapunov_exponents: ``(N,)`` Lyapunov exponents.

    Returns
    -------
        ``D_KY``. Returns ``0.0`` if the largest exponent is negative
        (stable fixed point, zero-dimensional attractor).

    Parameters
    ----------
    lyapunov_exponents : FloatArray
        The ordered Lyapunov spectrum.
    """
    le = _validate_spectrum(lyapunov_exponents)
    if le.size == 0:
        return 0.0
    expected = _kaplan_yorke_exact_reference(le)
    backend_fn = _dispatch("ky")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray], float]", backend_fn)
        le_sorted = np.sort(le)[::-1]
        try:
            return _validate_ky_dimension(
                fn(np.ascontiguousarray(le_sorted, dtype=np.float64)),
                n_exponents=int(le_sorted.size),
                expected=expected,
                atol=1e-9 if ACTIVE_BACKEND == "mojo" else 1e-12,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    return expected
