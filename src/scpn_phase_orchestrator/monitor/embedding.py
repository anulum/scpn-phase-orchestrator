# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding primitives + wrappers

"""Delay-embedding analysis with a 5-backend fallback chain.

Three compute primitives on the multi-language chain:

* :func:`delay_embed` — time-delay embedding matrix.
* :func:`mutual_information` — Fraser-Swinney 1986 average mutual
  information.
* :func:`nearest_neighbor_distances` — brute-force ``k=1`` kNN in
  the embedded space (consumed by FNN).

Two wrappers stay Python-side (they are control flow over the
primitives):

* :func:`optimal_delay` — first local minimum of MI (Fraser-Swinney).
* :func:`optimal_dimension` — Kennel-Brown-Abarbanel 1992 FNN.
* :func:`auto_embed` — convenience that chains ``optimal_delay``,
  ``optimal_dimension``, and :func:`delay_embed`.

The Rust backend exposes native ``optimal_delay_rust`` and
``optimal_dimension_rust`` entry points; when Rust is active those
wrappers use the native path for maximum throughput. The Python
fallback composes the primitives through the dispatcher.

MI and NN are exposed by Julia / Go / Mojo / Python only — Rust
does not expose standalone MI or kNN FFI; those slots dispatch to
the next available backend in the chain.
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
    "EmbeddingResult",
    "auto_embed",
    "delay_embed",
    "mutual_information",
    "nearest_neighbor_distances",
    "optimal_delay",
    "optimal_dimension",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        delay_embed_rust,
        optimal_delay_rust,
        optimal_dimension_rust,
    )

    def _de(signal: FloatArray, delay: int, dimension: int) -> FloatArray:
        flat = np.ascontiguousarray(signal, dtype=np.float64)
        return np.asarray(
            delay_embed_rust(flat, int(delay), int(dimension)),
            dtype=np.float64,
        )

    return {
        "de": _de,
        "mi": None,  # Rust has no standalone MI kernel.
        "nn": None,  # Rust has no standalone kNN kernel.
        "optimal_delay": optimal_delay_rust,
        "optimal_dimension": optimal_dimension_rust,
    }


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._embedding_mojo import (
        _ensure_exe,
        delay_embed_mojo,
        mutual_information_mojo,
        nearest_neighbor_distances_mojo,
    )

    _ensure_exe()
    return {
        "de": delay_embed_mojo,
        "mi": mutual_information_mojo,
        "nn": nearest_neighbor_distances_mojo,
    }


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._embedding_julia import (
        delay_embed_julia,
        mutual_information_julia,
        nearest_neighbor_distances_julia,
    )

    return {
        "de": delay_embed_julia,
        "mi": mutual_information_julia,
        "nn": nearest_neighbor_distances_julia,
    }


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._embedding_go import (
        _load_lib,
        delay_embed_go,
        mutual_information_go,
        nearest_neighbor_distances_go,
    )

    _load_lib()
    return {
        "de": delay_embed_go,
        "mi": mutual_information_go,
        "nn": nearest_neighbor_distances_go,
    }


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
}
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    _BACKEND_CACHE.clear()
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
    """Return the backend function or ``None`` to signal Python fallback.

    If the active backend lacks ``fn_name`` (e.g. Rust has no standalone
    MI kernel), we fall through to the next available backend in the
    chain rather than crashing or silently diverging.
    """
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for name in ordered_backends:
        if name in seen:
            continue
        seen.add(name)
        if name == "python":
            return None
        try:
            fn = _load_backend(name)[fn_name]
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        if fn is not None:
            return fn
    return None


@dataclass
class EmbeddingResult:
    """Delay-embedding output."""

    trajectory: FloatArray
    delay: int
    dimension: int
    T_effective: int

    def __post_init__(self) -> None:
        trajectory = _validate_embedded(self.trajectory)
        delay = _validate_int_at_least(self.delay, name="delay", minimum=1)
        dimension = _validate_int_at_least(self.dimension, name="dimension", minimum=1)
        t_effective = _validate_int_at_least(
            self.T_effective,
            name="T_effective",
            minimum=0,
        )
        if trajectory.shape != (t_effective, dimension):
            raise ValueError(
                f"trajectory shape {trajectory.shape} does not match "
                f"(T_effective={t_effective}, dimension={dimension})"
            )
        self.trajectory = trajectory
        self.delay = delay
        self.dimension = dimension
        self.T_effective = t_effective


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _validate_signal(signal: object, *, name: str = "signal") -> FloatArray:
    if _contains_boolean_alias(signal):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(signal)
    if _contains_complex_alias(raw):
        raise ValueError(f"{name} must contain real-valued samples")
    try:
        array = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite one-dimensional float array"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_embedded(embedded: object) -> FloatArray:
    if _contains_boolean_alias(embedded):
        raise ValueError("embedded must not contain boolean values")
    raw = np.asarray(embedded)
    if _contains_complex_alias(raw):
        raise ValueError("embedded must contain real-valued coordinates")
    try:
        array = np.atleast_2d(raw.astype(np.float64, copy=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "embedded must be a finite two-dimensional float array"
        ) from exc
    if array.ndim != 2:
        raise ValueError(f"embedded must be two-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("embedded must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_non_negative_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real, got {value!r}")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return result


def _validate_delay_embedding_output(
    value: object,
    *,
    signal: FloatArray,
    delay: int,
    t_effective: int,
    dimension: int,
) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("delay embedding output must not contain boolean values")
    raw = np.asarray(value)
    if _contains_complex_alias(raw):
        raise ValueError("delay embedding output must contain real values")
    try:
        embedded = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("delay embedding output must be numeric") from exc
    if embedded.shape == (t_effective * dimension,):
        embedded = embedded.reshape(t_effective, dimension)
    if embedded.shape != (t_effective, dimension):
        raise ValueError(
            f"delay embedding output shape {embedded.shape} does not match "
            f"({t_effective}, {dimension})"
        )
    if not np.all(np.isfinite(embedded)):
        raise ValueError("delay embedding output must contain only finite values")
    indices = np.arange(dimension, dtype=np.int64) * int(delay)
    rows = (
        np.arange(t_effective, dtype=np.int64)[:, np.newaxis] + indices[np.newaxis, :]
    )
    expected = signal[rows]
    if not np.array_equal(embedded, expected):
        raise ValueError("delay embedding output must match exact indexing")
    return np.ascontiguousarray(embedded, dtype=np.float64)


def _validate_non_negative_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or _contains_boolean_alias(value):
        raise ValueError(f"{name} must not be a boolean value")
    raw = np.asarray(value)
    if _contains_complex_alias(raw):
        raise ValueError(f"{name} must contain real values")
    try:
        scalar = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if scalar.shape != ():
        raise ValueError(f"{name} must be scalar")
    result = float(scalar)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_nn_output(
    distances: object,
    indices: object,
    *,
    n_points: int,
) -> tuple[FloatArray, IntArray]:
    if _contains_boolean_alias(distances):
        raise ValueError("nearest-neighbor distances must not contain boolean values")
    if _contains_boolean_alias(indices):
        raise ValueError("nearest-neighbor indices must not contain boolean values")
    raw_dist = np.asarray(distances)
    raw_idx = np.asarray(indices)
    if _contains_complex_alias(raw_dist):
        raise ValueError("nearest-neighbor distances must contain real values")
    if _contains_complex_alias(raw_idx):
        raise ValueError("nearest-neighbor indices must be integer values")
    try:
        dist = raw_dist.astype(np.float64, copy=True)
        idx_float = raw_idx.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("nearest-neighbor backend output must be numeric") from exc
    if dist.shape != (n_points,) or idx_float.shape != (n_points,):
        raise ValueError(
            "nearest-neighbor backend output shape must match number of points"
        )
    if not np.all(np.isfinite(dist)) or np.any(dist < 0.0):
        raise ValueError("nearest-neighbor distances must be finite and non-negative")
    if not np.all(np.isfinite(idx_float)):
        raise ValueError("nearest-neighbor indices must be finite")
    if not np.all(np.equal(idx_float, np.floor(idx_float))):
        raise ValueError("nearest-neighbor indices must be integral")
    idx = idx_float.astype(np.int64, copy=False)
    if np.any(idx < 0) or np.any(idx >= n_points):
        raise ValueError("nearest-neighbor indices must be in range")
    if n_points > 1 and np.any(idx == np.arange(n_points, dtype=np.int64)):
        raise ValueError("nearest-neighbor indices must not point to self")
    return (
        np.ascontiguousarray(dist, dtype=np.float64),
        np.ascontiguousarray(idx, dtype=np.int64),
    )


def delay_embed(
    signal: object,
    delay: object,
    dimension: object,
) -> FloatArray:
    """Time-delay embedding: ``v(t) = [x(t), x(t+τ), x(t+2τ), …]``.

    Parameters
    ----------
    signal : object
        Real-valued time series, shape ``(T,)``.
    delay : object
        Embedding delay ``τ`` in samples.
    dimension : object
        Embedding dimension.

    Returns
    -------
    FloatArray
        The time-delay embedding, shape ``(M, dimension)``.

    Raises
    ------
    ValueError
        If ``delay`` or ``dimension`` is non-positive or too large for the signal.
    """
    s = _validate_signal(signal)
    delay = _validate_int_at_least(delay, name="delay", minimum=1)
    dimension = _validate_int_at_least(dimension, name="dimension", minimum=1)
    t_eff = int(s.size) - (dimension - 1) * delay
    if t_eff <= 0:
        msg = (
            f"Signal too short (T={s.size}) for delay={delay}, "
            f"dimension={dimension}: need T > {(dimension - 1) * delay}"
        )
        raise ValueError(msg)

    backend_fn = _dispatch("de")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int], FloatArray]", backend_fn)
        try:
            return _validate_delay_embedding_output(
                fn(s, delay, dimension),
                signal=s,
                delay=delay,
                t_effective=t_eff,
                dimension=dimension,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    indices = np.arange(dimension) * delay
    rows = np.arange(t_eff)[:, np.newaxis] + indices[np.newaxis, :]
    trajectory: FloatArray = np.asarray(s[rows], dtype=np.float64)
    return _validate_delay_embedding_output(
        trajectory,
        signal=s,
        delay=delay,
        t_effective=t_eff,
        dimension=dimension,
    )


def mutual_information(
    signal: object,
    lag: object,
    n_bins: object = 32,
) -> float:
    """Fraser-Swinney 1986 average mutual information at ``lag``.

    Parameters
    ----------
    signal : object
        Real-valued time series, shape ``(T,)``.
    lag : object
        Lag in samples.
    n_bins : object
        Number of histogram bins.

    Returns
    -------
    float
        The average mutual information at the given lag.
    """
    s = _validate_signal(signal)
    lag = _validate_int_at_least(lag, name="lag", minimum=0)
    n_bins = _validate_int_at_least(n_bins, name="n_bins", minimum=2)
    if s.size - lag <= 0:
        return 0.0

    backend_fn = _dispatch("mi")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int], float]", backend_fn)
        try:
            return _validate_non_negative_scalar(
                fn(s, lag, n_bins),
                name="mutual_information",
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    t_total = s.size - lag
    x = s[:t_total]
    y = s[lag : lag + t_total]
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)
    total = hist_xy.sum()
    if total <= 0:
        return 0.0
    p_xy = hist_xy / total
    p_x = hist_x / total
    p_y = hist_y / total
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))
    return _validate_non_negative_scalar(mi, name="mutual_information")


def nearest_neighbor_distances(
    embedded: object,
) -> tuple[FloatArray, IntArray]:
    """Brute-force ``k = 1`` kNN on the rows of ``embedded``.

    Parameters
    ----------
    embedded : object
        Delay-embedded trajectory, shape ``(M, dimension)``.

    Returns
    -------
    tuple[FloatArray, IntArray]
        The nearest-neighbour distances and their indices.
    """
    e = _validate_embedded(embedded)
    t, m = int(e.shape[0]), int(e.shape[1])
    if t == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)

    backend_fn = _dispatch("nn")
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int], tuple[FloatArray, IntArray]]",
            backend_fn,
        )
        try:
            dist, idx = fn(np.ascontiguousarray(e.ravel(), dtype=np.float64), t, m)
            return _validate_nn_output(dist, idx, n_points=t)
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    nn_dist = np.full(t, np.inf)
    nn_idx = np.zeros(t, dtype=np.int64)
    for i in range(t):
        diffs = e - e[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        dists[i] = np.inf
        j = int(np.argmin(dists))
        nn_dist[i] = dists[j]
        nn_idx[i] = j
    return _validate_nn_output(nn_dist, nn_idx, n_points=t)


def optimal_delay(
    signal: object,
    max_lag: object = 100,
    n_bins: object = 32,
) -> int:
    """First local minimum of :func:`mutual_information` vs ``lag``.

    Parameters
    ----------
    signal : object
        Real-valued time series, shape ``(T,)``.
    max_lag : object
        Largest lag to search.
    n_bins : object
        Number of histogram bins.

    Returns
    -------
    int
        The first mutual-information minimum, as a lag in samples.
    """
    s = _validate_signal(signal)
    max_lag = _validate_int_at_least(max_lag, name="max_lag", minimum=1)
    n_bins = _validate_int_at_least(n_bins, name="n_bins", minimum=2)

    if ACTIVE_BACKEND == "rust":
        try:
            fn = cast(
                "Callable[[FloatArray, int, int], int]",
                _load_backend("rust")["optimal_delay"],
            )
            return int(fn(s, max_lag, n_bins))
        except (ImportError, RuntimeError, OSError, KeyError):
            max_lag = int(max_lag)

    max_lag = min(max_lag, s.size // 2)
    mi_values = np.array([mutual_information(s, lag, n_bins) for lag in range(max_lag)])
    for i in range(1, len(mi_values) - 1):
        if mi_values[i] < mi_values[i - 1] and mi_values[i] < mi_values[i + 1]:
            return i
    return 1


def optimal_dimension(
    signal: object,
    delay: object,
    max_dim: object = 10,
    rtol: object = 15.0,
    atol: object = 2.0,
) -> int:
    """Kennel-Brown-Abarbanel 1992 FNN to select embedding dimension.

    Parameters
    ----------
    signal : object
        Real-valued time series, shape ``(T,)``.
    delay : object
        Embedding delay ``τ`` in samples.
    max_dim : object
        Largest embedding dimension to test.
    rtol : object
        Relative tolerance for the false-nearest-neighbour test.
    atol : object
        Absolute tolerance for the false-nearest-neighbour test.

    Returns
    -------
    int
        The selected embedding dimension.
    """
    s = _validate_signal(signal)
    delay = _validate_int_at_least(delay, name="delay", minimum=1)
    max_dim = _validate_int_at_least(max_dim, name="max_dim", minimum=1)
    rtol = _validate_non_negative_real(rtol, name="rtol")
    atol = _validate_non_negative_real(atol, name="atol")
    sigma = float(np.std(s))
    if sigma == 0:
        return 1

    if ACTIVE_BACKEND == "rust":
        try:
            fn = cast(
                "Callable[..., int]",
                _load_backend("rust")["optimal_dimension"],
            )
            return int(
                fn(s, delay, max_dim, rtol, atol),
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            max_dim = int(max_dim)

    for m in range(1, max_dim + 1):
        t_next = s.size - m * delay
        if t_next <= 1:
            return m
        emb_m = delay_embed(s, delay, m)
        t_m = emb_m.shape[0]
        nn_dist, nn_idx = nearest_neighbor_distances(emb_m)

        n_false = 0
        n_valid = 0
        for i in range(t_m):
            j = int(nn_idx[i])
            d = nn_dist[i]
            if d == 0 or not np.isfinite(d):
                continue
            i_next = i + m * delay
            j_next = j + m * delay
            if i_next >= s.size or j_next >= s.size:
                continue
            n_valid += 1
            extra = abs(s[i_next] - s[j_next])
            if extra / d > rtol:
                n_false += 1
                continue
            new_dist = (d * d + extra * extra) ** 0.5
            if new_dist / sigma > atol:
                n_false += 1
        fnn_frac = n_false / n_valid if n_valid > 0 else 0.0
        if fnn_frac < 0.01:
            return m
    return max_dim


def auto_embed(
    signal: object,
    max_lag: object = 100,
    max_dim: object = 10,
) -> EmbeddingResult:
    """``optimal_delay`` ∘ ``optimal_dimension`` ∘ ``delay_embed``.

    Parameters
    ----------
    signal : object
        Real-valued time series, shape ``(T,)``.
    max_lag : object
        Largest lag to search.
    max_dim : object
        Largest embedding dimension to test.

    Returns
    -------
    EmbeddingResult
        The auto-selected delay/dimension embedding result.
    """
    tau = optimal_delay(signal, max_lag)
    m = optimal_dimension(signal, tau, max_dim)
    traj = delay_embed(signal, tau, m)
    return EmbeddingResult(
        trajectory=traj,
        delay=tau,
        dimension=m,
        T_effective=int(traj.shape[0]),
    )
