# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delay embedding primitives + wrappers

"""Delay-embedding analysis with a 5-backend fallback chain.

Original measurement and parameter types exclude temporal, Boolean and text
aliases before conversion; numeric object samples remain supported.

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
``optimal_dimension_rust`` entry points; when Rust is the selected
owner those wrappers use the native path. Other owners compose the
primitives through the dispatcher.

MI and NN are exposed by Julia / Go / Mojo / Python only — Rust
does not expose standalone MI or kNN FFI. In automatic selection
those slots dispatch to the next available backend in the chain; an
explicitly requested ``backend="rust"`` refuses them.

Every public function accepts ``backend``: ``"auto"`` (the fallback
chain) or one owner name. An explicit owner must be available and
must succeed; it never falls back silently. All owners share one
numerical law: mutual information bins each window over its own range
with NumPy ``histogram2d`` edges, and nearest neighbours take the first
minimum of the sequentially accumulated squared distance. Native
mutual-information and nearest-neighbour results are checked against
that reference before they are returned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator.monitor._julia_runtime import require_juliacall_main

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "EmbeddingResult",
    "auto_embed",
    "delay_embed",
    "embedding_backend",
    "mutual_information",
    "nearest_neighbor_distances",
    "optimal_delay",
    "optimal_dimension",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    """Load the Rust delay-embedding backend callables."""
    module = import_module("spo_kernel")
    delay_embed_rust: object = getattr(module, "delay_embed_rust", None)
    optimal_delay_rust: object = getattr(module, "optimal_delay_rust", None)
    optimal_dimension_rust: object = getattr(module, "optimal_dimension_rust", None)
    if not all(
        callable(fn)
        for fn in (delay_embed_rust, optimal_delay_rust, optimal_dimension_rust)
    ):
        raise ImportError("spo_kernel lacks callable delay-embedding exports")
    native_delay_embed = cast(
        "Callable[[FloatArray, int, int], object]", delay_embed_rust
    )

    def _de(signal: FloatArray, delay: int, dimension: int) -> FloatArray:
        """Call the Rust delay-embedding kernel with a contiguous signal."""
        flat = np.ascontiguousarray(signal, dtype=np.float64)
        return np.asarray(
            native_delay_embed(flat, int(delay), int(dimension)),
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
    """Load the Mojo delay-embedding backend callables."""
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
    """Load the Julia delay-embedding backend callables."""
    require_juliacall_main()
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
    """Load the Go delay-embedding backend callables."""
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
    """Load and cache the named backend callables."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
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


def _dispatch(fn_name: str, backend: str = "auto") -> tuple[str, object | None]:
    """Resolve a callable together with its owner; require explicit owners.

    In automatic selection a backend that lacks ``fn_name`` (Rust has no
    standalone MI or kNN kernel) falls through to the next available owner.
    An explicitly requested owner must provide the callable.

    Parameters
    ----------
    fn_name : str
        Backend slot: ``"de"``, ``"mi"``, ``"nn"``, ``"optimal_delay"`` or
        ``"optimal_dimension"``.
    backend : str
        ``"auto"`` or one of ``rust``, ``mojo``, ``julia``, ``go``, ``python``.

    Returns
    -------
    tuple[str, object | None]
        The owner name and its callable, or ``("python", None)`` for the
        Python reference.

    Raises
    ------
    ValueError
        If ``backend`` is not a known owner.
    RuntimeError
        If an explicitly requested owner is unavailable or lacks the slot.
    """
    _require_known_backend(backend)
    required = backend != "auto"
    ordered = [ACTIVE_BACKEND, *AVAILABLE_BACKENDS] if backend == "auto" else [backend]
    seen: set[str] = set()
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        if name == "python":
            return "python", None
        try:
            fn = _load_backend(name).get(fn_name)
        except (ImportError, RuntimeError, OSError, KeyError) as exc:
            if required:
                raise RuntimeError(f"embedding backend {name!r} unavailable") from exc
            continue
        if callable(fn):
            return name, fn
        if required:
            raise RuntimeError(f"embedding backend {name!r} lacks {fn_name}")
    return "python", None


_SLOTS = ("de", "mi", "nn", "optimal_delay", "optimal_dimension")


def embedding_backend(slot: str, backend: str = "auto") -> str:
    """Return the owner that serves one embedding operation.

    Parameters
    ----------
    slot : str
        ``"de"`` (delay embedding), ``"mi"`` (mutual information), ``"nn"``
        (nearest neighbours), ``"optimal_delay"`` or ``"optimal_dimension"``.
    backend : str
        ``"auto"`` or an explicit owner name.

    Returns
    -------
    str
        The owner name: ``rust``, ``mojo``, ``julia``, ``go`` or ``python``.
        For the two searches, ``python`` means the search composes the
        primitives of the requested owner.

    Raises
    ------
    ValueError
        If ``slot`` or ``backend`` is unknown.
    RuntimeError
        If an explicitly requested owner is unavailable or lacks the operation.
    """
    if slot not in _SLOTS:
        raise ValueError(f"unknown embedding operation: {slot!r}")
    if slot in ("optimal_delay", "optimal_dimension") and backend not in (
        "rust",
        "auto",
    ):
        _require_known_backend(backend)
        return "python"
    if slot in ("optimal_delay", "optimal_dimension") and (
        backend == "auto" and ACTIVE_BACKEND != "rust"
    ):
        return "python"
    owner, _ = _dispatch(slot, backend)
    return owner


@dataclass
class EmbeddingResult:
    """Delay-embedding output."""

    trajectory: FloatArray
    delay: int
    dimension: int
    T_effective: int

    def __post_init__(self) -> None:
        """Validate and normalise the embedded trajectory record."""
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
    """Return whether the value contains any boolean alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _is_string_like(value: object) -> bool:
    """Return whether ``value`` is a Python or NumPy string scalar."""
    return isinstance(value, (str, bytes, np.str_, np.bytes_))


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string scalar parsable as a float."""
    if not _is_string_like(value):
        return False
    try:
        float(cast("str | bytes", value))
    except (TypeError, ValueError):
        return False
    return True


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether the value contains numeric-string aliases."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    saw_string = False
    for item in raw.astype(object, copy=False).flat:
        if not _is_string_like(item):
            continue
        saw_string = True
        if not _is_numeric_string_alias(item):
            return False
    return saw_string


def _validate_signal(signal: object, *, name: str = "signal") -> FloatArray:
    """Return the signal as a validated 1-D finite array, else raise."""
    if _contains_boolean_alias(signal):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(signal)
    if _contains_complex_alias(raw):
        raise ValueError(f"{name} must contain real-valued samples")
    if _contains_numeric_string_alias(raw):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    try:
        require_real_values(signal, name=name, allow_object=True)
        array = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite one-dimensional float array"
        ) from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_embedded(embedded: object) -> FloatArray:
    """Return the embedded matrix as a validated 2-D finite array, else raise."""
    if _contains_boolean_alias(embedded):
        raise ValueError("embedded must not contain boolean values")
    raw = np.asarray(embedded)
    if _contains_complex_alias(raw):
        raise ValueError("embedded must contain real-valued coordinates")
    if _contains_numeric_string_alias(raw):
        raise ValueError("embedded must not contain numeric-string aliases")
    try:
        require_real_values(embedded, name="embedded", allow_object=True)
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
    """Return ``value`` as an integer at least the minimum, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    require_real_values(value, name=name)
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_non_negative_real(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite real, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real, got {value!r}")
    require_real_values(value, name=name)
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
    """Return the backend delay-embedding output matching the reference."""
    if _contains_boolean_alias(value):
        raise ValueError("delay embedding output must not contain boolean values")
    raw = np.asarray(value)
    if _contains_complex_alias(raw):
        raise ValueError("delay embedding output must contain real values")
    if _contains_numeric_string_alias(raw):
        raise ValueError(
            "delay embedding output must not contain numeric-string aliases"
        )
    try:
        require_real_values(value, name="delay embedding output", allow_object=True)
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
    """Return ``value`` as a non-negative finite scalar, else raise."""
    if isinstance(value, (bool, np.bool_)) or _contains_boolean_alias(value):
        raise ValueError(f"{name} must not be a boolean value")
    raw = np.asarray(value)
    if _contains_complex_alias(raw):
        raise ValueError(f"{name} must contain real values")
    if _contains_numeric_string_alias(raw):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    try:
        require_real_values(value, name=name, allow_object=True)
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
    """Return the validated nearest-neighbour backend output, else raise."""
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
    if _contains_numeric_string_alias(raw_dist):
        raise ValueError(
            "nearest-neighbor distances must not contain numeric-string aliases"
        )
    if _contains_numeric_string_alias(raw_idx):
        raise ValueError(
            "nearest-neighbor indices must not contain numeric-string aliases"
        )
    try:
        require_real_values(distances, name="distances", allow_object=True)
        require_real_values(indices, name="indices", allow_object=True)
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


_NATIVE_FAILURES = (ImportError, RuntimeError, OSError, KeyError)
_MI_RELATIVE_TOLERANCE = 1e-12


def _owner_failed(owner: str) -> RuntimeError:
    """Return the refusal raised when an explicitly requested owner fails."""
    return RuntimeError(f"embedding backend {owner!r} failed")


def _require_known_backend(backend: str) -> None:
    """Refuse an unknown owner name before any computation."""
    if backend not in (*_BACKEND_NAMES, "auto"):
        raise ValueError(f"unknown embedding backend: {backend!r}")


def delay_embed(
    signal: object,
    delay: object,
    dimension: object,
    backend: str = "auto",
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
    backend : str
        ``"auto"`` or an explicit owner (``rust``, ``mojo``, ``julia``,
        ``go``, ``python``).

    Returns
    -------
    FloatArray
        The time-delay embedding, shape ``(M, dimension)``.

    Raises
    ------
    ValueError
        If ``delay`` or ``dimension`` is non-positive or too large for the
        signal, or a backend returns a wrong embedding.
    RuntimeError
        If an explicitly requested backend is unavailable or fails.
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

    owner, backend_fn = _dispatch("de", backend)
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
        except _NATIVE_FAILURES as exc:
            if backend != "auto":
                raise _owner_failed(owner) from exc

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


def _mutual_information_reference(s: FloatArray, lag: int, n_bins: int) -> float:
    """Return the NumPy ``histogram2d`` mutual-information reference."""
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


def _validate_mi_against_reference(value: float, expected: float) -> float:
    """Return a native MI value only when it agrees with the reference."""
    if abs(value - expected) > _MI_RELATIVE_TOLERANCE * max(1.0, abs(expected)):
        raise ValueError("mutual_information backend output diverged from reference")
    return value


def mutual_information(
    signal: object,
    lag: object,
    n_bins: object = 32,
    backend: str = "auto",
) -> float:
    """Fraser-Swinney 1986 average mutual information at ``lag``.

    Each window is binned over its own range with NumPy ``histogram2d``
    edges; a value on an interior edge opens the next bin and the last edge
    closes the last bin.

    Parameters
    ----------
    signal : object
        Real-valued time series, shape ``(T,)``.
    lag : object
        Lag in samples.
    n_bins : object
        Number of histogram bins.
    backend : str
        ``"auto"`` or an explicit owner. Rust has no MI kernel, so
        ``backend="rust"`` is refused.

    Returns
    -------
    float
        The average mutual information at the given lag.

    Raises
    ------
    ValueError
        If an input is invalid or a backend result diverges from the reference.
    RuntimeError
        If an explicitly requested backend is unavailable, lacks MI or fails.
    """
    s = _validate_signal(signal)
    lag = _validate_int_at_least(lag, name="lag", minimum=0)
    n_bins = _validate_int_at_least(n_bins, name="n_bins", minimum=2)
    owner, backend_fn = _dispatch("mi", backend)
    if s.size - lag <= 0:
        return 0.0

    expected = _mutual_information_reference(s, lag, n_bins)
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int], float]", backend_fn)
        try:
            value = _validate_non_negative_scalar(
                fn(s, lag, n_bins),
                name="mutual_information",
            )
        except _NATIVE_FAILURES as exc:
            if backend != "auto":
                raise _owner_failed(owner) from exc
        else:
            return _validate_mi_against_reference(value, expected)
    return expected


def _nearest_neighbor_reference(e: FloatArray) -> tuple[FloatArray, IntArray]:
    """Return the first minimum of the sequentially summed squared distance."""
    t = int(e.shape[0])
    nn_dist = np.full(t, np.inf)
    nn_idx = np.zeros(t, dtype=np.int64)
    for i in range(t):
        squared = np.zeros(t, dtype=np.float64)
        for k in range(int(e.shape[1])):
            difference = e[:, k] - e[i, k]
            squared += difference * difference
        squared[i] = np.inf
        j = int(np.argmin(squared))
        nn_dist[i] = np.sqrt(squared[j])
        nn_idx[i] = j
    return _validate_nn_output(nn_dist, nn_idx, n_points=t)


def nearest_neighbor_distances(
    embedded: object,
    backend: str = "auto",
) -> tuple[FloatArray, IntArray]:
    """Brute-force ``k = 1`` kNN on the rows of ``embedded``.

    The neighbour of a row is the first minimum of the squared distance
    accumulated over coordinates in order; the reported distance is its
    square root.

    Parameters
    ----------
    embedded : object
        Delay-embedded trajectory, shape ``(M, dimension)``.
    backend : str
        ``"auto"`` or an explicit owner. Rust has no kNN kernel, so
        ``backend="rust"`` is refused.

    Returns
    -------
    tuple[FloatArray, IntArray]
        The nearest-neighbour distances and their indices.

    Raises
    ------
    ValueError
        If the input is invalid or a backend result differs from the reference.
    RuntimeError
        If an explicitly requested backend is unavailable, lacks kNN or fails.
    """
    e = _validate_embedded(embedded)
    t, m = int(e.shape[0]), int(e.shape[1])
    owner, backend_fn = _dispatch("nn", backend)
    if t == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)

    expected_dist, expected_idx = _nearest_neighbor_reference(e)
    if backend_fn is not None:
        fn = cast(
            "Callable[[FloatArray, int, int], tuple[FloatArray, IntArray]]",
            backend_fn,
        )
        try:
            dist, idx = _validate_nn_output(
                *fn(np.ascontiguousarray(e.ravel(), dtype=np.float64), t, m),
                n_points=t,
            )
        except _NATIVE_FAILURES as exc:
            if backend != "auto":
                raise _owner_failed(owner) from exc
        else:
            return _validate_nn_against_reference(dist, idx, e, expected_dist)
    return expected_dist, expected_idx


def _validate_nn_against_reference(
    dist: FloatArray,
    idx: IntArray,
    e: FloatArray,
    expected_dist: FloatArray,
) -> tuple[FloatArray, IntArray]:
    """Return native neighbours that are true nearest neighbours of each row.

    A native compiler may contract the squared-distance sum into fused
    multiply-adds, so distances can differ from the reference in the last
    bits. Each returned distance must equal the reference distance to its own
    neighbour within ``1e-14`` relative, and that neighbour must be minimal
    within ``1e-12`` relative.
    """
    chosen = np.zeros(idx.size, dtype=np.float64)
    for k in range(int(e.shape[1])):
        difference = e[:, k] - e[idx, k]
        chosen += difference * difference
    chosen_dist = np.sqrt(chosen)
    if not np.allclose(dist, chosen_dist, rtol=1e-14, atol=0.0):
        raise ValueError("nearest-neighbor distances differ from their neighbours")
    if np.any(chosen_dist > expected_dist * (1.0 + 1e-12)):
        raise ValueError("nearest-neighbor indices are not nearest neighbours")
    return dist, idx


def _validate_selected_int(value: object, *, name: str, lower: int, upper: int) -> int:
    """Return a native integer selection inside ``[lower, upper]``, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} backend output must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} backend output {result} outside [{lower}, {upper}]")
    return result


def optimal_delay(
    signal: object,
    max_lag: object = 100,
    n_bins: object = 32,
    backend: str = "auto",
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
    backend : str
        ``"auto"`` or an explicit owner. Rust uses its native search; other
        owners compose their own MI primitive.

    Returns
    -------
    int
        The first mutual-information minimum, as a lag in samples (1 when
        there is none).

    Raises
    ------
    ValueError
        If an input is invalid or a native result is out of range.
    RuntimeError
        If an explicitly requested backend is unavailable or fails.
    """
    s = _validate_signal(signal)
    max_lag = _validate_int_at_least(max_lag, name="max_lag", minimum=1)
    n_bins = _validate_int_at_least(n_bins, name="n_bins", minimum=2)
    _require_known_backend(backend)
    searched = min(max_lag, s.size // 2)

    use_rust = backend == "rust" or (backend == "auto" and ACTIVE_BACKEND == "rust")
    if use_rust:
        owner, native = _dispatch(
            "optimal_delay", "rust" if backend == "rust" else "auto"
        )
        if owner == "rust" and native is not None:
            fn = cast("Callable[[FloatArray, int, int], int]", native)
            try:
                return _validate_selected_int(
                    fn(s, max_lag, n_bins),
                    name="optimal_delay",
                    lower=1,
                    upper=max(1, searched - 2),
                )
            except _NATIVE_FAILURES as exc:
                if backend != "auto":
                    raise _owner_failed(owner) from exc

    mi_values = np.array(
        [mutual_information(s, lag, n_bins, backend=backend) for lag in range(searched)]
    )
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
    backend: str = "auto",
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
    backend : str
        ``"auto"`` or an explicit owner. Rust uses its native search; other
        owners compose their own embedding and kNN primitives.

    Returns
    -------
    int
        The selected embedding dimension.

    Raises
    ------
    ValueError
        If an input is invalid or a native result is out of range.
    RuntimeError
        If an explicitly requested backend is unavailable or fails.
    """
    s = _validate_signal(signal)
    delay = _validate_int_at_least(delay, name="delay", minimum=1)
    max_dim = _validate_int_at_least(max_dim, name="max_dim", minimum=1)
    rtol = _validate_non_negative_real(rtol, name="rtol")
    atol = _validate_non_negative_real(atol, name="atol")
    _require_known_backend(backend)
    sigma = float(np.std(s))
    if sigma == 0:
        return 1

    use_rust = backend == "rust" or (backend == "auto" and ACTIVE_BACKEND == "rust")
    if use_rust:
        owner, native = _dispatch(
            "optimal_dimension", "rust" if backend == "rust" else "auto"
        )
        if owner == "rust" and native is not None:
            fn = cast("Callable[..., int]", native)
            try:
                return _validate_selected_int(
                    fn(s, delay, max_dim, rtol, atol),
                    name="optimal_dimension",
                    lower=1,
                    upper=max_dim,
                )
            except _NATIVE_FAILURES as exc:
                if backend != "auto":
                    raise _owner_failed(owner) from exc

    for m in range(1, max_dim + 1):
        t_next = s.size - m * delay
        if t_next <= 1:
            return m
        emb_m = delay_embed(s, delay, m, backend=backend)
        t_m = emb_m.shape[0]
        nn_dist, nn_idx = nearest_neighbor_distances(emb_m, backend=backend)

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
    backend: str = "auto",
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
    backend : str
        ``"auto"`` or an explicit owner passed to every stage.

    Returns
    -------
    EmbeddingResult
        The auto-selected delay/dimension embedding result.
    """
    tau = optimal_delay(signal, max_lag, backend=backend)
    m = optimal_dimension(signal, tau, max_dim, backend=backend)
    traj = delay_embed(signal, tau, m, backend=backend)
    return EmbeddingResult(
        trajectory=traj,
        delay=tau,
        dimension=m,
        T_effective=int(traj.shape[0]),
    )
