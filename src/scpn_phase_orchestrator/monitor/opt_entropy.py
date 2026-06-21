# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ordinal-Pattern Transition Entropy (OPT-entropy)

"""Ordinal-pattern transition entropy of a scalar observable.

Two primitives, each with a five-backend fallback chain:

* ``ordinal_pattern_sequence`` — the Bandt–Pompe ordinal-pattern code of every
  sliding window, encoded as the Lehmer code of its stable ascending argsort
  permutation, an integer in ``[0, D! − 1]``.
* ``transition_entropy`` — the normalised Shannon entropy of the
  consecutive-pattern transition distribution, in ``[0, 1]``.

Transition entropy collapses as a dynamical system regularises ahead of a
first-order (explosive) synchronisation onset, which makes it the compute core
of the explosive-synchronisation early-warning monitor in
``monitor/explosive_sync.py``. This module is the compute primitive; the monitor
is the detection layer built on top.

The dispatcher is self-contained (it carries its own validation and NumPy
reference, like ``monitor/npe.py``) so the core never imports the experimental
accelerator package at import time; the optional Go/Julia/Mojo bridges are
loaded lazily and validate their output against this reference.

References
----------
* Bandt & Pompe 2002, *Phys. Rev. Lett.* 88, 174102 — permutation entropy
  and ordinal patterns.
* McCullough, Small, Stemler & Iu 2015, *Chaos* 25, 053101 — ordinal
  partition (pattern transition) networks for continuous dynamics.
* Gómez-Gardeñes, Gómez, Arenas & Moreno 2011, *Phys. Rev. Lett.* 106,
  128701 — explosive synchronisation as a first-order transition.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ArrayPayload: TypeAlias = NDArray[np.generic]

DEFAULT_DIMENSION = 3
DEFAULT_DELAY = 1
MIN_DIMENSION = 2
MAX_DIMENSION = 7

# Native backends (Rust / Julia / Go) agree to machine precision; the Mojo
# text-protocol round-trip floors the scalar entropy at ~1e-10 because each
# transition probability is re-parsed from a decimal string before the log
# summation. The per-backend parity tests assert the tight 1e-12 bound on the
# compiled bridges; this dispatcher boundary absorbs the Mojo text floor.
_DISPATCH_ENTROPY_TOLERANCE = 1.0e-9

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "DEFAULT_DELAY",
    "DEFAULT_DIMENSION",
    "MAX_DIMENSION",
    "MIN_DIMENSION",
    "ordinal_pattern_sequence",
    "transition_entropy",
]


def _factorial(value: int) -> int:
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


def _ordinal_window_count(length: int, dimension: int, delay: int) -> int:
    count = length - (dimension - 1) * delay
    return count if count > 0 else 0


def _contains_boolean_alias(raw: ArrayPayload) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def _contains_complex_alias(raw: ArrayPayload) -> bool:
    if np.iscomplexobj(raw):
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, complex | np.complexfloating) for value in raw.flat)


def _validate_series(series: object) -> FloatArray:
    raw = np.asarray(series)
    if _contains_boolean_alias(raw):
        raise ValueError("series must not contain boolean values")
    if _contains_complex_alias(raw):
        raise ValueError("series must contain real-valued samples")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("series must be a one-dimensional float array") from exc
    if array.ndim != 1:
        raise ValueError(f"series shape {array.shape} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("series must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_dimension(dimension: object) -> int:
    if isinstance(dimension, (bool, np.bool_)) or not isinstance(dimension, Integral):
        raise ValueError(f"dimension must be an integer, got {dimension!r}")
    value = int(dimension)
    if value < MIN_DIMENSION or value > MAX_DIMENSION:
        raise ValueError(
            f"dimension must lie in [{MIN_DIMENSION}, {MAX_DIMENSION}], got {value}"
        )
    return value


def _validate_delay(delay: object) -> int:
    if isinstance(delay, (bool, np.bool_)) or not isinstance(delay, Integral):
        raise ValueError(f"delay must be an integer, got {delay!r}")
    value = int(delay)
    if value < 1:
        raise ValueError(f"delay must be a positive integer, got {value}")
    return value


def _validate_ordinal_params(dimension: object, delay: object) -> tuple[int, int]:
    return _validate_dimension(dimension), _validate_delay(delay)


def _validate_codes_output(
    codes: object,
    *,
    n_windows: int,
    dimension: int,
    expected: IntArray,
) -> IntArray:
    raw = np.asarray(codes)
    if _contains_boolean_alias(raw):
        raise ValueError("ordinal pattern backend output must not contain booleans")
    if _contains_complex_alias(raw):
        raise ValueError("ordinal pattern backend output must contain real values")
    floated = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(floated)):
        raise ValueError("ordinal pattern backend output must be finite")
    rounded = np.rint(floated)
    if not np.array_equal(rounded, floated):
        raise ValueError("ordinal pattern backend output must be integer-valued")
    array = rounded.astype(np.int64, copy=False)
    if array.ndim != 1 or array.size != n_windows:
        raise ValueError(
            f"ordinal pattern backend output size {array.size} does not match "
            f"{n_windows}"
        )
    fact_d = _factorial(dimension)
    if array.size and (int(array.min()) < 0 or int(array.max()) >= fact_d):
        raise ValueError(
            f"ordinal pattern codes must lie in [0, {fact_d - 1}] for dimension "
            f"{dimension}"
        )
    if not np.array_equal(array, expected):
        raise ValueError("ordinal pattern backend output must match exact reference")
    return np.ascontiguousarray(array, dtype=np.int64)


def _validate_entropy_output(
    value: object,
    *,
    expected: float,
    atol: float,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(
            f"transition entropy backend output must be a real scalar, got {value!r}"
        )
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(
            f"transition entropy backend output must be finite, got {value!r}"
        )
    tolerance = 1.0e-12
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError(
            f"transition entropy backend output must lie in [0, 1], got {value!r}"
        )
    score = min(1.0, max(0.0, score))
    if abs(score - expected) > atol:
        raise ValueError("transition entropy backend output must match exact reference")
    return score


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import ordinal_pattern_sequence as rust_codes
    from spo_kernel import transition_entropy as rust_entropy

    return {
        "ordinal_pattern_sequence": rust_codes,
        "transition_entropy": rust_entropy,
    }


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._opt_entropy_mojo import (
        _ensure_exe,
        ordinal_pattern_sequence_mojo,
        transition_entropy_mojo,
    )

    _ensure_exe()
    return {
        "ordinal_pattern_sequence": ordinal_pattern_sequence_mojo,
        "transition_entropy": transition_entropy_mojo,
    }


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._opt_entropy_julia import (
        ordinal_pattern_sequence_julia,
        transition_entropy_julia,
    )

    return {
        "ordinal_pattern_sequence": ordinal_pattern_sequence_julia,
        "transition_entropy": transition_entropy_julia,
    }


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._opt_entropy_go import (
        _load_lib,
        ordinal_pattern_sequence_go,
        transition_entropy_go,
    )

    _load_lib()
    return {
        "ordinal_pattern_sequence": ordinal_pattern_sequence_go,
        "transition_entropy": transition_entropy_go,
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


def _dispatch(fn_name: str) -> object:
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
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        if fn is None:
            continue
        return fn
    return None


def _stable_argsort(window: NDArray[np.float64]) -> list[int]:
    dimension = int(window.shape[0])
    used = [False] * dimension
    perm = [0] * dimension
    for rank in range(dimension):
        best = -1
        for idx in range(dimension):
            if used[idx]:
                continue
            if (
                best == -1
                or window[idx] < window[best]
                or (window[idx] == window[best] and idx < best)
            ):
                best = idx
        perm[rank] = best
        used[best] = True
    return perm


def _lehmer_code(perm: list[int], fact: list[int]) -> int:
    dimension = len(perm)
    code = 0
    for i in range(dimension):
        smaller = 0
        for j in range(i + 1, dimension):
            if perm[j] < perm[i]:
                smaller += 1
        code += smaller * fact[dimension - 1 - i]
    return code


def _ordinal_codes_reference(
    series: FloatArray, dimension: int, delay: int
) -> IntArray:
    count = _ordinal_window_count(int(series.shape[0]), dimension, delay)
    codes = np.empty(count, dtype=np.int64)
    fact = [_factorial(k) for k in range(dimension)]
    for m in range(count):
        window = series[m : m + (dimension - 1) * delay + 1 : delay]
        codes[m] = _lehmer_code(_stable_argsort(window), fact)
    return codes


def _transition_entropy_reference(
    series: FloatArray, dimension: int, delay: int
) -> float:
    codes = _ordinal_codes_reference(series, dimension, delay)
    n_codes = int(codes.shape[0])
    if n_codes < 2:
        return 0.0
    fact_d = _factorial(dimension)
    keys = codes[:-1] * fact_d + codes[1:]
    total = int(keys.shape[0])
    _, counts = np.unique(keys, return_counts=True)
    distinct = int(counts.shape[0])
    if distinct < 2:
        return 0.0
    entropy = 0.0
    for count in counts.tolist():
        probability = count / total
        entropy -= probability * math.log(probability)
    # distinct >= 2 guarantees log(distinct) >= log(2) > 0, so the ratio is a
    # well-defined value in [0, 1]; the clamp absorbs only float round-off.
    return min(1.0, max(0.0, entropy / math.log(distinct)))


def ordinal_pattern_sequence(
    series: FloatArray,
    dimension: int = DEFAULT_DIMENSION,
    delay: int = DEFAULT_DELAY,
) -> IntArray:
    """Lehmer-encoded ordinal-pattern sequence of a scalar series.

    Parameters
    ----------
    series : FloatArray
        Finite real one-dimensional samples, shape ``(T,)``.
    dimension : int
        Embedding dimension ``D`` in ``[2, 7]`` (default 3); each window
        spans ``D`` samples.
    delay : int
        Positive embedding delay ``τ`` (default 1).

    Returns
    -------
    IntArray
        Pattern codes in ``[0, D! − 1]``, shape ``(T − (D − 1)·τ,)``;
        empty when the series is shorter than one window.
    """
    series = _validate_series(series)
    dimension, delay = _validate_ordinal_params(dimension, delay)
    count = _ordinal_window_count(int(series.shape[0]), dimension, delay)
    expected = _ordinal_codes_reference(series, dimension, delay)
    if count == 0:
        return expected

    backend_fn = _dispatch("ordinal_pattern_sequence")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int], IntArray]", backend_fn)
        try:
            return _validate_codes_output(
                fn(series, dimension, delay),
                n_windows=count,
                dimension=dimension,
                expected=expected,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    return expected


def transition_entropy(
    series: FloatArray,
    dimension: int = DEFAULT_DIMENSION,
    delay: int = DEFAULT_DELAY,
) -> float:
    """Normalised ordinal-pattern transition entropy in ``[0, 1]``.

    The consecutive ordinal-pattern pairs of ``series`` form a transition
    distribution; this returns its Shannon entropy normalised by ``ln(L)``
    where ``L`` is the number of distinct observed transitions. ``~0`` means
    near-deterministic transitions (regular / locked dynamics); ``~1`` means
    a uniform spread of transitions (disordered dynamics).

    Parameters
    ----------
    series : FloatArray
        Finite real one-dimensional samples, shape ``(T,)``.
    dimension : int
        Embedding dimension ``D`` in ``[2, 7]`` (default 3).
    delay : int
        Positive embedding delay ``τ`` (default 1).

    Returns
    -------
    float
        The normalised transition entropy; ``0.0`` when fewer than two
        transitions or a single transition type is observed.
    """
    series = _validate_series(series)
    dimension, delay = _validate_ordinal_params(dimension, delay)
    expected = _transition_entropy_reference(series, dimension, delay)

    backend_fn = _dispatch("transition_entropy")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int], float]", backend_fn)
        try:
            return _validate_entropy_output(
                fn(series, dimension, delay),
                expected=expected,
                atol=_DISPATCH_ENTROPY_TOLERANCE,
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    return expected
