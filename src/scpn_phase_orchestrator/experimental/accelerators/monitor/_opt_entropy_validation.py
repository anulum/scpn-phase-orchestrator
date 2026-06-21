# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct OPT-entropy backend validation

"""Shared typed validation for ordinal-pattern-transition backend bridges.

The canonical reference computation lives here so every polyglot bridge
(Go, Julia, Mojo) and the public dispatcher all verify against one source
of truth:

* ``ordinal_pattern_sequence`` — Bandt–Pompe ordinal patterns of a scalar
  series, each window encoded as the Lehmer code of its stable ascending
  argsort permutation, an integer in ``[0, D! − 1]``.
* ``transition_entropy`` — the normalised Shannon entropy of the
  consecutive ordinal-pattern transition distribution.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
ArrayPayload: TypeAlias = NDArray[np.generic]

MIN_DIMENSION = 2
MAX_DIMENSION = 7

__all__ = [
    "FloatArray",
    "IntArray",
    "MAX_DIMENSION",
    "MIN_DIMENSION",
    "expected_ordinal_pattern_backend_output",
    "expected_transition_entropy_backend_output",
    "factorial",
    "ordinal_window_count",
    "validate_ordinal_params",
    "validate_ordinal_pattern_backend_output",
    "validate_series_backend_input",
    "validate_transition_entropy_backend_inputs",
    "validate_transition_entropy_backend_output",
]


def factorial(value: int) -> int:
    """Return ``value!`` for the small embedding dimensions used here."""
    result = 1
    for factor in range(2, value + 1):
        result *= factor
    return result


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


def validate_series_backend_input(series: object) -> FloatArray:
    """Return a finite real one-dimensional scalar series for direct backends."""
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


def validate_ordinal_params(dimension: object, delay: object) -> tuple[int, int]:
    """Return validated ``(dimension, delay)`` for an ordinal-pattern call."""
    return _validate_dimension(dimension), _validate_delay(delay)


def ordinal_window_count(length: int, dimension: int, delay: int) -> int:
    """Return the number of ordinal windows ``M = T − (D − 1)·τ`` (≥ 0)."""
    count = length - (dimension - 1) * delay
    return count if count > 0 else 0


def _ordinal_codes(series: FloatArray, dimension: int, delay: int) -> IntArray:
    """Return the Lehmer-encoded ordinal-pattern sequence of ``series``.

    Each window ``(x_m, x_{m+τ}, …, x_{m+(D−1)τ})`` is sorted ascending with
    ties broken by sample index (the Bandt–Pompe convention); the resulting
    permutation is encoded by its Lehmer code into ``[0, D! − 1]``.
    """
    length = int(series.shape[0])
    count = ordinal_window_count(length, dimension, delay)
    codes = np.empty(count, dtype=np.int64)
    fact = [factorial(k) for k in range(dimension)]
    for m in range(count):
        window = [float(series[m + k * delay]) for k in range(dimension)]
        perm = _stable_argsort(window, dimension)
        codes[m] = _lehmer_code(perm, dimension, fact)
    return codes


def _stable_argsort(window: list[float], dimension: int) -> list[int]:
    """Return the ascending argsort with index tie-breaking (selection sort)."""
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


def _lehmer_code(perm: list[int], dimension: int, fact: list[int]) -> int:
    """Return the Lehmer code of permutation ``perm`` in ``[0, D! − 1]``."""
    code = 0
    for i in range(dimension):
        smaller = 0
        for j in range(i + 1, dimension):
            if perm[j] < perm[i]:
                smaller += 1
        code += smaller * fact[dimension - 1 - i]
    return code


def _transition_entropy_from_codes(codes: IntArray, dimension: int) -> float:
    """Return the normalised ordinal-pattern transition entropy.

    Consecutive pattern pairs are packed into single integer keys, counted
    by ascending key order, and summed into a Shannon entropy normalised by
    ``ln(L)`` where ``L`` is the number of distinct observed transitions.
    """
    n_codes = int(codes.shape[0])
    if n_codes < 2:
        return 0.0
    fact_d = factorial(dimension)
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


def expected_ordinal_pattern_backend_output(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> IntArray:
    """Return the exact ordinal-pattern code sequence required of a backend."""
    return _ordinal_codes(series, dimension, delay)


def expected_transition_entropy_backend_output(
    series: FloatArray,
    dimension: int,
    delay: int,
) -> float:
    """Return the exact normalised transition-entropy scalar from a backend."""
    return _transition_entropy_from_codes(
        _ordinal_codes(series, dimension, delay), dimension
    )


def validate_ordinal_pattern_backend_output(
    codes: object,
    *,
    n_windows: int,
    dimension: int,
    expected: IntArray | None = None,
) -> IntArray:
    """Return a validated ordinal-pattern code vector from a backend."""
    raw = np.asarray(codes)
    if _contains_boolean_alias(raw):
        raise ValueError("ordinal pattern backend output must not contain booleans")
    if _contains_complex_alias(raw):
        raise ValueError("ordinal pattern backend output must contain real values")
    if not np.all(np.isfinite(raw.astype(np.float64, copy=False))):
        raise ValueError("ordinal pattern backend output must be finite")
    rounded = np.rint(raw.astype(np.float64, copy=True))
    if not np.allclose(rounded, raw.astype(np.float64, copy=False), rtol=0.0, atol=0.0):
        raise ValueError("ordinal pattern backend output must be integer-valued")
    array = rounded.astype(np.int64, copy=False)
    if array.ndim != 1 or array.size != n_windows:
        raise ValueError(
            f"ordinal pattern backend output size {array.size} does not match "
            f"{n_windows}"
        )
    fact_d = factorial(dimension)
    if array.size and (int(array.min()) < 0 or int(array.max()) >= fact_d):
        raise ValueError(
            f"ordinal pattern codes must lie in [0, {fact_d - 1}] for dimension "
            f"{dimension}"
        )
    if expected is not None and not np.array_equal(array, expected):
        raise ValueError("ordinal pattern backend output must match exact reference")
    return np.ascontiguousarray(array, dtype=np.int64)


def validate_transition_entropy_backend_inputs(
    series: object,
    dimension: object,
    delay: object,
) -> tuple[FloatArray, int, int]:
    """Return a validated ``(series, dimension, delay)`` triple."""
    validated_series = validate_series_backend_input(series)
    validated_dimension, validated_delay = validate_ordinal_params(dimension, delay)
    return validated_series, validated_dimension, validated_delay


def validate_transition_entropy_backend_output(
    value: object,
    *,
    expected: float | None = None,
    atol: float = 1.0e-12,
) -> float:
    """Return a validated normalised transition-entropy backend scalar."""
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
    if expected is not None and abs(score - expected) > atol:
        raise ValueError("transition entropy backend output must match exact reference")
    return score
