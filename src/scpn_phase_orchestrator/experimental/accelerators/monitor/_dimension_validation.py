# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct dimension backend validation

"""Shared typed validation for direct fractal-dimension backend bridge calls."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator.monitor._dimension_numerics import (
    kaplan_yorke_reference,
    row_distances,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "FloatArray",
    "IntArray",
    "expected_correlation_integral_backend_output",
    "expected_kaplan_yorke_backend_output",
    "validate_correlation_integral_backend_inputs",
    "validate_correlation_integral_backend_output",
    "validate_kaplan_yorke_backend_output",
    "validate_kaplan_yorke_backend_input",
]


def _contains_boolean_alias(value: object) -> bool:
    """Inspect original scalar types before NumPy can coerce mixed booleans."""
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string-like scalar parsable as a float."""
    if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether the original scalar payload contains numeric strings."""
    raw = np.asarray(value, dtype=object)
    return any(_is_numeric_string_alias(item) for item in raw.flat)


def _has_complex_payload(value: object) -> bool:
    """Return whether the value carries a complex-number payload."""
    raw = np.asarray(value)
    return bool(np.iscomplexobj(raw) or _contains_complex_alias(value))


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    """Return ``value`` as an integer at least the minimum, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite float vector, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_numeric_string_alias(raw):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must contain real values")
    try:
        require_real_values(value, name=name, allow_object=True)
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        msg = f"{name} must be a finite one-dimensional float array"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_index_vector(value: object, *, name: str, upper_bound: int) -> IntArray:
    """Return ``value`` as a validated integer index vector, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_numeric_string_alias(raw):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must contain integer indices")
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {raw.shape}")
    if upper_bound <= 0 and raw.size:
        raise ValueError(f"{name} must be empty when t is zero")
    indices: list[int] = []
    for item in np.asarray(value, dtype=object):
        if isinstance(item, Integral):
            index = int(item)
        elif (
            isinstance(item, Real)
            and isfinite(float(item))
            and float(item).is_integer()
        ):
            index = int(float(item))
        else:
            raise ValueError(f"{name} must contain integer indices")
        if index < 0 or index >= upper_bound or index > np.iinfo(np.int64).max:
            raise ValueError(f"{name} indices must lie in [0, {upper_bound})")
        indices.append(index)
    return np.array(indices, dtype=np.int64)


def _validate_epsilons(epsilons: object) -> FloatArray:
    """Return the validated epsilon radii, else raise."""
    eps = _validate_float_vector(epsilons, name="epsilons")
    if np.any(eps < 0.0):
        raise ValueError("epsilons must contain only finite non-negative values")
    if np.any(np.diff(eps) < 0.0):
        raise ValueError("epsilons must be sorted in non-decreasing order")
    return eps


def validate_correlation_integral_backend_inputs(
    traj_flat: object,
    t: object,
    d: object,
    idx_i: object,
    idx_j: object,
    epsilons: object,
) -> tuple[FloatArray, int, int, IntArray, IntArray, FloatArray]:
    """Return validated direct-backend correlation-integral arguments."""
    t_int = _validate_int_at_least(t, name="t", minimum=0)
    d_int = _validate_int_at_least(d, name="d", minimum=1)
    traj = _validate_float_vector(traj_flat, name="traj_flat")
    expected = t_int * d_int
    if traj.size != expected:
        raise ValueError(
            f"traj_flat length {traj.size} does not match t*d = {expected}"
        )
    ii = _validate_index_vector(idx_i, name="idx_i", upper_bound=t_int)
    jj = _validate_index_vector(idx_j, name="idx_j", upper_bound=t_int)
    if ii.size != jj.size:
        raise ValueError("idx_i and idx_j must have the same length")
    if np.any(ii == jj):
        raise ValueError("idx_i and idx_j must not describe self-pairs")
    eps = _validate_epsilons(epsilons)
    return traj, t_int, d_int, ii, jj, eps


def validate_kaplan_yorke_backend_input(
    lyapunov_exponents: object,
) -> FloatArray:
    """Return a finite real one-dimensional Lyapunov spectrum."""
    return _validate_float_vector(lyapunov_exponents, name="lyapunov_exponents")


def expected_correlation_integral_backend_output(
    traj_flat: FloatArray,
    t: int,
    d: int,
    idx_i: IntArray,
    idx_j: IntArray,
    epsilons: FloatArray,
) -> FloatArray:
    """Return the exact Grassberger-Procaccia result for direct backend inputs."""
    if idx_i.size == 0:
        return np.zeros(epsilons.size, dtype=np.float64)
    traj = np.ascontiguousarray(traj_flat.reshape(t, d), dtype=np.float64)
    dists = row_distances(traj[idx_i], traj[idx_j])
    return np.ascontiguousarray(
        np.array([np.sum(dists < eps) / dists.size for eps in epsilons]),
        dtype=np.float64,
    )


def expected_kaplan_yorke_backend_output(lyapunov_exponents: FloatArray) -> float:
    """Return the exact Kaplan-Yorke dimension for a Lyapunov spectrum."""
    return kaplan_yorke_reference(lyapunov_exponents)


def validate_correlation_integral_backend_output(
    values: object,
    epsilons: object,
    *,
    expected: object | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    """Validate a direct-backend Grassberger-Procaccia C(epsilon) vector."""
    result = _validate_float_vector(values, name="correlation_integral")
    eps = _validate_epsilons(epsilons)
    if result.size != eps.size:
        raise ValueError(
            "correlation_integral length "
            f"{result.size} does not match epsilons length {eps.size}"
        )
    if np.any((result < -1e-12) | (result > 1.0 + 1e-12)):
        raise ValueError("correlation_integral values must lie in [0, 1]")
    if np.any(np.diff(result) < -1e-12):
        raise ValueError("correlation_integral must be non-decreasing in epsilon")
    clipped = np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float64)
    if expected is not None:
        expected_values = _validate_float_vector(expected, name="expected")
        if expected_values.shape != clipped.shape:
            raise ValueError("expected correlation_integral shape must match output")
        if not np.allclose(clipped, expected_values, rtol=0.0, atol=atol):
            raise ValueError(
                "correlation_integral backend output must match exact reference"
            )
    return clipped


def validate_kaplan_yorke_backend_output(
    value: object,
    lyapunov_exponents: object,
    *,
    expected: object | None = None,
    atol: float = 1e-12,
) -> float:
    """Validate a direct-backend Kaplan-Yorke dimension estimate."""
    spectrum = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("kaplan_yorke_dimension must not be a boolean value")
    raw = np.asarray(value)
    if _is_numeric_string_alias(value) or _contains_numeric_string_alias(raw):
        raise ValueError("kaplan_yorke_dimension must not be a numeric-string alias")
    if _has_complex_payload(value):
        raise ValueError("kaplan_yorke_dimension must be real-valued")
    try:
        dimension = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("kaplan_yorke_dimension must be a finite real scalar") from exc
    if not np.isfinite(dimension):
        raise ValueError("kaplan_yorke_dimension must be finite")
    if dimension < -1e-12 or dimension > spectrum.size + 1e-12:
        raise ValueError("kaplan_yorke_dimension must lie in [0, spectrum length]")
    clipped = min(max(dimension, 0.0), float(spectrum.size))
    if expected is not None:
        expected_raw = np.asarray(expected)
        if _is_numeric_string_alias(expected) or _contains_numeric_string_alias(
            expected_raw
        ):
            raise ValueError(
                "expected kaplan_yorke_dimension must not be a numeric-string alias"
            )
        if _has_complex_payload(expected):
            raise ValueError("expected kaplan_yorke_dimension must be real-valued")
        try:
            expected_value = float(expected_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "expected kaplan_yorke_dimension must be a finite real scalar"
            ) from exc
        if not np.isfinite(expected_value):
            raise ValueError("expected kaplan_yorke_dimension must be finite")
        if abs(clipped - expected_value) > atol:
            raise ValueError(
                "kaplan_yorke_dimension backend output must match exact reference"
            )
    return clipped
