# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct dimension backend validation

"""Shared typed validation for direct fractal-dimension backend bridge calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "FloatArray",
    "IntArray",
    "validate_correlation_integral_backend_inputs",
    "validate_correlation_integral_backend_output",
    "validate_kaplan_yorke_backend_output",
    "validate_kaplan_yorke_backend_input",
]


def _contains_boolean_alias(raw: np.ndarray) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a finite one-dimensional float array"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_index_vector(value: object, *, name: str, upper_bound: int) -> IntArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain integer indices")
    try:
        numeric = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional integer array") from exc
    if numeric.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {numeric.shape}")
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError(f"{name} must contain integer indices")
    indices = numeric.astype(np.int64, copy=True)
    if upper_bound <= 0:
        if indices.size:
            raise ValueError(f"{name} must be empty when t is zero")
        return np.ascontiguousarray(indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= upper_bound):
        raise ValueError(f"{name} indices must lie in [0, {upper_bound})")
    return np.ascontiguousarray(indices, dtype=np.int64)


def _validate_epsilons(epsilons: object) -> FloatArray:
    eps = _validate_float_vector(epsilons, name="epsilons")
    if np.any(eps < 0.0):
        raise ValueError("epsilons must contain only finite non-negative values")
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


def validate_correlation_integral_backend_output(
    values: object,
    epsilons: object,
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
    return np.ascontiguousarray(np.clip(result, 0.0, 1.0), dtype=np.float64)


def validate_kaplan_yorke_backend_output(
    value: object,
    lyapunov_exponents: object,
) -> float:
    """Validate a direct-backend Kaplan-Yorke dimension estimate."""

    spectrum = validate_kaplan_yorke_backend_input(lyapunov_exponents)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("kaplan_yorke_dimension must not be a boolean value")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("kaplan_yorke_dimension must be real-valued")
    try:
        dimension = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("kaplan_yorke_dimension must be a finite real scalar") from exc
    if not np.isfinite(dimension):
        raise ValueError("kaplan_yorke_dimension must be finite")
    if dimension < -1e-12 or dimension > spectrum.size + 1e-12:
        raise ValueError("kaplan_yorke_dimension must lie in [0, spectrum length]")
    return min(max(dimension, 0.0), float(spectrum.size))
