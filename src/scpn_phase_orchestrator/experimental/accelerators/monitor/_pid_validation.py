# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PID backend boundary validation

"""Shared validation for direct partial-information-decomposition backends."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

__all__ = [
    "validate_pid_backend_inputs",
    "validate_pid_scalar_output",
]


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _validate_count(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        vector = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite one-dimensional float array"
        ) from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _validate_group(value: object, *, name: str, n: int) -> IntArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be integer-valued")
    try:
        indices = raw.astype(np.int64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer index array") from exc
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {indices.shape}")
    if indices.size > 0 and (np.any(indices < 0) or np.any(indices >= n)):
        raise ValueError(f"{name} indices must lie in [0, {n})")
    return np.ascontiguousarray(indices, dtype=np.int64)


def validate_pid_backend_inputs(
    phase_history_flat: object,
    t: object,
    n: object,
    group_a: object,
    group_b: object,
    n_bins: object,
) -> tuple[FloatArray, int, int, IntArray, IntArray, int]:
    """Validate direct PID backend inputs before optional runtime loading."""
    t_int = _validate_count(t, name="t", minimum=0)
    n_int = _validate_count(n, name="n", minimum=1)
    bins = _validate_count(n_bins, name="n_bins", minimum=2)
    history = _validate_float_vector(phase_history_flat, name="phase_history")
    expected = t_int * n_int
    if history.size != expected:
        raise ValueError(
            f"phase_history length {history.size} does not match t*n={expected}"
        )
    group_a_idx = _validate_group(group_a, name="group_a", n=n_int)
    group_b_idx = _validate_group(group_b, name="group_b", n=n_int)
    return history, t_int, n_int, group_a_idx, group_b_idx, bins


def validate_pid_scalar_output(value: object, *, name: str) -> float:
    """Validate a single PID component returned by a backend."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not be a boolean value")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a real scalar")
    try:
        scalar = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real scalar") from exc
    if scalar.shape != ():
        raise ValueError(f"{name} must be a scalar")
    result = float(scalar)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
