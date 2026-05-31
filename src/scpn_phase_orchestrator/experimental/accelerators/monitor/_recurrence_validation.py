# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct recurrence backend validation

"""Shared typed validation for direct recurrence backend bridge calls."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "validate_cross_recurrence_backend_inputs",
    "validate_recurrence_backend_inputs",
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


def _validate_epsilon(epsilon: object) -> float:
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError(f"epsilon must be a finite non-negative real, got {epsilon!r}")
    result = float(epsilon)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon!r}")
    return result


def _validate_angular(angular: object) -> bool:
    if not isinstance(angular, (bool, np.bool_)):
        raise ValueError(f"angular must be a boolean flag, got {angular!r}")
    return bool(angular)


def _validate_flat_trajectory(
    value: object,
    *,
    name: str,
    t: int,
    d: int,
) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real-valued trajectory samples")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a finite one-dimensional float array"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    expected = t * d
    if array.size != expected:
        raise ValueError(f"{name} length {array.size} does not match t*d = {expected}")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_recurrence_backend_inputs(
    traj_flat: object,
    t: object,
    d: object,
    epsilon: object,
    angular: object,
) -> tuple[FloatArray, int, int, float, bool]:
    """Return validated direct-backend recurrence-matrix arguments."""

    t_int = _validate_int_at_least(t, name="t", minimum=0)
    d_int = _validate_int_at_least(d, name="d", minimum=1)
    return (
        _validate_flat_trajectory(traj_flat, name="traj_flat", t=t_int, d=d_int),
        t_int,
        d_int,
        _validate_epsilon(epsilon),
        _validate_angular(angular),
    )


def validate_cross_recurrence_backend_inputs(
    traj_a_flat: object,
    traj_b_flat: object,
    t: object,
    d: object,
    epsilon: object,
    angular: object,
) -> tuple[FloatArray, FloatArray, int, int, float, bool]:
    """Return validated direct-backend cross-recurrence arguments."""

    t_int = _validate_int_at_least(t, name="t", minimum=0)
    d_int = _validate_int_at_least(d, name="d", minimum=1)
    return (
        _validate_flat_trajectory(
            traj_a_flat,
            name="traj_a_flat",
            t=t_int,
            d=d_int,
        ),
        _validate_flat_trajectory(
            traj_b_flat,
            name="traj_b_flat",
            t=t_int,
            d=d_int,
        ),
        t_int,
        d_int,
        _validate_epsilon(epsilon),
        _validate_angular(angular),
    )
