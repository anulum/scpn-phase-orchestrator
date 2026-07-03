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
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ArrayPayload: TypeAlias = NDArray[np.generic]

__all__ = [
    "FloatArray",
    "expected_recurrence_backend_output",
    "validate_cross_recurrence_backend_inputs",
    "validate_recurrence_backend_output",
    "validate_recurrence_backend_inputs",
]


def _contains_boolean_alias(raw: ArrayPayload) -> bool:
    """Return whether the value contains any boolean alias."""
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


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


def _contains_numeric_string_alias(raw: ArrayPayload) -> bool:
    """Return whether the array contains only numeric string aliases."""
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    saw_string = False
    for value in raw.astype(object, copy=False).flat:
        if not _is_string_like(value):
            continue
        saw_string = True
        if not _is_numeric_string_alias(value):
            return False
    return saw_string


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    """Return ``value`` as an integer at least the minimum, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_epsilon(epsilon: object) -> float:
    """Return the validated epsilon radius, else raise."""
    if isinstance(epsilon, (bool, np.bool_)) or not isinstance(epsilon, Real):
        raise ValueError(f"epsilon must be a finite non-negative real, got {epsilon!r}")
    result = float(epsilon)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon!r}")
    return result


def _validate_angular(angular: object) -> bool:
    """Return ``value`` as a validated angular (phase) value, else raise."""
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
    """Return ``value`` as a validated flattened trajectory, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_numeric_string_alias(raw):
        raise ValueError(f"{name} must not contain numeric-string aliases")
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


def expected_recurrence_backend_output(
    traj_a_flat: FloatArray,
    traj_b_flat: FloatArray,
    *,
    t: int,
    d: int,
    epsilon: float,
    angular: bool,
) -> NDArray[np.uint8]:
    """Return the exact binary recurrence relation required from a backend."""
    a = traj_a_flat.reshape(t, d)
    b = traj_b_flat.reshape(t, d)
    diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
    if angular:
        dist = np.sqrt(np.sum(4.0 * np.sin(diff / 2.0) ** 2, axis=2))
    else:
        dist = np.sqrt(np.sum(diff**2, axis=2))
    return np.ascontiguousarray((dist <= epsilon).astype(np.uint8).ravel())


def validate_recurrence_backend_output(
    value: object,
    *,
    t: object,
    name: str,
    expected: object | None = None,
) -> NDArray[np.uint8]:
    """Validate direct-backend recurrence output before returning it."""
    t_int = _validate_int_at_least(t, name="t", minimum=0)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} output must be array-like") from exc
    if array.size != t_int * t_int:
        raise ValueError(f"{name} output size must be {t_int * t_int}")
    if _contains_numeric_string_alias(array):
        raise ValueError(f"{name} output must not contain numeric-string aliases")
    try:
        numeric = array.reshape(t_int, t_int).astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} output must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} output must contain only finite values")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} output must contain only 0/1 values")
    if name == "recurrence_matrix":
        if not np.all(numeric.diagonal() == 1.0):
            raise ValueError("recurrence_matrix output must have true diagonal")
        if not np.array_equal(numeric, numeric.T):
            raise ValueError("recurrence_matrix output must be symmetric")
    result = np.ascontiguousarray(numeric.ravel().astype(np.uint8), dtype=np.uint8)
    if expected is not None:
        try:
            expected_array = np.asarray(expected).reshape(t_int, t_int)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} expected output must have size {t_int * t_int}"
            ) from exc
        if _contains_numeric_string_alias(expected_array):
            raise ValueError(
                f"{name} expected output must not contain numeric-string aliases"
            )
        try:
            expected_numeric = expected_array.astype(np.uint8, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} expected output must be numeric") from exc
        if not np.array_equal(result.reshape(t_int, t_int), expected_numeric):
            raise ValueError(f"{name} output must match exact recurrence threshold")
    return result
