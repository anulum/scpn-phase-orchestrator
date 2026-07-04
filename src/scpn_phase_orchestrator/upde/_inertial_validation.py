# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - inertial backend validation

"""Shared validation for direct inertial Kuramoto accelerator bridges."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

FloatArray: TypeAlias = NDArray[np.float64]
TWO_PI = 2.0 * np.pi

__all__ = ["validate_inertial_inputs", "validate_inertial_output"]


def _is_string_like(value: object) -> bool:
    """Return whether ``value`` is a Python or NumPy string scalar."""
    return isinstance(value, (str, bytes, np.str_, np.bytes_))


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string scalar accepted by ``float``."""
    if not _is_string_like(value):
        return False
    text = value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
    if text.strip() == "":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` contains a stringified numeric scalar."""
    if _is_numeric_string_alias(value):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.dtype.kind not in {"O", "S", "U"}:
        return False
    object_array = array.astype(object, copy=False)
    return any(_is_numeric_string_alias(item) for item in object_array.flat)


def _as_real_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real vector, else raise."""
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if contains_boolean_alias(value):
        raise ValueError(f"{name} must be real-valued, not boolean")
    arr = np.asarray(value)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional float64 vector")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a finite real-valued vector")
    if np.iscomplexobj(arr):
        raise ValueError(f"{name} must be real-valued")
    out = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_positive_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if _is_numeric_string_alias(value):
        raise ValueError(
            f"{name} must be >= 1 as a non-boolean integer, not a numeric-string alias"
        )
    if contains_boolean_alias(value):
        raise ValueError(f"{name} must be a non-boolean integer")
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer")
    return int(value)


def _validate_positive_real(value: Any, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if _is_numeric_string_alias(value):
        raise ValueError(
            f"{name} must be positive finite real, not a numeric-string alias"
        )
    if contains_boolean_alias(value):
        raise ValueError(f"{name} must be positive finite real, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real")
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} must be positive finite real")
    return out


def _validate_length(value: Any, *, name: str, n: int) -> FloatArray:
    """Assert the array length matches the expected size, else raise."""
    arr = _as_real_vector(value, name=name)
    if arr.size != n:
        raise ValueError(f"{name} must have length n")
    return arr


def _validate_positive_vector(value: Any, *, name: str, n: int) -> FloatArray:
    """Return ``value`` as a validated strictly positive vector, else raise."""
    arr = _validate_length(value, name=name, n=n)
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must contain only positive finite values")
    return arr


def _validate_coupling(value: Any, *, n: int) -> FloatArray:
    """Return the coupling as a validated finite matrix, else raise."""
    arr = _as_real_vector(value, name="knm_flat")
    if arr.size != n * n:
        raise ValueError("knm_flat must have exactly n*n entries")
    mat = arr.reshape(n, n)
    if np.any(np.diag(mat) != 0.0):
        raise ValueError("knm_flat diagonal must be zero")
    return arr


def validate_inertial_inputs(
    theta: Any,
    omega_dot: Any,
    power: Any,
    knm_flat: Any,
    inertia: Any,
    damping: Any,
    n: Any,
    dt: Any,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    float,
]:
    """Validate and normalise direct inertial backend inputs."""
    n_i = _validate_positive_int(n, name="n")
    theta_v = _validate_length(theta, name="theta", n=n_i)
    omega_v = _validate_length(omega_dot, name="omega_dot", n=n_i)
    power_v = _validate_length(power, name="power", n=n_i)
    knm_v = _validate_coupling(knm_flat, n=n_i)
    inertia_v = _validate_positive_vector(inertia, name="inertia", n=n_i)
    damping_v = _validate_positive_vector(damping, name="damping", n=n_i)
    dt_f = _validate_positive_real(dt, name="dt")
    return theta_v, omega_v, power_v, knm_v, inertia_v, damping_v, n_i, dt_f


def validate_inertial_output(
    theta: Any,
    omega_dot: Any,
    *,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    """Validate direct backend state output."""
    theta_v = _validate_length(theta, name="inertial backend theta", n=n)
    omega_v = _validate_length(omega_dot, name="inertial backend omega_dot", n=n)
    if np.any(theta_v < -1e-12) or np.any(theta_v >= TWO_PI + 1e-12):
        raise ValueError("inertial backend theta must be in [0, 2*pi)")
    return np.mod(theta_v, TWO_PI), omega_v
