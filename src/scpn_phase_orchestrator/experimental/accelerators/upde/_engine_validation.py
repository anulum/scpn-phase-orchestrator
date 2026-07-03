# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - UPDE backend validation contracts

"""Shared input contracts for direct UPDE polyglot backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = [
    "validate_upde_backend_inputs",
    "validate_upde_backend_output",
    "validate_upde_schedule_backend_inputs",
]

FloatArray: TypeAlias = NDArray[np.float64]
ValidatedInputs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    int,
    str,
    int,
    float,
    float,
]
ValidatedScheduleInputs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    int,
    str,
    int,
    float,
    float,
]

_METHODS = frozenset({"euler", "rk4", "rk45"})


def _as_real_finite_array(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real array, else raise."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be real-valued, not boolean")
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued, not complex")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    out = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated 1-D finite array, else raise."""
    array = _as_real_finite_array(value, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one oscillator")
    return array


def _as_square_flat(value: Any, *, name: str, n: int) -> FloatArray:
    """Return ``value`` as a validated flattened square matrix, else raise."""
    array = _as_real_finite_array(value, name=name)
    if array.ndim == 2:
        if array.shape != (n, n):
            raise ValueError(f"{name} must have shape ({n}, {n})")
        matrix = array
    elif array.ndim == 1:
        if array.size != n * n:
            raise ValueError(f"{name} must contain {n * n} flattened values")
        matrix = array.reshape((n, n))
    else:
        raise ValueError(f"{name} must be a square matrix or flattened matrix")
    return np.ascontiguousarray(matrix.ravel(), dtype=np.float64)


def _as_finite_real(value: Any, *, name: str, positive: bool = False) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be a real scalar, not boolean")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_non_negative_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _as_positive_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_method(value: Any) -> str:
    """Return the validated integration-method name, else raise."""
    if not isinstance(value, str):
        raise TypeError("method must be a string")
    if value not in _METHODS:
        expected = sorted(_METHODS)
        raise ValueError(f"unknown method {value!r}; expected one of {expected}")
    return value


def validate_upde_backend_inputs(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> ValidatedInputs:
    """Normalise and validate direct UPDE backend call arguments."""
    p = _as_vector(phases, name="phases").copy()
    o = _as_vector(omegas, name="omegas")
    if o.size != p.size:
        raise ValueError("omegas must have the same length as phases")
    k = _as_square_flat(knm, name="knm", n=int(p.size))
    if np.any(np.diag(k.reshape((p.size, p.size))) != 0.0):
        raise ValueError("knm diagonal must be exactly zero")
    a = _as_square_flat(alpha, name="alpha", n=int(p.size))
    return (
        p,
        o,
        k,
        a,
        _as_finite_real(zeta, name="zeta"),
        _as_finite_real(psi, name="psi"),
        _as_finite_real(dt, name="dt", positive=True),
        _as_non_negative_int(n_steps, name="n_steps"),
        _as_method(method),
        _as_positive_int(n_substeps, name="n_substeps"),
        _as_finite_real(atol, name="atol", positive=True),
        _as_finite_real(rtol, name="rtol", positive=True),
    )


def validate_upde_schedule_backend_inputs(
    phases: FloatArray,
    omega_schedule: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    method: str,
    n_substeps: int,
    atol: float,
    rtol: float,
) -> ValidatedScheduleInputs:
    """Normalise direct UPDE backend arguments with per-step frequencies."""
    p = _as_vector(phases, name="phases").copy()
    schedule = _as_real_finite_array(omega_schedule, name="omega_schedule")
    if schedule.ndim != 2:
        raise ValueError("omega_schedule must be a two-dimensional matrix")
    if schedule.shape[0] == 0:
        raise ValueError("omega_schedule must contain at least one step")
    if schedule.shape[1] != p.size:
        raise ValueError("omega_schedule column count must match phases")
    schedule = np.ascontiguousarray(schedule, dtype=np.float64)
    k = _as_square_flat(knm, name="knm", n=int(p.size))
    if np.any(np.diag(k.reshape((p.size, p.size))) != 0.0):
        raise ValueError("knm diagonal must be exactly zero")
    a = _as_square_flat(alpha, name="alpha", n=int(p.size))
    return (
        p,
        schedule,
        k,
        a,
        _as_finite_real(zeta, name="zeta"),
        _as_finite_real(psi, name="psi"),
        _as_finite_real(dt, name="dt", positive=True),
        int(schedule.shape[0]),
        _as_method(method),
        _as_positive_int(n_substeps, name="n_substeps"),
        _as_finite_real(atol, name="atol", positive=True),
        _as_finite_real(rtol, name="rtol", positive=True),
    )


def validate_upde_backend_output(value: Any, *, n: int) -> FloatArray:
    """Validate a backend phase-vector result before returning it."""
    out = _as_real_finite_array(value, name="result")
    if out.ndim != 1:
        raise ValueError("result must be a one-dimensional vector")
    if out.size != n:
        raise ValueError(f"result must contain {n} values")
    return out
