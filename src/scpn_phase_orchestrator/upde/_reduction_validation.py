# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reduction backend validation contracts

"""Shared validation for direct Ott-Antonsen accelerator backends."""

from __future__ import annotations

import math
from numbers import Complex, Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = [
    "validate_oa_frequency_sample",
    "validate_oa_inputs",
    "validate_oa_output",
    "validate_oa_steady_state_output",
]

ValidatedOAInputs: TypeAlias = tuple[float, float, float, float, float, float, int]
OAOutput: TypeAlias = tuple[float, float, float, float]
FloatArray: TypeAlias = NDArray[np.float64]

_UNIT_DISK_TOLERANCE = 1e-12
_OUTPUT_MAGNITUDE_TOLERANCE = 1e-9
_OUTPUT_PHASE_TOLERANCE = 1e-8
_ZERO_RADIUS_TOLERANCE = 1e-12


def _is_bool_alias(value: Any) -> bool:
    """Return whether the value is a boolean alias."""
    return isinstance(value, (bool, np.bool_))


def _as_real_scalar(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if _is_bool_alias(value) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _as_positive_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if _is_bool_alias(value) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1")
    return out


def _as_real_array(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a finite real array without alias coercion."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    object_values = array.dtype == np.object_
    if array.dtype == np.bool_ or (object_values and contains_boolean_alias(array)):
        raise ValueError(f"{name} must be real-valued, not boolean")
    contains_complex_object = object_values and any(
        isinstance(item, Complex) and not isinstance(item, Real) for item in array.flat
    )
    if np.iscomplexobj(array) or contains_complex_object:
        raise ValueError(f"{name} must be real-valued, not complex")
    numeric_object = object_values and all(
        isinstance(item, Real) for item in array.flat
    )
    if not np.issubdtype(array.dtype, np.number) and not numeric_object:
        raise ValueError(f"{name} must be numeric")
    out = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def validate_oa_frequency_sample(value: Any, *, name: str) -> FloatArray:
    """Validate an empirical frequency sample before Lorentzian fitting."""
    array = _as_real_array(value, name=name)
    if array.ndim != 1:
        raise ValueError(f"{name} shape {array.shape} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one frequency")
    return array


def validate_oa_steady_state_output(value: Any) -> float:
    """Validate a steady-state order parameter before publication."""
    radius = _as_real_scalar(value, name="backend steady-state R output")
    if radius < -_UNIT_DISK_TOLERANCE or radius > 1.0 + _UNIT_DISK_TOLERANCE:
        raise ValueError("backend steady-state R output must lie in [0, 1]")
    return min(1.0, max(0.0, radius))


def validate_oa_inputs(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> ValidatedOAInputs:
    """Validate direct Ott-Antonsen RK4 kernel inputs before runtime loading."""
    z_re_f = _as_real_scalar(z_re, name="z_re")
    z_im_f = _as_real_scalar(z_im, name="z_im")
    if math.hypot(z_re_f, z_im_f) > 1.0 + _UNIT_DISK_TOLERANCE:
        raise ValueError("z must lie in the OA unit disk")
    omega_0_f = _as_real_scalar(omega_0, name="omega_0")
    delta_f = _as_real_scalar(delta, name="delta")
    if delta_f < 0.0:
        raise ValueError("delta must be non-negative")
    k_coupling_f = _as_real_scalar(k_coupling, name="k_coupling")
    dt_f = _as_real_scalar(dt, name="dt")
    if dt_f <= 0.0:
        raise ValueError("dt must be positive")
    return (
        z_re_f,
        z_im_f,
        omega_0_f,
        delta_f,
        k_coupling_f,
        dt_f,
        _as_positive_int(n_steps, name="n_steps"),
    )


def validate_oa_output(
    z_re: object,
    z_im: object,
    radius: object,
    psi: object,
) -> OAOutput:
    """Validate a backend OA state before publishing it to the public API."""
    z_re_f = _as_real_scalar(z_re, name="backend z_re output")
    z_im_f = _as_real_scalar(z_im, name="backend z_im output")
    radius_f = _as_real_scalar(radius, name="backend R output")
    psi_f = _as_real_scalar(psi, name="backend psi output")
    magnitude = math.hypot(z_re_f, z_im_f)
    if magnitude > 1.0 + _UNIT_DISK_TOLERANCE:
        raise ValueError("backend z output must lie in the OA unit disk")
    if radius_f < -_UNIT_DISK_TOLERANCE or radius_f > 1.0 + _UNIT_DISK_TOLERANCE:
        raise ValueError("backend R output must lie in [0, 1]")
    if not math.isclose(
        radius_f,
        magnitude,
        rel_tol=_OUTPUT_MAGNITUDE_TOLERANCE,
        abs_tol=_OUTPUT_MAGNITUDE_TOLERANCE,
    ):
        raise ValueError("backend R must match |z|")
    if magnitude > _ZERO_RADIUS_TOLERANCE:
        expected_psi = math.atan2(z_im_f, z_re_f)
        wrapped_error = math.atan2(
            math.sin(psi_f - expected_psi),
            math.cos(psi_f - expected_psi),
        )
        if abs(wrapped_error) > _OUTPUT_PHASE_TOLERANCE:
            raise ValueError("backend psi must match atan2(z_im, z_re)")
    return (
        z_re_f,
        z_im_f,
        min(1.0, max(0.0, radius_f)),
        psi_f,
    )
