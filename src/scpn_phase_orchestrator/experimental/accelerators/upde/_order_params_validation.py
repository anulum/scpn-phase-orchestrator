# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order-parameter backend validation

"""Typed validation for direct Kuramoto order-parameter accelerators."""

from __future__ import annotations

from typing import SupportsFloat, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_UNIT_INTERVAL_TOL = 1.0e-12

__all__ = [
    "FloatArray",
    "IntArray",
    "validate_layer_coherence_inputs",
    "validate_order_parameter_inputs",
    "validate_order_parameter_output",
    "validate_plv_inputs",
    "validate_unit_interval_output",
]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        values = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in values.flat)


def validate_phase_vector(value: object, *, name: str) -> FloatArray:
    """Return a contiguous finite real phase vector."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be array-like") from exc
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional phase vector")
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    values = np.ascontiguousarray(raw.astype(np.float64, copy=True))
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values


def validate_index_vector(value: object, *, name: str, n_phases: int) -> IntArray:
    """Return a contiguous non-repeating oscillator index vector."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index vector")
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer values")
    values = np.ascontiguousarray(raw.astype(np.int64, copy=True))
    if values.size > 0 and (np.any(values < 0) or np.any(values >= n_phases)):
        raise ValueError(f"{name} must reference existing oscillators")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must not repeat oscillators")
    return values


def validate_order_parameter_inputs(phases: object) -> FloatArray:
    """Validate phases before dispatching to an optional order-parameter runtime."""
    return validate_phase_vector(phases, name="phases")


def validate_plv_inputs(
    phases_a: object,
    phases_b: object,
) -> tuple[FloatArray, FloatArray]:
    """Validate equal-length PLV phase vectors."""
    a64 = validate_phase_vector(phases_a, name="phases_a")
    b64 = validate_phase_vector(phases_b, name="phases_b")
    if a64.size != b64.size:
        raise ValueError(
            f"PLV requires equal-length arrays, got {a64.size} vs {b64.size}"
        )
    return a64, b64


def validate_layer_coherence_inputs(
    phases: object,
    indices: object,
) -> tuple[FloatArray, IntArray]:
    """Validate layer-coherence phase and oscillator-index vectors."""
    phases64 = validate_phase_vector(phases, name="phases")
    indices64 = validate_index_vector(indices, name="indices", n_phases=phases64.size)
    return phases64, indices64


def _finite_scalar(value: object, *, name: str) -> float:
    """Return ``value`` as a finite scalar, else raise ``ValueError``."""
    try:
        scalar = float(cast("SupportsFloat | str | bytes | bytearray", value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def validate_unit_interval_output(value: object, *, name: str) -> float:
    """Validate a finite coherence magnitude in the physical unit interval."""
    scalar = _finite_scalar(value, name=name)
    if scalar < -_UNIT_INTERVAL_TOL or scalar > 1.0 + _UNIT_INTERVAL_TOL:
        raise ValueError(f"{name} must lie in [0, 1]")
    return float(np.clip(scalar, 0.0, 1.0))


def validate_order_parameter_output(r: object, psi: object) -> tuple[float, float]:
    """Validate and canonicalise backend ``(R, psi)`` output."""
    r_value = validate_unit_interval_output(r, name="R")
    psi_value = _finite_scalar(psi, name="mean phase")
    return r_value, float(psi_value % TWO_PI)
