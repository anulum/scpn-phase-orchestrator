# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entropy production backend input validation

"""Shared adapter-boundary validation for entropy-production backends."""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "validate_entropy_prod_backend_inputs",
    "validate_entropy_prod_backend_output",
]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    array = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    array = np.asarray(value, dtype=object)
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


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


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether the value contains numeric-string aliases."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    saw_string = False
    for item in raw.astype(object, copy=False).flat:
        if not _is_string_like(item):
            continue
        saw_string = True
        if not _is_numeric_string_alias(item):
            return False
    return saw_string


def _has_complex_payload(value: object) -> bool:
    """Return whether the value carries a complex-number payload."""
    array = np.asarray(value)
    return bool(np.iscomplexobj(array) or _contains_complex_alias(value))


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if _has_complex_payload(value):
        raise ValueError(f"{name} must be a finite real-valued scalar")
    if _is_numeric_string_alias(value):
        raise ValueError(f"{name} must not be a numeric-string alias")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated 1-D finite array, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must contain real-valued samples")
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional float array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} shape {array.shape} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_matrix(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    """Return ``value`` as a validated finite matrix, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must contain real-valued couplings")
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a two-dimensional float array") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_entropy_prod_backend_inputs(
    phases: object,
    omegas: object,
    knm: object,
    alpha: object,
    dt: object,
) -> tuple[FloatArray, FloatArray, FloatArray, float, float]:
    """Validate direct entropy-production backend adapter inputs.

    The public monitor validates before dispatch. This helper protects direct
    adapter use and prevents boolean aliases, non-finite scalars, and shape
    mismatches from being silently coerced before entering polyglot runtimes.
    """
    phases_array = _validate_vector(phases, name="phases")
    n = int(phases_array.size)
    omegas_array = _validate_vector(omegas, name="omegas")
    if omegas_array.shape != phases_array.shape:
        raise ValueError(
            f"omegas shape {omegas_array.shape} does not match {phases_array.shape}"
        )
    knm_array = _validate_matrix(knm, name="knm", expected_shape=(n, n))
    alpha_value = _validate_finite_float(alpha, name="alpha")
    dt_value = _validate_finite_float(dt, name="dt")
    if dt_value < 0.0:
        raise ValueError(f"dt must be non-negative, got {dt!r}")
    return phases_array, omegas_array, knm_array, alpha_value, dt_value


def validate_entropy_prod_backend_output(value: object) -> float:
    """Validate a direct-backend entropy-production-rate scalar."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("entropy_production_rate must not be a boolean value")
    raw = np.asarray(value)
    if _has_complex_payload(value):
        raise ValueError("entropy_production_rate must be real-valued")
    if _contains_numeric_string_alias(value):
        raise ValueError(
            "entropy_production_rate must not contain numeric-string aliases"
        )
    try:
        result = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "entropy_production_rate must be a finite real scalar"
        ) from exc
    if not np.isfinite(result):
        raise ValueError("entropy_production_rate must be finite")
    if result < -1e-12:
        raise ValueError("entropy_production_rate must be non-negative")
    return max(result, 0.0)
