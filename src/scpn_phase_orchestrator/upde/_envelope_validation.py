# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - envelope backend validation contracts

"""Shared validation for direct RMS-envelope accelerator backends."""

from __future__ import annotations

from numbers import Integral
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = [
    "validate_envelope_modulation_input",
    "validate_envelope_modulation_output",
    "validate_extract_envelope_input",
    "validate_extract_envelope_output",
]

FloatArray: TypeAlias = NDArray[np.float64]


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


def _as_finite_vector(value: Any, *, name: str, allow_empty: bool) -> FloatArray:
    """Return ``value`` as a validated finite vector, else raise."""
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be real-valued, not boolean")
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued, not complex")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    out = np.ascontiguousarray(array.ravel(), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not allow_empty and out.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_positive_window(value: Any) -> int:
    """Return ``value`` as a validated positive window length, else raise."""
    if _is_numeric_string_alias(value):
        raise ValueError("window must not be a numeric-string alias")
    if contains_boolean_alias(value):
        raise TypeError("window must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise TypeError("window must be an integer")
    out = int(value)
    if out < 1:
        raise ValueError("window must be >= 1")
    return out


def validate_extract_envelope_input(
    amps: FloatArray,
    window: int,
) -> tuple[FloatArray, int]:
    """Validate and normalise direct RMS-envelope extraction inputs."""
    return (
        _as_finite_vector(amps, name="amps", allow_empty=True),
        _as_positive_window(window),
    )


def validate_extract_envelope_output(value: Any, *, n: int) -> FloatArray:
    """Validate a direct backend RMS-envelope vector."""
    out = _as_finite_vector(value, name="envelope", allow_empty=True)
    if out.size != n:
        raise ValueError(f"envelope must contain {n} values")
    if np.any(out < 0.0):
        raise ValueError("envelope values must be non-negative")
    return out


def validate_envelope_modulation_input(env: FloatArray) -> FloatArray:
    """Validate and normalise direct modulation-depth inputs."""
    return _as_finite_vector(env, name="env", allow_empty=True)


def validate_envelope_modulation_output(value: Any) -> float:
    """Validate a direct backend modulation-depth scalar."""
    if _contains_numeric_string_alias(value):
        raise ValueError("modulation depth must not contain numeric-string aliases")
    if contains_boolean_alias(value):
        raise TypeError("modulation depth must be real-valued, not boolean")
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError("modulation depth must be real-valued, not complex")
    if array.shape not in ((), (1,)):
        raise ValueError("modulation depth must be a scalar")
    out = float(array.reshape(-1)[0])
    if not np.isfinite(out):
        raise ValueError("modulation depth must be finite")
    if out < 0.0 or out > 1.0 + 1e-12:
        raise ValueError("modulation depth must lie in [0, 1]")
    return min(out, 1.0)
