# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spatial modulator backend validation

"""Shared validation for direct spatial-modulator accelerator calls."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["validate_spatial_modulator_inputs", "validate_spatial_modulator_output"]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype != object:
            return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        return True
    if isinstance(value, np.ndarray) and raw.dtype != object:
        return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.ravel())


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive")
    return parsed


def _validate_scalar(value: object, *, name: str, positive: bool = False) -> float:
    """Return ``value`` as a validated finite scalar, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if positive and parsed <= 0.0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _validate_flat(value: object, *, name: str, expected: int) -> FloatArray:
    """Return ``value`` as a validated flattened array, else raise."""
    if _contains_boolean_alias(value) or _contains_complex_alias(value):
        raise ValueError(f"{name} must be finite real-valued")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite real-valued") from exc
    if arr.shape != (expected,):
        raise ValueError(f"{name} length {arr.size} does not match {expected}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def validate_spatial_modulator_inputs(
    k_nm_flat: object,
    positions_flat: object,
    n: object,
    dim: object,
    k_base: object,
    decay_form_code: object,
    decay_exponent: object,
    decay_length_scale: object,
    epsilon: object,
) -> tuple[FloatArray, FloatArray, int, int, float, int, float, float, float]:
    """Validate direct backend inputs before optional runtime loading."""
    n_int = _validate_positive_int(n, name="n")
    dim_int = _validate_positive_int(dim, name="dim")
    if isinstance(decay_form_code, (bool, np.bool_)) or not isinstance(
        decay_form_code,
        Integral,
    ):
        raise ValueError("decay_form_code must be 0, 1, 2, or 3")
    form_int = int(decay_form_code)
    if form_int not in {0, 1, 2, 3}:
        raise ValueError("decay_form_code must be 0, 1, 2, or 3")
    k = _validate_flat(k_nm_flat, name="k_nm_flat", expected=n_int * n_int)
    p = _validate_flat(positions_flat, name="positions_flat", expected=n_int * dim_int)
    matrix = k.reshape(n_int, n_int)
    if np.max(np.abs(np.diag(matrix))) > 1.0e-12:
        raise ValueError("k_nm_flat diagonal must be zero")
    return (
        k,
        p,
        n_int,
        dim_int,
        _validate_scalar(k_base, name="k_base"),
        form_int,
        _validate_scalar(decay_exponent, name="decay_exponent", positive=True),
        _validate_scalar(decay_length_scale, name="decay_length_scale", positive=True),
        _validate_scalar(epsilon, name="epsilon", positive=True),
    )


def validate_spatial_modulator_output(value: object, *, n: int) -> FloatArray:
    """Validate direct backend output cardinality and physics invariants."""
    if _contains_boolean_alias(value) or _contains_complex_alias(value):
        raise ValueError("spatial modulator output must be finite real-valued")
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("spatial modulator output must be finite real-valued") from exc
    if out.shape != (n * n,):
        raise ValueError(
            f"spatial modulator output length {out.size} does not match {n * n}"
        )
    if not np.all(np.isfinite(out)):
        raise ValueError("spatial modulator output must contain only finite values")
    matrix = out.reshape(n, n)
    if np.max(np.abs(np.diag(matrix))) > 1.0e-10:
        raise ValueError("spatial modulator output diagonal must be zero")
    return np.ascontiguousarray(out, dtype=np.float64)
