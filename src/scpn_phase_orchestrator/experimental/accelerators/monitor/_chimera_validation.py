# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — chimera backend boundary validation

"""Shared validation for direct chimera accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(raw: object) -> bool:
    try:
        array = np.asarray(raw, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_n(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("n must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"n must be non-negative, got {result}")
    return result


def _validate_float_vector(value: object, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite one-dimensional float array"
        ) from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_chimera_backend_inputs(
    phases: object,
    knm_flat: object,
    n: object,
) -> tuple[FloatArray, FloatArray, int]:
    """Validate direct local-order inputs before optional runtime loading."""

    n_int = _validate_n(n)
    phases_vec = _validate_float_vector(phases, "phases")
    if phases_vec.size != n_int:
        raise ValueError(f"phases length {phases_vec.size} does not match n={n_int}")
    knm_vec = _validate_float_vector(knm_flat, "knm_flat")
    expected = n_int * n_int
    if knm_vec.size != expected:
        raise ValueError(
            f"knm_flat length {knm_vec.size} does not match n*n={expected}"
        )
    if n_int:
        diagonal = np.diag(knm_vec.reshape(n_int, n_int))
        if not np.allclose(diagonal, 0.0, rtol=0.0, atol=1e-15):
            raise ValueError("knm_flat self-coupling diagonal must be zero")
    return phases_vec, knm_vec, n_int
