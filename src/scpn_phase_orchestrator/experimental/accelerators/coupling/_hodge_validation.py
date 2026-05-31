# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge backend boundary validation

"""Shared validation for direct Hodge accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["validate_hodge_backend_inputs"]


def _contains_boolean_alias(value: object) -> bool:
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _validate_n(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("n must be a non-negative integer")
    n_int = int(value)
    if n_int < 0:
        raise ValueError(f"n must be non-negative, got {n_int}")
    return n_int


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        vector = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite one-dimensional float array"
        ) from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector, dtype=np.float64)


def validate_hodge_backend_inputs(
    knm_flat: object,
    phases: object,
    n: object,
) -> tuple[FloatArray, FloatArray, int]:
    """Validate direct Hodge inputs before optional runtime loading."""

    n_int = _validate_n(n)
    k = _validate_float_vector(knm_flat, name="knm_flat")
    expected_k = n_int * n_int
    if k.size != expected_k:
        raise ValueError(f"knm_flat length {k.size} does not match n*n={expected_k}")
    p = _validate_float_vector(phases, name="phases")
    if p.size != n_int:
        raise ValueError(f"phases length {p.size} does not match n={n_int}")
    return k, p, n_int
