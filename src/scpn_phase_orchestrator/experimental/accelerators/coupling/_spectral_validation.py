# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spectral backend boundary validation

"""Shared validation for direct spectral accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["validate_spectral_backend_inputs"]


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


def _validate_knm_flat(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("knm_flat must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("knm_flat must be real-valued")
    try:
        knm_flat = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "knm_flat must be a finite one-dimensional float array"
        ) from exc
    if knm_flat.ndim != 1:
        raise ValueError(
            f"knm_flat must be one-dimensional, got shape {knm_flat.shape}"
        )
    if not np.all(np.isfinite(knm_flat)):
        raise ValueError("knm_flat must contain only finite values")
    return np.ascontiguousarray(knm_flat, dtype=np.float64)


def validate_spectral_backend_inputs(
    knm_flat: object,
    n: object,
) -> tuple[FloatArray, int]:
    """Validate direct spectral inputs before optional runtime loading."""

    n_int = _validate_n(n)
    k = _validate_knm_flat(knm_flat)
    expected = n_int * n_int
    if k.size != expected:
        raise ValueError(f"knm_flat length {k.size} does not match n*n={expected}")
    return k, n_int
