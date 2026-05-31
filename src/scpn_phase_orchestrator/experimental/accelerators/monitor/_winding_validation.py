# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — winding backend boundary validation

"""Shared validation for direct winding accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _contains_boolean_alias(raw: object) -> bool:
    try:
        array = np.asarray(raw, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def validate_winding_backend_inputs(
    phases_flat: object,
    t: object,
    n: object,
) -> tuple[FloatArray, int, int]:
    """Validate flat phase history before optional runtime loading."""

    t_int = _validate_int(t, "t", minimum=2)
    n_int = _validate_int(n, "n", minimum=1)
    raw = np.asarray(phases_flat)
    if _contains_boolean_alias(phases_flat):
        raise ValueError("phases_flat must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("phases_flat must be real-valued")
    try:
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "phases_flat must be a finite one-dimensional float array"
        ) from exc
    if phases.ndim != 1:
        raise ValueError(
            f"phases_flat must be one-dimensional, got shape {phases.shape}"
        )
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases_flat must contain only finite values")
    expected = t_int * n_int
    if phases.size != expected:
        raise ValueError(
            f"phases_flat length {phases.size} does not match t*n={expected}"
        )
    return np.ascontiguousarray(phases, dtype=np.float64), t_int, n_int


def validate_winding_backend_output(
    value: object,
    *,
    t: object,
    n: object,
) -> IntArray:
    """Validate direct-backend winding output before returning it."""

    t_int = _validate_int(t, "t", minimum=2)
    n_int = _validate_int(n, "n", minimum=1)
    if _contains_boolean_alias(value):
        raise ValueError("winding output must not contain boolean values")
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("winding output must be array-like") from exc
    if array.shape != (n_int,):
        raise ValueError(f"winding output shape {array.shape} must be ({n_int},)")
    try:
        numeric = array.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("winding output must be numeric") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError("winding output must contain only finite values")
    if not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError("winding output must contain integer values")
    max_abs_winding = int(np.ceil(max(t_int - 1, 0) / 2.0))
    if np.any(np.abs(numeric) > max_abs_winding):
        raise ValueError("winding output exceeds wrapped-increment bound")
    return np.ascontiguousarray(numeric.astype(np.int64), dtype=np.int64)
