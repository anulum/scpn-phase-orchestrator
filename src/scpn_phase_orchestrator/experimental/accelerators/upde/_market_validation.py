# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - market backend validation contracts

"""Shared validation for direct financial-market accelerator backends."""

from __future__ import annotations

from numbers import Integral
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "validate_market_order_inputs",
    "validate_market_order_output",
    "validate_market_plv_inputs",
    "validate_market_plv_output",
]

FloatArray: TypeAlias = NDArray[np.float64]

_UNIT_INTERVAL_TOLERANCE = 1e-12
_PLV_MATRIX_TOLERANCE = 1e-9


def _as_positive_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_real_finite_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real vector, else raise."""
    array = np.asarray(value)
    if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.bool_):
        raise TypeError(f"{name} must be real-valued, not boolean")
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued, not complex")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    out = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _validate_flat_phase_payload(value: Any, *, t: int, n: int) -> FloatArray:
    """Return ``value`` as a validated flattened phase payload, else raise."""
    phases = _as_real_finite_vector(value, name="phases_flat")
    expected = t * n
    if phases.size != expected:
        raise ValueError(f"phases_flat must contain {expected} values")
    return phases


def validate_market_order_inputs(
    phases_flat: FloatArray,
    t: int,
    n: int,
) -> tuple[FloatArray, int, int]:
    """Validate direct market order-parameter backend inputs."""
    t_i = _as_positive_int(t, name="t")
    n_i = _as_positive_int(n, name="n")
    return _validate_flat_phase_payload(phases_flat, t=t_i, n=n_i), t_i, n_i


def validate_market_plv_inputs(
    phases_flat: FloatArray,
    t: int,
    n: int,
    window: int,
) -> tuple[FloatArray, int, int, int]:
    """Validate direct market phase-locking backend inputs."""
    t_i = _as_positive_int(t, name="t")
    n_i = _as_positive_int(n, name="n")
    window_i = _as_positive_int(window, name="window")
    if window_i > t_i:
        raise ValueError("window must not exceed t")
    return (
        _validate_flat_phase_payload(phases_flat, t=t_i, n=n_i),
        t_i,
        n_i,
        window_i,
    )


def validate_market_order_output(value: Any, *, t: int) -> FloatArray:
    """Validate direct backend ``R(t)`` output before publication."""
    t_i = _as_positive_int(t, name="t")
    out = _as_real_finite_vector(value, name="order parameter")
    if out.size != t_i:
        raise ValueError(f"order parameter must contain {t_i} values")
    if np.any(out < -_UNIT_INTERVAL_TOLERANCE) or np.any(
        out > 1.0 + _UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("order parameter values must lie in [0, 1]")
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0), dtype=np.float64)


def validate_market_plv_output(
    value: Any,
    *,
    t: int,
    n: int,
    window: int,
) -> FloatArray:
    """Validate direct backend rolling PLV matrices before publication."""
    t_i = _as_positive_int(t, name="t")
    n_i = _as_positive_int(n, name="n")
    window_i = _as_positive_int(window, name="window")
    if window_i > t_i:
        raise ValueError("window must not exceed t")
    n_windows = t_i - window_i + 1
    expected = n_windows * n_i * n_i
    out = _as_real_finite_vector(value, name="phase-locking value")
    if out.size != expected:
        raise ValueError(f"phase-locking value must contain {expected} values")
    if np.any(out < -_UNIT_INTERVAL_TOLERANCE) or np.any(
        out > 1.0 + _UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError("phase-locking values must lie in [0, 1]")
    matrix = out.reshape(n_windows, n_i, n_i)
    if not np.allclose(
        np.diagonal(matrix, axis1=1, axis2=2),
        1.0,
        atol=_PLV_MATRIX_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("phase-locking value diagonals must be one")
    if not np.allclose(
        matrix,
        np.swapaxes(matrix, 1, 2),
        atol=_PLV_MATRIX_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("phase-locking matrices must be symmetric")
    clipped = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(clipped, dtype=np.float64)
