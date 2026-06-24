# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Twin-confidence backend validation

"""Shared validation for direct twin-confidence polyglot backend calls.

Each non-Python backend bridge validates its arguments through
:func:`validate_twin_divergence_backend_inputs` and its result through
:func:`validate_twin_divergence_backend_output`, so every backend enforces the
same contract as the NumPy reference in
``scpn_phase_orchestrator.monitor.twin_confidence``.
"""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "validate_twin_divergence_backend_inputs",
    "validate_twin_divergence_backend_output",
]

FloatArray: TypeAlias = NDArray[np.float64]

_JS_MAX: float = float(np.log(2.0))
_PARITY_TOL: float = 1e-9


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):  # pragma: no cover - numpy always coerces
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated 1-D finite array, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real-valued vector")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_order_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated order-parameter vector, else raise."""
    array = _validate_vector(value, name=name)
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} order-parameter values must lie in [0, 1]")
    return array


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer, got {number}")
    return number


def validate_twin_divergence_backend_inputs(
    model_phases: object,
    observed_phases: object,
    model_order: object,
    observed_order: object,
    n: object,
    w: object,
    n_bins: object,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, int, int, int]:
    """Validate and normalise direct twin-divergence backend arguments.

    Parameters
    ----------
    model_phases, observed_phases : object
        Model and observed phase vectors (radians) of equal length ``n``.
    model_order, observed_order : object
        Model and observed order-parameter windows in ``[0, 1]`` of length ``w``.
    n, w, n_bins : object
        Phase count, order-window length, and histogram bin count.

    Returns
    -------
    tuple[FloatArray, FloatArray, FloatArray, FloatArray, int, int, int]
        The validated arrays followed by ``(n, w, n_bins)``.

    Raises
    ------
    ValueError
        If any array is non-finite, mis-shaped, out of range, or the integer
        arguments are not positive or disagree with the array lengths.
    """
    model_phases64 = _validate_vector(model_phases, name="model_phases")
    observed_phases64 = _validate_vector(observed_phases, name="observed_phases")
    model_order64 = _validate_order_vector(model_order, name="model_order")
    observed_order64 = _validate_order_vector(observed_order, name="observed_order")
    n_int = _validate_positive_int(n, name="n")
    w_int = _validate_positive_int(w, name="w")
    n_bins_int = _validate_positive_int(n_bins, name="n_bins")
    if model_phases64.size != n_int or observed_phases64.size != n_int:
        raise ValueError("phase vector lengths must equal n")
    if model_order64.size != w_int or observed_order64.size != w_int:
        raise ValueError("order vector lengths must equal w")
    return (
        model_phases64,
        observed_phases64,
        model_order64,
        observed_order64,
        n_int,
        w_int,
        n_bins_int,
    )


def validate_twin_divergence_backend_output(value: object) -> FloatArray:
    """Validate a direct backend ``(js, w1)`` result.

    Parameters
    ----------
    value : object
        The backend output, coercible to a two-element float array.

    Returns
    -------
    FloatArray
        The validated ``[phase_js_divergence, order_wasserstein]`` array.

    Raises
    ------
    ValueError
        If the result is not a finite two-element pair in the valid ranges
        ``[0, ln 2]`` and ``[0, 1]`` (within parity tolerance).
    """
    array = np.asarray(value, dtype=np.float64).ravel()
    if array.shape != (2,):
        raise ValueError(f"backend output shape {array.shape} is not (2,)")
    if not np.all(np.isfinite(array)):
        raise ValueError("backend output must be finite")
    js = float(array[0])
    w1 = float(array[1])
    if js < -_PARITY_TOL or js > _JS_MAX + _PARITY_TOL:
        raise ValueError(f"Jensen–Shannon divergence {js} outside [0, ln 2]")
    if w1 < -_PARITY_TOL or w1 > 1.0 + _PARITY_TOL:
        raise ValueError(f"Wasserstein-1 distance {w1} outside [0, 1]")
    return np.ascontiguousarray(array, dtype=np.float64)
