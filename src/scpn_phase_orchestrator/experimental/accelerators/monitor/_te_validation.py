# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy backend validation

"""Shared validation for direct transfer-entropy polyglot backend calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_phase_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real-valued phase vector")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite phase vector") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_phase_series_flat(
    value: object,
    *,
    n_osc: int,
    n_time: int,
) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError("phase_series must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("phase_series must be a finite real-valued phase series")
    try:
        array = raw.astype(np.float64, copy=True).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError("phase_series must be a finite phase series") from exc
    expected_size = n_osc * n_time
    if array.size != expected_size:
        raise ValueError(
            f"phase_series size {array.size} does not match {expected_size}",
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("phase_series must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    scalar = int(value)
    if scalar < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {scalar}")
    return scalar


def validate_phase_te_backend_inputs(
    source: object,
    target: object,
    n_bins: object,
) -> tuple[FloatArray, FloatArray, int]:
    """Validate direct pairwise transfer-entropy backend arguments."""
    source_values = _validate_phase_vector(source, name="source")
    target_values = _validate_phase_vector(target, name="target")
    return (
        source_values,
        target_values,
        _validate_int_at_least(n_bins, name="n_bins", minimum=2),
    )


def validate_te_matrix_backend_inputs(
    phase_series: object,
    n_osc: object,
    n_time: object,
    n_bins: object,
) -> tuple[FloatArray, int, int, int]:
    """Validate direct transfer-entropy matrix backend arguments."""
    oscillator_count = _validate_int_at_least(n_osc, name="n_osc", minimum=1)
    timestep_count = _validate_int_at_least(n_time, name="n_time", minimum=1)
    return (
        _validate_phase_series_flat(
            phase_series,
            n_osc=oscillator_count,
            n_time=timestep_count,
        ),
        oscillator_count,
        timestep_count,
        _validate_int_at_least(n_bins, name="n_bins", minimum=2),
    )
