# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - PAC backend validation contracts

"""Shared validation for direct phase-amplitude-coupling backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = [
    "validate_modulation_index_inputs",
    "validate_modulation_index_output",
    "validate_pac_matrix_inputs",
    "validate_pac_matrix_output",
]

FloatArray: TypeAlias = NDArray[np.float64]

_UNIT_INTERVAL_TOLERANCE = 1e-12


def _as_int(value: Any, *, name: str, minimum: int) -> int:
    """Return ``value`` as a validated integer, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return out


def _as_finite_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite vector, else raise."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be real-valued, not boolean")
    array = np.asarray(value)
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


def _as_unit_interval_vector(value: Any, *, name: str, expected: int) -> FloatArray:
    """Return ``value`` as a validated vector in [0, 1], else raise."""
    out = _as_finite_vector(value, name=name)
    if out.size != expected:
        raise ValueError(f"{name} must contain {expected} values")
    if np.any(out < -_UNIT_INTERVAL_TOLERANCE) or np.any(
        out > 1.0 + _UNIT_INTERVAL_TOLERANCE
    ):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0), dtype=np.float64)


def _validate_non_negative_amplitude(values: FloatArray, *, name: str) -> None:
    """Return ``value`` as a validated non-negative amplitude, else raise."""
    if np.any(values < 0.0):
        raise ValueError(f"{name} must contain non-negative amplitudes")


def validate_modulation_index_inputs(
    theta_low: FloatArray,
    amp_high: FloatArray,
    n_bins: int,
) -> tuple[FloatArray, FloatArray, int]:
    """Validate and align direct modulation-index backend inputs."""
    bins = _as_int(n_bins, name="n_bins", minimum=2)
    theta = _as_finite_vector(theta_low, name="theta_low")
    amp = _as_finite_vector(amp_high, name="amp_high")
    _validate_non_negative_amplitude(amp, name="amp_high")
    sample_count = min(theta.size, amp.size)
    return (
        np.ascontiguousarray(theta[:sample_count], dtype=np.float64),
        np.ascontiguousarray(amp[:sample_count], dtype=np.float64),
        bins,
    )


def validate_modulation_index_output(value: Any) -> float:
    """Validate direct backend Tort modulation-index output."""
    if contains_boolean_alias(value):
        raise TypeError("modulation index must be a real scalar, not boolean")
    if not isinstance(value, Real):
        raise TypeError("modulation index must be a real scalar")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("modulation index must be finite")
    if out < -_UNIT_INTERVAL_TOLERANCE or out > 1.0 + _UNIT_INTERVAL_TOLERANCE:
        raise ValueError("modulation index must lie in [0, 1]")
    return float(min(max(out, 0.0), 1.0))


def validate_pac_matrix_inputs(
    phases_flat: FloatArray,
    amplitudes_flat: FloatArray,
    t: int,
    n: int,
    n_bins: int,
) -> tuple[FloatArray, FloatArray, int, int, int]:
    """Validate direct pairwise PAC-matrix backend inputs."""
    t_i = _as_int(t, name="t", minimum=1)
    n_i = _as_int(n, name="n", minimum=1)
    bins = _as_int(n_bins, name="n_bins", minimum=2)
    expected = t_i * n_i
    phases = _as_finite_vector(phases_flat, name="phases_flat")
    if phases.size != expected:
        raise ValueError(f"phases_flat must contain {expected} values")
    amplitudes = _as_finite_vector(amplitudes_flat, name="amplitudes_flat")
    if amplitudes.size != expected:
        raise ValueError(f"amplitudes_flat must contain {expected} values")
    _validate_non_negative_amplitude(amplitudes, name="amplitudes_flat")
    return phases, amplitudes, t_i, n_i, bins


def validate_pac_matrix_output(value: Any, *, n: int) -> FloatArray:
    """Validate direct backend pairwise PAC-matrix output."""
    n_i = _as_int(n, name="n", minimum=1)
    return _as_unit_interval_vector(value, name="PAC matrix", expected=n_i * n_i)
