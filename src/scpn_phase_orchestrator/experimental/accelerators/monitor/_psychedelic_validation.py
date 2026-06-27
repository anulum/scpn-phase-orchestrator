# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — psychedelic backend boundary validation

"""Shared validation for direct psychedelic accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "validate_psychedelic_backend_inputs",
    "validate_psychedelic_entropy_backend_output",
]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _has_complex_payload(value: object) -> bool:
    """Return whether the value carries a complex-number payload."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return _contains_complex_alias(value)
    return bool(np.iscomplexobj(raw) or _contains_complex_alias(value))


def _validate_phase_vector(value: object) -> FloatArray:
    """Return the phases as a validated 1-D finite array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError("phases must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError("phases must be real-valued")
    try:
        raw = np.asarray(value)
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite one-dimensional float array") from exc
    if phases.ndim != 1:
        raise ValueError(f"phases must be one-dimensional, got shape {phases.shape}")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases must contain only finite values")
    return np.ascontiguousarray(phases, dtype=np.float64)


def _validate_n_bins(value: object) -> int:
    """Return ``n_bins`` as an integer at least 2, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    n_bins = int(value)
    if n_bins < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return n_bins


def validate_psychedelic_backend_inputs(
    phases: object,
    n_bins: object,
) -> tuple[FloatArray, int]:
    """Validate entropy inputs before optional runtime loading."""
    return _validate_phase_vector(phases), _validate_n_bins(n_bins)


def validate_psychedelic_entropy_backend_output(
    value: object,
    n_bins: int,
) -> float:
    """Validate direct backend circular-entropy outputs."""
    if _contains_boolean_alias(value):
        raise ValueError("entropy backend output must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError("entropy backend output must be real-valued")
    try:
        raw = np.asarray(value)
        entropy = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("entropy backend output must be numeric") from exc
    if entropy.shape != ():
        raise ValueError("entropy backend output must be scalar")
    scalar = float(entropy)
    if not np.isfinite(scalar):
        raise ValueError("entropy backend output must be finite")
    tolerance = 1e-12
    upper = float(np.log(float(n_bins)))
    if scalar < -tolerance or scalar > upper + tolerance:
        raise ValueError("entropy backend output must lie in [0, log(n_bins)]")
    return min(upper, max(0.0, scalar))
