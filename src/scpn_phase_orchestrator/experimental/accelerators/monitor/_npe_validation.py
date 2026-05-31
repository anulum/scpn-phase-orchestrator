# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct NPE backend validation

"""Shared typed validation for direct NPE backend bridge calls."""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "validate_npe_backend_inputs",
    "validate_npe_backend_output",
    "validate_phase_distance_backend_output",
    "validate_phase_distance_backend_input",
]


def _contains_boolean_alias(raw: np.ndarray) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def validate_phase_distance_backend_input(phases: object) -> FloatArray:
    """Return a finite real one-dimensional phase vector for direct backends."""

    raw = np.asarray(phases)
    if _contains_boolean_alias(raw):
        raise ValueError("phases must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("phases must contain real-valued phase samples")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a one-dimensional float array") from exc
    if array.ndim != 1:
        raise ValueError(f"phases shape {array.shape} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("phases must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_phase_distance_backend_output(
    distances: object,
    *,
    n_phases: int,
) -> FloatArray:
    """Return a validated pairwise circular-distance matrix from a backend."""

    raw = np.asarray(distances)
    if _contains_boolean_alias(raw):
        raise ValueError("phase distance backend output must not contain booleans")
    if np.iscomplexobj(raw):
        raise ValueError("phase distance backend output must contain real values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase distance backend output must be numeric") from exc
    expected = n_phases * n_phases
    if array.size != expected:
        raise ValueError(
            "phase distance backend output size "
            f"{array.size} does not match {expected}"
        )
    matrix = array.reshape((n_phases, n_phases))
    if not np.all(np.isfinite(matrix)):
        raise ValueError("phase distance backend output must be finite")
    tolerance = 1.0e-12
    if np.any(matrix < -tolerance) or np.any(matrix > np.pi + tolerance):
        raise ValueError("phase distance backend output must lie in [0, pi]")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-10):
        raise ValueError("phase distance backend output must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-10):
        raise ValueError("phase distance backend output diagonal must be zero")
    matrix = np.clip(matrix, 0.0, np.pi)
    np.fill_diagonal(matrix, 0.0)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_max_radius(max_radius: object) -> float:
    if isinstance(max_radius, (bool, np.bool_)) or not isinstance(max_radius, Real):
        raise ValueError(
            f"max_radius must be a finite non-negative real, got {max_radius!r}"
        )
    radius = float(max_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError(
            f"max_radius must be finite and non-negative, got {max_radius!r}"
        )
    if radius > np.pi + 1e-12:
        raise ValueError(f"max_radius must not exceed pi, got {max_radius!r}")
    return radius


def validate_npe_backend_inputs(
    phases: object,
    max_radius: object,
) -> tuple[FloatArray, float]:
    """Return validated NPE phase vector and filtration cutoff."""

    return validate_phase_distance_backend_input(phases), _validate_max_radius(
        max_radius
    )


def validate_npe_backend_output(value: object) -> float:
    """Return a validated normalised persistent-entropy backend scalar."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"NPE backend output must be a real scalar, got {value!r}")
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"NPE backend output must be finite, got {value!r}")
    tolerance = 1.0e-12
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError(f"NPE backend output must lie in [0, 1], got {value!r}")
    return min(1.0, max(0.0, score))
