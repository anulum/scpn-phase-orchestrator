# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare backend boundary validation

"""Shared validation for direct Poincare accelerator calls."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "validate_phase_poincare_backend_inputs",
    "validate_poincare_backend_outputs",
    "validate_poincare_section_backend_inputs",
]


def _contains_boolean_alias(raw: object) -> bool:
    try:
        array = np.asarray(raw, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _contains_complex_alias(raw: object) -> bool:
    try:
        array = np.asarray(raw, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in array.flat)


def _validate_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real value")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_float_vector(value: object, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_complex_alias(value):
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


def validate_poincare_section_backend_inputs(
    traj_flat: object,
    t: object,
    d: object,
    normal: object,
    offset: object,
    direction_id: object,
) -> tuple[FloatArray, int, int, FloatArray, float, int]:
    """Validate direct section-backend inputs before optional runtime loading."""

    t_int = _validate_int(t, "t", minimum=1)
    d_int = _validate_int(d, "d", minimum=1)
    trajectory = _validate_float_vector(traj_flat, "traj_flat")
    expected = t_int * d_int
    if trajectory.size != expected:
        raise ValueError(
            f"traj_flat length {trajectory.size} does not match t*d={expected}"
        )
    normal_vec = _validate_float_vector(normal, "normal")
    if normal_vec.shape != (d_int,):
        raise ValueError(f"normal shape {normal_vec.shape} does not match ({d_int},)")
    offset_float = _validate_finite_real(offset, "offset")
    direction_int = _validate_int(direction_id, "direction_id", minimum=0)
    if direction_int not in {0, 1, 2}:
        raise ValueError("direction_id must be 0, 1, or 2")
    return trajectory, t_int, d_int, normal_vec, offset_float, direction_int


def validate_phase_poincare_backend_inputs(
    phases_flat: object,
    t: object,
    n: object,
    oscillator_idx: object,
    section_phase: object,
) -> tuple[FloatArray, int, int, int, float]:
    """Validate direct phase-backend inputs before optional runtime loading."""

    t_int = _validate_int(t, "t", minimum=1)
    n_int = _validate_int(n, "n", minimum=1)
    phases = _validate_float_vector(phases_flat, "phases_flat")
    expected = t_int * n_int
    if phases.size != expected:
        raise ValueError(
            f"phases_flat length {phases.size} does not match t*n={expected}"
        )
    idx = _validate_int(oscillator_idx, "oscillator_idx", minimum=0)
    if idx >= n_int:
        raise ValueError(f"oscillator_idx must be in [0, {n_int}), got {idx}")
    section = _validate_finite_real(section_phase, "section_phase")
    return phases, t_int, n_int, idx, section


def validate_poincare_backend_outputs(
    crossings_flat: object,
    times: object,
    n_cr: object,
    *,
    t: int,
    dim: int,
) -> tuple[FloatArray, FloatArray, int]:
    """Validate direct Poincare backend crossing payloads before return."""

    t_int = _validate_int(t, "t", minimum=1)
    dim_int = _validate_int(dim, "dim", minimum=1)
    count = _validate_int(n_cr, "n_cr", minimum=0)
    max_crossings = max(t_int - 1, 0)
    if count > max_crossings:
        raise ValueError(
            f"n_cr must not exceed the {max_crossings} available intervals"
        )

    crossings = _validate_float_vector(crossings_flat, "crossings_flat")
    expected_crossings = t_int * dim_int
    if crossings.size != expected_crossings:
        raise ValueError(
            "crossings_flat length "
            f"{crossings.size} does not match t*dim={expected_crossings}"
        )
    crossing_times = _validate_float_vector(times, "times")
    if crossing_times.size != t_int:
        raise ValueError(f"times length {crossing_times.size} does not match t={t_int}")
    active_times = crossing_times[:count]
    if active_times.size:
        tolerance = 1.0e-12
        if np.any(active_times < -tolerance) or np.any(
            active_times > max_crossings + tolerance
        ):
            raise ValueError("active crossing times must lie within sampled intervals")
        if active_times.size > 1 and np.any(np.diff(active_times) <= tolerance):
            raise ValueError("active crossing times must be strictly increasing")
    return crossings, crossing_times, count
