# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ITPC backend boundary validation

"""Shared validation for direct ITPC accelerator calls."""

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
    return any(isinstance(value, (bool, np.bool_)) for value in array.flat)


def _validate_int(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_phase_buffer(
    phases_flat: object,
    n_trials: int,
    n_tp: int,
) -> FloatArray:
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
    expected = n_trials * n_tp
    if expected == 0 and phases.size == 0:
        return np.ascontiguousarray(phases.ravel(), dtype=np.float64)
    if phases.ndim != 1:
        raise ValueError(
            f"phases_flat must be one-dimensional, got shape {phases.shape}"
        )
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases_flat must contain only finite values")
    if phases.size != expected:
        raise ValueError(
            f"phases_flat length {phases.size} does not match n_trials*n_tp={expected}"
        )
    return np.ascontiguousarray(phases, dtype=np.float64)


def validate_compute_itpc_backend_inputs(
    phases_flat: object,
    n_trials: object,
    n_tp: object,
) -> tuple[FloatArray, int, int]:
    """Validate direct ITPC inputs before optional runtime loading."""

    trials = _validate_int(n_trials, "n_trials", minimum=0)
    timepoints = _validate_int(n_tp, "n_tp", minimum=0)
    phases = _validate_phase_buffer(phases_flat, trials, timepoints)
    return phases, trials, timepoints


def validate_itpc_persistence_backend_inputs(
    phases_flat: object,
    n_trials: object,
    n_tp: object,
    pause_indices: object,
) -> tuple[FloatArray, int, int, IntArray]:
    """Validate direct ITPC persistence inputs before optional runtime loading."""

    phases, trials, timepoints = validate_compute_itpc_backend_inputs(
        phases_flat,
        n_trials,
        n_tp,
    )
    raw_idx = np.asarray(pause_indices, dtype=object)
    if raw_idx.ndim != 1:
        raise ValueError("pause_indices must be a one-dimensional integer array")
    flat_idx = raw_idx.ravel()
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool)
        for value in flat_idx
    ):
        raise ValueError("pause_indices must contain only integer indices")
    indices = np.ascontiguousarray(flat_idx, dtype=np.int64)
    return phases, trials, timepoints, indices
