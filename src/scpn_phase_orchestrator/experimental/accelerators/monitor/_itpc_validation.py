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


def expected_compute_itpc_backend_output(
    phases_flat: FloatArray,
    n_trials: int,
    n_tp: int,
) -> FloatArray:
    """Return the exact NumPy ITPC reference for validated backend payloads."""

    if n_tp == 0:
        return np.zeros(0, dtype=np.float64)
    if n_trials == 0:
        return np.zeros(n_tp, dtype=np.float64)
    phases = np.asarray(phases_flat, dtype=np.float64).reshape(n_trials, n_tp)
    return np.ascontiguousarray(
        np.abs(np.mean(np.exp(1j * phases), axis=0)),
        dtype=np.float64,
    )


def expected_itpc_persistence_backend_output(
    phases_flat: FloatArray,
    n_trials: int,
    n_tp: int,
    pause_indices: IntArray,
) -> float:
    """Return the exact NumPy persistence reference for validated payloads."""

    if n_trials == 0 or n_tp == 0 or pause_indices.size == 0:
        return 0.0
    itpc = expected_compute_itpc_backend_output(phases_flat, n_trials, n_tp)
    valid = pause_indices[(pause_indices >= 0) & (pause_indices < itpc.size)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(itpc[valid]))


def validate_compute_itpc_backend_output(
    value: object,
    n_tp: int,
    *,
    expected: FloatArray | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    """Validate direct ITPC vectors returned by optional backends."""

    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError("ITPC backend output must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("ITPC backend output must be real-valued")
    try:
        itpc = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("ITPC backend output must be numeric") from exc
    if itpc.shape != (n_tp,):
        raise ValueError(
            f"ITPC backend output shape {itpc.shape} does not match ({n_tp},)"
        )
    if not np.all(np.isfinite(itpc)):
        raise ValueError("ITPC backend output must contain only finite values")
    tolerance = 1e-12
    if np.any(itpc < -tolerance) or np.any(itpc > 1.0 + tolerance):
        raise ValueError("ITPC backend output must lie in [0, 1]")
    clipped = np.ascontiguousarray(np.clip(itpc, 0.0, 1.0), dtype=np.float64)
    if expected is not None:
        reference = np.asarray(expected, dtype=np.float64)
        if reference.shape != clipped.shape:
            raise ValueError("ITPC exact reference shape must match backend output")
        if not np.allclose(clipped, reference, rtol=0.0, atol=atol):
            raise ValueError("ITPC backend output diverged from exact reference")
    return clipped


def validate_itpc_persistence_backend_output(
    value: object,
    *,
    expected: float | None = None,
    atol: float = 1e-12,
) -> float:
    """Validate direct ITPC persistence scalars returned by optional backends."""

    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError("ITPC persistence backend output must not contain booleans")
    if np.iscomplexobj(raw):
        raise ValueError("ITPC persistence backend output must be real-valued")
    try:
        scalar = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("ITPC persistence backend output must be numeric") from exc
    if scalar.shape != ():
        raise ValueError("ITPC persistence backend output must be scalar")
    score = float(scalar)
    if not np.isfinite(score):
        raise ValueError("ITPC persistence backend output must be finite")
    tolerance = 1e-12
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError("ITPC persistence backend output must lie in [0, 1]")
    clipped = min(1.0, max(0.0, score))
    if expected is not None and not np.isclose(
        clipped,
        float(expected),
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(
            "ITPC persistence backend output diverged from exact reference"
        )
    return clipped


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
