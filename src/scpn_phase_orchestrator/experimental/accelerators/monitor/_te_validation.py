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

__all__ = [
    "FloatArray",
    "expected_phase_te_backend_output",
    "expected_te_matrix_backend_output",
    "validate_phase_te_backend_inputs",
    "validate_te_backend_output",
    "validate_te_matrix_backend_inputs",
    "validate_te_matrix_backend_output",
]

TWO_PI = 2.0 * np.pi


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


def _conditional_entropy(
    target: NDArray[np.int64],
    condition: NDArray[np.int64],
    n_cond_bins: int,
) -> float:
    n = len(target)
    entropy = 0.0
    for c in range(n_cond_bins):
        mask = condition == c
        count = int(np.sum(mask))
        if count < 2:
            continue
        vals = target[mask]
        _, counts = np.unique(vals, return_counts=True)
        probs = counts / count
        entropy -= (count / n) * float(np.sum(probs * np.log(probs + 1e-30)))
    return entropy


def expected_phase_te_backend_output(
    source: FloatArray,
    target: FloatArray,
    n_bins: int,
) -> float:
    """Return the exact NumPy transfer-entropy reference for validated payloads."""

    n_samples = min(int(source.size), int(target.size))
    if n_samples < 3:
        return 0.0
    source_values = np.asarray(source[:n_samples], dtype=np.float64)
    target_values = np.asarray(target[:n_samples], dtype=np.float64)
    n = n_samples - 1
    bins = np.linspace(0.0, TWO_PI, n_bins + 1)
    src_binned = np.clip(
        np.digitize(source_values[:n] % TWO_PI, bins) - 1,
        0,
        n_bins - 1,
    ).astype(np.int64)
    tgt_binned = np.clip(
        np.digitize(target_values[:n] % TWO_PI, bins) - 1,
        0,
        n_bins - 1,
    ).astype(np.int64)
    tgt_next = np.clip(
        np.digitize(target_values[1 : n + 1] % TWO_PI, bins) - 1,
        0,
        n_bins - 1,
    ).astype(np.int64)
    h_y_yt = _conditional_entropy(tgt_next, tgt_binned, n_bins)
    joint_cond = (tgt_binned * n_bins + src_binned).astype(np.int64)
    h_y_yt_x = _conditional_entropy(tgt_next, joint_cond, n_bins * n_bins)
    return max(h_y_yt - h_y_yt_x, 0.0)


def expected_te_matrix_backend_output(
    phase_series: FloatArray,
    n_osc: int,
    n_time: int,
    n_bins: int,
) -> FloatArray:
    """Return the exact NumPy transfer-entropy matrix reference."""

    series = np.asarray(phase_series, dtype=np.float64).reshape(n_osc, n_time)
    matrix = np.zeros((n_osc, n_osc), dtype=np.float64)
    for i in range(n_osc):
        for j in range(n_osc):
            if i != j:
                matrix[i, j] = expected_phase_te_backend_output(
                    series[i],
                    series[j],
                    n_bins,
                )
    return np.ascontiguousarray(matrix, dtype=np.float64)


def validate_te_backend_output(
    value: object,
    *,
    n_bins: int,
    expected: float | None = None,
    atol: float = 1e-12,
) -> float:
    """Validate a direct pairwise transfer-entropy backend scalar."""

    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError("transfer entropy backend output must not be boolean")
    if np.iscomplexobj(raw):
        raise ValueError("transfer entropy backend output must be real")
    try:
        scalar = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("transfer entropy backend output must be numeric") from exc
    if scalar.shape != ():
        raise ValueError("transfer entropy backend output must be scalar")
    result = float(scalar)
    max_entropy = float(np.log(n_bins))
    tolerance = 1.0e-12
    if not np.isfinite(result) or result < -tolerance:
        raise ValueError("transfer entropy backend output must be non-negative")
    if result > max_entropy + tolerance:
        raise ValueError("transfer entropy backend output must not exceed log(n_bins)")
    result = max(result, 0.0)
    if expected is not None and not np.isclose(
        result,
        float(expected),
        rtol=0.0,
        atol=atol,
    ):
        raise ValueError(
            "transfer entropy backend output diverged from exact "
            "transfer-entropy reference"
        )
    return result


def validate_te_matrix_backend_output(
    value: object,
    *,
    n_osc: int,
    n_bins: int,
    expected: FloatArray | None = None,
    atol: float = 1e-12,
) -> FloatArray:
    """Validate a direct transfer-entropy matrix backend payload."""

    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError("transfer entropy matrix backend output has boolean values")
    if np.iscomplexobj(raw):
        raise ValueError("transfer entropy matrix backend output must be real")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "transfer entropy matrix backend output must be numeric"
        ) from exc
    expected_size = n_osc * n_osc
    if matrix.size != expected_size:
        raise ValueError(
            "transfer entropy matrix backend output size "
            f"{matrix.size} does not match {expected_size}"
        )
    matrix = matrix.reshape(n_osc, n_osc)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("transfer entropy matrix backend output must be finite")
    tolerance = 1.0e-12
    if np.any(matrix < -tolerance):
        raise ValueError("transfer entropy matrix backend output must be non-negative")
    max_entropy = float(np.log(n_bins))
    if np.any(matrix > max_entropy + tolerance):
        raise ValueError(
            "transfer entropy matrix backend output must not exceed log(n_bins)"
        )
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=tolerance):
        raise ValueError("transfer entropy matrix backend output diagonal must be zero")
    matrix = np.maximum(matrix, 0.0)
    np.fill_diagonal(matrix, 0.0)
    if expected is not None:
        reference = np.asarray(expected, dtype=np.float64).reshape(n_osc, n_osc)
        if not np.allclose(matrix, reference, rtol=0.0, atol=atol):
            raise ValueError(
                "transfer entropy matrix backend output diverged from exact "
                "transfer-entropy reference"
            )
    return np.ascontiguousarray(matrix, dtype=np.float64)
