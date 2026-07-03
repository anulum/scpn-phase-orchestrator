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
ArrayPayload: TypeAlias = NDArray[np.generic]

__all__ = [
    "FloatArray",
    "expected_npe_backend_output",
    "expected_phase_distance_backend_output",
    "validate_npe_backend_inputs",
    "validate_npe_backend_output",
    "validate_phase_distance_backend_output",
    "validate_phase_distance_backend_input",
]


def _contains_boolean_alias(raw: ArrayPayload) -> bool:
    """Return whether the value contains any boolean alias."""
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def _contains_complex_alias(raw: ArrayPayload) -> bool:
    """Return whether the value contains any complex-number alias."""
    if np.iscomplexobj(raw):
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, complex | np.complexfloating) for value in raw.flat)


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string-like scalar parsable as a float."""
    if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _contains_numeric_string_alias(raw: ArrayPayload) -> bool:
    """Return whether the array contains a numeric string alias."""
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    object_array = raw.astype(object, copy=False)
    return any(_is_numeric_string_alias(value) for value in object_array.flat)


def validate_phase_distance_backend_input(phases: object) -> FloatArray:
    """Return a finite real one-dimensional phase vector for direct backends."""
    raw = np.asarray(phases)
    if _contains_boolean_alias(raw):
        raise ValueError("phases must not contain boolean values")
    if _contains_numeric_string_alias(raw):
        raise ValueError("phases must not contain numeric-string aliases")
    if _contains_complex_alias(raw):
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


def expected_phase_distance_backend_output(phases: FloatArray) -> FloatArray:
    """Return the exact wrapped circular phase-distance matrix."""
    diff = phases[:, np.newaxis] - phases[np.newaxis, :]
    matrix = np.abs(np.arctan2(np.sin(diff), np.cos(diff)))
    np.fill_diagonal(matrix, 0.0)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _npe_from_distance_matrix(distances: FloatArray, max_radius: float) -> float:
    """Return the single-linkage H0 persistent entropy of a distance matrix."""
    n = int(distances.shape[0])
    if n < 2:
        return 0.0
    triu_idx = np.triu_indices(n, k=1)
    edges = distances[triu_idx]
    sorted_idx = np.argsort(edges)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        """Return the union-find root of ``node`` with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lifetimes: list[float] = []
    for idx in sorted_idx:
        i, j = int(triu_idx[0][idx]), int(triu_idx[1][idx])
        distance = float(edges[idx])
        if distance > max_radius:
            break
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            lifetimes.append(distance)
            if rank[root_i] < rank[root_j]:
                parent[root_i] = root_j
            elif rank[root_i] > rank[root_j]:
                parent[root_j] = root_i
            else:
                parent[root_j] = root_i
                rank[root_i] += 1

    if not lifetimes:
        return 0.0
    total = sum(lifetimes)
    if total < 1.0e-15:
        return 0.0
    probabilities = np.array(lifetimes, dtype=np.float64) / total
    probabilities = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    max_entropy = np.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    if max_entropy < 1.0e-15:
        return 0.0
    return min(1.0, max(0.0, entropy / max_entropy))


def expected_npe_backend_output(phases: FloatArray, max_radius: float) -> float:
    """Return the exact NPE scalar required from a backend."""
    distances = expected_phase_distance_backend_output(phases)
    return _npe_from_distance_matrix(distances, max_radius)


def validate_phase_distance_backend_output(
    distances: object,
    *,
    n_phases: int,
    expected: object | None = None,
    atol: float = 1.0e-10,
) -> FloatArray:
    """Return a validated pairwise circular-distance matrix from a backend."""
    raw = np.asarray(distances)
    if _contains_boolean_alias(raw):
        raise ValueError("phase distance backend output must not contain booleans")
    if _contains_numeric_string_alias(raw):
        raise ValueError(
            "phase distance backend output must not contain numeric-string aliases"
        )
    if _contains_complex_alias(raw):
        raise ValueError("phase distance backend output must contain real values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase distance backend output must be numeric") from exc
    expected_count = n_phases * n_phases
    if array.size != expected_count:
        raise ValueError(
            "phase distance backend output size "
            f"{array.size} does not match {expected_count}"
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
    if expected is not None:
        try:
            expected_matrix = np.asarray(expected, dtype=np.float64).reshape(
                n_phases,
                n_phases,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "expected phase distance backend output must match "
                f"({n_phases}, {n_phases})"
            ) from exc
        if not np.allclose(matrix, expected_matrix, rtol=0.0, atol=atol):
            raise ValueError(
                "phase distance backend output must match exact circular "
                "phase distances"
            )
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_max_radius(max_radius: object) -> float:
    """Return ``max_radius`` as a validated value in [0, pi], else raise."""
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


def validate_npe_backend_output(
    value: object,
    *,
    expected: float | None = None,
    atol: float = 1.0e-12,
) -> float:
    """Return a validated normalised persistent-entropy backend scalar."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"NPE backend output must be a real scalar, got {value!r}")
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"NPE backend output must be finite, got {value!r}")
    tolerance = 1.0e-12
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError(f"NPE backend output must lie in [0, 1], got {value!r}")
    score = min(1.0, max(0.0, score))
    if expected is not None and abs(score - expected) > atol:
        raise ValueError("NPE backend output must match exact NPE")
    return score
