# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge backend boundary validation

"""Shared validation for direct Hodge accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
HodgeTuple: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]

__all__ = ["validate_hodge_backend_inputs", "validate_hodge_backend_output"]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype != object:
            return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(raw):
        return True
    if isinstance(value, np.ndarray) and raw.dtype != object:
        return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _validate_n(value: object) -> int:
    """Return the validated oscillator count, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("n must be a non-negative integer")
    n_int = int(value)
    if n_int < 0:
        raise ValueError(f"n must be non-negative, got {n_int}")
    return n_int


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite float vector, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or _contains_complex_alias(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        vector = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite one-dimensional float array"
        ) from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _validate_simplex_array(
    value: object,
    *,
    name: str,
    arity: int,
    count: int,
    n: int,
) -> IntArray:
    """Validate a flattened simplex index array (edges or triangles)."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be integer-valued")
    try:
        indices = raw.astype(np.int64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer index array") from exc
    if indices.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {indices.shape}")
    if indices.size != arity * count:
        raise ValueError(f"{name} length {indices.size} does not match {arity}*{count}")
    if count > 0 and (np.any(indices < 0) or np.any(indices >= n)):
        raise ValueError(f"{name} indices must lie in [0, {n})")
    return np.ascontiguousarray(indices, dtype=np.int64)


def _validate_count(value: object, *, name: str) -> int:
    """Return the validated element count, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be non-negative, got {count}")
    return count


def validate_hodge_backend_inputs(
    knm_flat: object,
    phases: object,
    n: object,
    edges_flat: object,
    n_edges: object,
    tris_flat: object,
    n_tris: object,
) -> tuple[FloatArray, FloatArray, int, IntArray, int, IntArray, int]:
    """Validate direct Hodge inputs before optional runtime loading."""
    n_int = _validate_n(n)
    k = _validate_float_vector(knm_flat, name="knm_flat")
    expected_k = n_int * n_int
    if k.size != expected_k:
        raise ValueError(f"knm_flat length {k.size} does not match n*n={expected_k}")
    p = _validate_float_vector(phases, name="phases")
    if p.size != n_int:
        raise ValueError(f"phases length {p.size} does not match n={n_int}")
    n_edges_int = _validate_count(n_edges, name="n_edges")
    edges = _validate_simplex_array(
        edges_flat,
        name="edges_flat",
        arity=2,
        count=n_edges_int,
        n=n_int,
    )
    n_tris_int = _validate_count(n_tris, name="n_tris")
    tris = _validate_simplex_array(
        tris_flat,
        name="tris_flat",
        arity=3,
        count=n_tris_int,
        n=n_int,
    )
    return k, p, n_int, edges, n_edges_int, tris, n_tris_int


def _validate_output_matrix(value: object, *, n: int) -> FloatArray:
    """Return one validated backend flow matrix."""
    if _contains_boolean_alias(value) or _contains_complex_alias(value):
        raise ValueError("Hodge backend output must be finite real-valued")
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Hodge backend output must be finite real-valued") from exc

    expected_shape = (n, n)
    if raw.shape == (n * n,):
        matrix = raw.reshape(expected_shape)
    elif raw.shape == expected_shape:
        matrix = raw
    else:
        raise ValueError("Hodge backend returned matrices with invalid shape")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Hodge backend returned non-finite values")
    if matrix.size and np.max(np.abs(matrix + matrix.T)) > 1.0e-10:
        raise ValueError("Hodge backend returned non-antisymmetric matrices")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def validate_hodge_backend_output(output: object, *, n: object) -> HodgeTuple:
    """Validate direct Hodge backend output before publication or parity fallback."""
    n_int = _validate_n(n)
    if not isinstance(output, tuple) or len(output) != 3:
        raise ValueError(
            "Hodge backend output must contain gradient, curl, and harmonic matrices"
        )
    gradient_raw, curl_raw, harmonic_raw = output
    return (
        _validate_output_matrix(gradient_raw, n=n_int),
        _validate_output_matrix(curl_raw, n=n_int),
        _validate_output_matrix(harmonic_raw, n=n_int),
    )
