# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes backend boundary validation

"""Shared validation for direct AttnRes accelerator calls."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "validate_attnres_backend_inputs",
    "validate_attnres_backend_output",
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


def _validate_non_negative_int(value: object, *, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{name} must be non-negative, got {resolved}")
    return resolved


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer")
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be positive, got {resolved}")
    return resolved


def _validate_block_size(value: object) -> int:
    """Return the validated attention block size, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("block_size must be -1 or a positive integer")
    resolved = int(value)
    if resolved != -1 and resolved < 1:
        raise ValueError("block_size must be -1 or a positive integer")
    return resolved


def _validate_temperature(value: object) -> float:
    """Return the validated attention temperature, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("temperature must be a finite positive real value")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError("temperature must be a finite positive real value")
    return resolved


def _validate_lambda(value: object) -> float:
    """Return the validated residual lambda, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("lambda_ must be a finite non-negative real value")
    resolved = float(value)
    if not np.isfinite(resolved) or resolved < 0.0:
        raise ValueError("lambda_ must be a finite non-negative real value")
    return resolved


def _validate_float_array(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real float array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_complex_alias(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real float array") from exc
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real float array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite float vector, else raise."""
    vector = _validate_float_array(value, name=name)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    return np.ascontiguousarray(vector, dtype=np.float64)


def _validate_output_topology(matrix: FloatArray, *, name: str) -> None:
    """Validate the AttnRes output coupling topology, else raise."""
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} diagonal must be zero")


def _validate_zero_edge_preservation(
    matrix: FloatArray,
    *,
    n: int,
    knm_flat: object | None,
    name: str,
) -> None:
    """Validate that the backend did not create edges absent from ``K_nm``."""
    if knm_flat is None:
        return
    reference = _validate_float_vector(knm_flat, name="knm_flat")
    expected = n * n
    if reference.size != expected:
        raise ValueError(
            f"knm_flat length {reference.size} does not match n*n={expected}"
        )
    zero_edges = np.isclose(reference.reshape(n, n), 0.0, rtol=0.0, atol=1e-12)
    if not np.allclose(matrix[zero_edges], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must preserve zero coupling edges")


def validate_attnres_backend_inputs(
    knm_flat: object,
    theta: object,
    w_q: object,
    w_k: object,
    w_v: object,
    w_o: object,
    n: object,
    n_heads: object,
    block_size: object,
    temperature: object,
    lambda_: object,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    int,
    int,
    float,
    float,
]:
    """Validate direct AttnRes inputs before optional runtime loading."""
    n_int = _validate_non_negative_int(n, name="n")
    n_heads_int = _validate_positive_int(n_heads, name="n_heads")
    block_size_int = _validate_block_size(block_size)
    temperature_float = _validate_temperature(temperature)
    lambda_float = _validate_lambda(lambda_)
    k = _validate_float_vector(knm_flat, name="knm_flat")
    expected_k = n_int * n_int
    if k.size != expected_k:
        raise ValueError(f"knm_flat length {k.size} does not match n*n={expected_k}")
    theta_vec = _validate_float_vector(theta, name="theta")
    if theta_vec.size != n_int:
        raise ValueError(f"theta length {theta_vec.size} does not match n={n_int}")
    q = _validate_float_vector(w_q, name="w_q")
    key = _validate_float_vector(w_k, name="w_k")
    value = _validate_float_vector(w_v, name="w_v")
    out = _validate_float_vector(w_o, name="w_o")
    if q.size != key.size or q.size != value.size:
        raise ValueError("w_q, w_k, and w_v flattened lengths must match")
    if out.size == 0:
        raise ValueError("w_o must contain at least one value")
    d_model_float = np.sqrt(float(out.size))
    d_model = int(d_model_float)
    if d_model * d_model != out.size or d_model < 2 or d_model % 2 != 0:
        raise ValueError("w_o flattened length must encode an even square d_model")
    if d_model % n_heads_int != 0:
        raise ValueError("d_model must be divisible by n_heads")
    d_head = d_model // n_heads_int
    expected_qkv = n_heads_int * d_model * d_head
    if q.size != expected_qkv:
        raise ValueError(
            f"w_q/w_k/w_v length {q.size} does not match H*d_model*d_head="
            f"{expected_qkv}"
        )
    return (
        k,
        theta_vec,
        q,
        key,
        value,
        out,
        n_int,
        n_heads_int,
        block_size_int,
        temperature_float,
        lambda_float,
    )


def validate_attnres_backend_output(
    value: object,
    *,
    n: int,
    knm_flat: object | None = None,
    name: str = "attnres output",
) -> FloatArray:
    """Validate direct AttnRes backend output before returning it."""
    n_int = _validate_non_negative_int(n, name="n")
    output = _validate_float_array(value, name=name)
    expected = n_int * n_int
    if output.shape == (expected,) or output.shape == (n_int, n_int):
        flat = output.reshape(expected)
    elif output.ndim == 1:
        raise ValueError(f"{name} length {output.size} does not match {expected}")
    else:
        raise ValueError(
            f"{name} shape {output.shape} does not match ({n_int}, {n_int}) "
            f"or flat length {expected}"
        )
    flat = np.ascontiguousarray(flat, dtype=np.float64)
    matrix = flat.reshape(n_int, n_int)
    _validate_output_topology(matrix, name=name)
    _validate_zero_edge_preservation(
        matrix,
        n=n_int,
        knm_flat=knm_flat,
        name=name,
    )
    return flat
