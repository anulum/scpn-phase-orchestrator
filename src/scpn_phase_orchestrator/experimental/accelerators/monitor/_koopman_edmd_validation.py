# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — shared validation for Koopman EDMD backends

"""Shared input/output contracts for the Koopman EDMD accelerator backends.

Every backend (Go, Julia, Mojo) receives the same row-major lifted snapshot
matrices and must return ``(A, B, C)`` of the contracted shapes with only finite
entries. These helpers enforce that contract; the numerical parity of each
backend against the NumPy reference is asserted in the dedicated parity test
rather than recomputed on every call, since the reference solve is itself the
expensive step the backends are meant to replace.
"""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "BackendDimensions",
    "validate_edmd_backend_inputs",
    "validate_edmd_backend_output",
]


class BackendDimensions:
    """The four EDMD snapshot dimensions ``(K, N, m, n)``."""

    __slots__ = ("input_dim", "lift_dim", "samples", "state_dim")

    def __init__(
        self, samples: int, lift_dim: int, input_dim: int, state_dim: int
    ) -> None:
        self.samples = samples
        self.lift_dim = lift_dim
        self.input_dim = input_dim
        self.state_dim = state_dim


def _contains_boolean_alias(value: object) -> bool:
    """Return whether a raw backend payload contains a boolean alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether a raw backend payload contains a complex alias."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(raw):
        return True
    if raw.dtype != object:
        return False
    raw_object = np.asarray(value, dtype=object)
    return any(
        isinstance(item, (complex, np.complexfloating)) for item in raw_object.flat
    )


def _contains_only_real_numbers(value: object) -> bool:
    """Return whether an object-dtype payload carries only real non-bool values."""
    raw = np.asarray(value, dtype=object)
    return all(
        isinstance(item, Real) and not isinstance(item, (bool, np.bool_))
        for item in raw.flat
    )


def _coerce_real_matrix(value: object, *, name: str) -> FloatArray:
    """Return a finite real matrix after rejecting lossy dtype aliases first."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_complex_alias(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only real numbers") from exc
    if raw.dtype == object:
        if not _contains_only_real_numbers(value):
            raise ValueError(f"{name} must contain only real numbers")
    elif raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{name} must contain only real numbers")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_edmd_backend_inputs(
    x_lift: object,
    inputs: object,
    y_lift: object,
    states: object,
) -> BackendDimensions:
    """Validate the snapshot matrices and return their shared dimensions.

    Parameters
    ----------
    x_lift, y_lift : numpy.ndarray
        Lifted snapshots of shape ``(K, N)``.
    inputs : numpy.ndarray
        Controls of shape ``(K, m)``.
    states : numpy.ndarray
        Raw states of shape ``(K, n)``.

    Returns
    -------
    BackendDimensions
        The validated ``(K, N, m, n)`` dimensions.

    Raises
    ------
    ValueError
        If the matrices are not real-valued, 2-D, non-empty, finite, or if they
        disagree on the sample count or lift dimension.
    """
    matrices = {
        "x_lift": _coerce_real_matrix(x_lift, name="x_lift"),
        "inputs": _coerce_real_matrix(inputs, name="inputs"),
        "y_lift": _coerce_real_matrix(y_lift, name="y_lift"),
        "states": _coerce_real_matrix(states, name="states"),
    }
    samples = matrices["x_lift"].shape[0]
    if any(matrix.shape[0] != samples for matrix in matrices.values()):
        raise ValueError("all snapshot matrices must share the sample count")
    if matrices["y_lift"].shape[1] != matrices["x_lift"].shape[1]:
        raise ValueError("x_lift and y_lift must share the lift dimension")
    return BackendDimensions(
        samples=int(samples),
        lift_dim=int(matrices["x_lift"].shape[1]),
        input_dim=int(matrices["inputs"].shape[1]),
        state_dim=int(matrices["states"].shape[1]),
    )


def validate_edmd_backend_output(
    state_matrix: object,
    input_matrix: object,
    output_matrix: object,
    dimensions: BackendDimensions,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validate the contracted shapes and finiteness of a backend's ``(A, B, C)``.

    Parameters
    ----------
    state_matrix, input_matrix, output_matrix : numpy.ndarray
        The backend matrices ``A`` ``(N, N)``, ``B`` ``(N, m)`` and ``C``
        ``(n, N)``.
    dimensions : BackendDimensions
        The dimensions returned by :func:`validate_edmd_backend_inputs`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The validated ``(A, B, C)`` as contiguous float arrays.

    Raises
    ------
    ValueError
        If any matrix has the wrong shape or contains lossy aliases,
        non-numeric, complex, non-finite, or boolean values.
    """
    n_lift = dimensions.lift_dim
    expected = {
        "A": (state_matrix, (n_lift, n_lift)),
        "B": (input_matrix, (n_lift, dimensions.input_dim)),
        "C": (output_matrix, (dimensions.state_dim, n_lift)),
    }
    validated: list[FloatArray] = []
    for name, (matrix, shape) in expected.items():
        array = _coerce_real_matrix(matrix, name=name)
        if array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
        validated.append(array)
    return validated[0], validated[1], validated[2]
