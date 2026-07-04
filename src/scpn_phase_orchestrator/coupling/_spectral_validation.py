# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — spectral backend boundary validation

"""Shared validation for direct spectral accelerator calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
SpectralOutput: TypeAlias = tuple[FloatArray, FloatArray]

__all__ = [
    "contains_numeric_string_alias",
    "is_numeric_string_alias",
    "validate_spectral_backend_inputs",
    "validate_spectral_backend_output",
]


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


def is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string scalar accepted by ``float``."""
    if not isinstance(value, (str, bytes, np.str_, np.bytes_)):
        return False
    try:
        if isinstance(value, (bytes, np.bytes_)):
            text = bytes(value).decode()
        else:
            text = str(value)
    except UnicodeDecodeError:
        return False
    if text.strip() == "":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def contains_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` contains a stringified numeric scalar."""
    if is_numeric_string_alias(value):
        return True
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    return any(
        is_numeric_string_alias(item) for item in raw.astype(object, copy=False).flat
    )


def _validate_n(value: object) -> int:
    """Return the validated oscillator count, else raise."""
    if is_numeric_string_alias(value):
        raise ValueError("n must not be a numeric-string alias")
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("n must be a non-negative integer")
    n_int = int(value)
    if n_int < 0:
        raise ValueError(f"n must be non-negative, got {n_int}")
    return n_int


def _validate_knm_flat(value: object) -> FloatArray:
    """Return ``value`` as a validated flattened coupling matrix, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError("knm_flat must not contain boolean values")
    if contains_numeric_string_alias(value):
        raise ValueError("knm_flat must not contain numeric-string aliases")
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or _contains_complex_alias(value):
        raise ValueError("knm_flat must be real-valued")
    try:
        knm_flat = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "knm_flat must be a finite one-dimensional float array"
        ) from exc
    if knm_flat.ndim != 1:
        raise ValueError(
            f"knm_flat must be one-dimensional, got shape {knm_flat.shape}"
        )
    if not np.all(np.isfinite(knm_flat)):
        raise ValueError("knm_flat must contain only finite values")
    return np.ascontiguousarray(knm_flat, dtype=np.float64)


def validate_spectral_backend_inputs(
    knm_flat: object,
    n: object,
) -> tuple[FloatArray, int]:
    """Validate direct spectral inputs before optional runtime loading."""
    n_int = _validate_n(n)
    k = _validate_knm_flat(knm_flat)
    expected = n_int * n_int
    if k.size != expected:
        raise ValueError(f"knm_flat length {k.size} does not match n*n={expected}")
    return k, n_int


def validate_spectral_backend_output(
    value: object,
    *,
    n: object,
) -> SpectralOutput:
    """Validate direct spectral output after optional runtime execution.

    Optional backends are untrusted across the Python boundary even after
    validated inputs. The accepted contract is exactly two finite real numeric
    vectors of length ``n``: eigenvalues sorted ascending and a non-zero
    Fiedler vector for non-trivial oscillator counts.
    """
    n_int = _validate_n(n)
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("spectral primitive output must be (eigvals, fiedler)")
    eigvals_raw = value[0]
    fiedler_raw = value[1]
    if (
        _contains_boolean_alias(eigvals_raw)
        or _contains_boolean_alias(fiedler_raw)
        or _contains_complex_alias(eigvals_raw)
        or _contains_complex_alias(fiedler_raw)
    ):
        raise ValueError("spectral primitive output must be real-valued numeric arrays")
    if contains_numeric_string_alias(eigvals_raw) or contains_numeric_string_alias(
        fiedler_raw
    ):
        raise ValueError(
            "spectral primitive output must not contain numeric-string aliases"
        )
    try:
        eigvals = np.asarray(eigvals_raw, dtype=np.float64)
        fiedler = np.asarray(fiedler_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("spectral primitive output must be numeric") from exc
    if eigvals.shape != (n_int,):
        raise ValueError(
            f"spectral eigenvalue shape {eigvals.shape} must be ({n_int},)"
        )
    if fiedler.shape != (n_int,):
        raise ValueError(f"spectral fiedler shape {fiedler.shape} must be ({n_int},)")
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(fiedler)):
        raise ValueError("spectral primitive output must contain only finite values")
    tolerance = 1e-10
    if np.any(eigvals < -tolerance):
        raise ValueError("spectral eigenvalues must be non-negative")
    if np.any(np.diff(eigvals) < -tolerance):
        raise ValueError("spectral eigenvalues must be sorted ascending")
    if n_int > 1 and np.linalg.norm(fiedler) <= tolerance:
        raise ValueError("spectral fiedler vector must be non-zero")
    return (
        np.ascontiguousarray(np.maximum(eigvals, 0.0), dtype=np.float64),
        np.ascontiguousarray(fiedler, dtype=np.float64),
    )
