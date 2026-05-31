# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov backend validation

"""Shared validation for direct Lyapunov polyglot backend calls."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real-valued vector")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_matrix(value: object, *, name: str, n: int) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real-valued matrix")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite matrix") from exc
    expected_shape = (n, n)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _validate_positive_real(value: object, *, name: str) -> float:
    scalar = _validate_finite_real(value, name=name)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive, got {scalar}")
    return scalar


def _validate_non_negative_real(value: object, *, name: str) -> float:
    scalar = _validate_finite_real(value, name=name)
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative, got {scalar}")
    return scalar


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    scalar = int(value)
    if scalar < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {scalar}")
    return scalar


def _validate_zero_diagonal(knm: FloatArray) -> None:
    if np.any(np.abs(np.diag(knm)) > 1e-12):
        raise ValueError(
            "knm diagonal must be zero; Kuramoto self-coupling is not a "
            "physical pair interaction"
        )


def validate_lyapunov_backend_inputs(
    phases_init: object,
    omegas: object,
    knm: object,
    alpha: object,
    dt: object,
    n_steps: object,
    qr_interval: object,
    zeta: object,
    psi: object,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    int,
    int,
    float,
    float,
]:
    """Validate direct polyglot Lyapunov backend arguments before runtime load."""
    phases = _validate_vector(phases_init, name="phases_init")
    n = int(phases.size)
    omega_values = _validate_vector(omegas, name="omegas")
    if omega_values.shape != phases.shape:
        raise ValueError(
            f"omegas shape {omega_values.shape} does not match {phases.shape}",
        )
    coupling = _validate_matrix(knm, name="knm", n=n)
    _validate_zero_diagonal(coupling)
    lag = _validate_matrix(alpha, name="alpha", n=n)
    return (
        phases,
        omega_values,
        coupling,
        lag,
        _validate_positive_real(dt, name="dt"),
        _validate_int_at_least(n_steps, name="n_steps", minimum=0),
        _validate_int_at_least(qr_interval, name="qr_interval", minimum=1),
        _validate_non_negative_real(zeta, name="zeta"),
        _validate_finite_real(psi, name="psi"),
    )
