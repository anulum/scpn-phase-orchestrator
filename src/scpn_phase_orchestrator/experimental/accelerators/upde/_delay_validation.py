# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Delayed Kuramoto backend boundary validation

"""Shared validation for direct delayed-Kuramoto accelerator calls."""

from __future__ import annotations

from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["validate_delay_backend_inputs"]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _count(value: object, *, name: str, minimum: int) -> int:
    """Return the validated element count, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    result = _finite_float(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real")
    return result


def _float_vector(value: object, *, name: str, size: int) -> FloatArray:
    """Return ``value`` as a validated finite float vector, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        vector = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {vector.shape}")
    if vector.size != size:
        raise ValueError(f"{name} length {vector.size} does not match {size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(vector, dtype=np.float64)


def validate_delay_backend_inputs(
    phases: object,
    omegas: object,
    knm_flat: object,
    alpha_flat: object,
    n: object,
    zeta: object,
    psi: object,
    dt: object,
    delay_steps: object,
    n_steps: object,
) -> tuple[
    FloatArray, FloatArray, FloatArray, FloatArray, int, float, float, float, int, int
]:
    """Validate direct delayed-Kuramoto inputs before optional runtime loading."""
    n_int = _count(n, name="n", minimum=1)
    delay_int = _count(delay_steps, name="delay_steps", minimum=1)
    steps_int = _count(n_steps, name="n_steps", minimum=0)
    zeta_f = _finite_float(zeta, name="zeta")
    psi_f = _finite_float(psi, name="psi")
    dt_f = _positive_float(dt, name="dt")
    phases_v = _float_vector(phases, name="phases", size=n_int)
    omegas_v = _float_vector(omegas, name="omegas", size=n_int)
    knm_v = _float_vector(knm_flat, name="knm_flat", size=n_int * n_int)
    alpha_v = _float_vector(alpha_flat, name="alpha_flat", size=n_int * n_int)
    return (
        phases_v,
        omegas_v,
        knm_v,
        alpha_v,
        n_int,
        zeta_f,
        psi_f,
        dt_f,
        delay_int,
        steps_int,
    )
