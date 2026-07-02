# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - basin-stability backend contracts

"""Shared validation for direct basin-stability accelerator backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = [
    "validate_basin_stability_inputs",
    "validate_basin_stability_output",
]

FloatArray: TypeAlias = NDArray[np.float64]
ValidatedInputs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    float,
    float,
    int,
    int,
]


def _as_finite_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite vector, else raise."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be real-valued, not boolean")
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued, not complex")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    out = np.ascontiguousarray(array, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if out.size == 0:
        raise ValueError(f"{name} must contain at least one oscillator")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_flat_matrix(value: Any, *, name: str, n: int) -> FloatArray:
    """Return ``value`` as a validated flattened matrix, else raise."""
    array = _as_finite_vector(value, name=name)
    expected = n * n
    if array.size != expected:
        raise ValueError(f"{name} must contain {expected} flattened values")
    return array


def _as_finite_real(value: Any, *, name: str, positive: bool = False) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be a real scalar, not boolean")
    if not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    if positive and out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_int(value: Any, *, name: str, minimum: int) -> int:
    """Return ``value`` as a validated integer, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise TypeError(f"{name} must be an integer, not boolean")
    if not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    out = int(value)
    if out < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return out


def validate_basin_stability_inputs(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> ValidatedInputs:
    """Validate deterministic one-trial basin-stability kernel inputs."""
    n_i = _as_int(n, name="n", minimum=1)
    p = _as_finite_vector(phases_init, name="phases_init")
    if p.size != n_i:
        raise ValueError("phases_init length must match n")
    o = _as_finite_vector(omegas, name="omegas")
    if o.size != n_i:
        raise ValueError("omegas length must match n")
    return (
        p,
        o,
        _as_flat_matrix(knm_flat, name="knm_flat", n=n_i),
        _as_flat_matrix(alpha_flat, name="alpha_flat", n=n_i),
        n_i,
        _as_finite_real(k_scale, name="k_scale"),
        _as_finite_real(dt, name="dt", positive=True),
        _as_int(n_transient, name="n_transient", minimum=0),
        _as_int(n_measure, name="n_measure", minimum=0),
    )


def validate_basin_stability_output(value: Any) -> float:
    """Validate an order-parameter result from a direct backend."""
    out = _as_finite_real(value, name="steady-state R")
    if out < 0.0 or out > 1.0 + 1e-12:
        raise ValueError("steady-state R must lie in [0, 1]")
    return min(out, 1.0)
