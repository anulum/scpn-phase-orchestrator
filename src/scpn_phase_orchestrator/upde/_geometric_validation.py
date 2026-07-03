# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - geometric backend validation contracts

"""Shared validation for direct torus-geometric accelerator backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = ["validate_torus_inputs", "validate_torus_output"]

FloatArray: TypeAlias = NDArray[np.float64]
ValidatedInputs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    float,
    float,
    float,
    int,
]

TWO_PI = 2.0 * np.pi


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


def validate_torus_inputs(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> ValidatedInputs:
    """Validate deterministic direct torus-run kernel inputs."""
    n_i = _as_int(n, name="n", minimum=1)
    p = _as_finite_vector(phases, name="phases")
    if p.size != n_i:
        raise ValueError("phases length must match n")
    o = _as_finite_vector(omegas, name="omegas")
    if o.size != n_i:
        raise ValueError("omegas length must match n")
    return (
        p,
        o,
        _as_flat_matrix(knm_flat, name="knm_flat", n=n_i),
        _as_flat_matrix(alpha_flat, name="alpha_flat", n=n_i),
        n_i,
        _as_finite_real(zeta, name="zeta"),
        _as_finite_real(psi, name="psi"),
        _as_finite_real(dt, name="dt", positive=True),
        _as_int(n_steps, name="n_steps", minimum=0),
    )


def validate_torus_output(value: Any, *, n: int) -> FloatArray:
    """Validate backend torus phases before returning them."""
    out = _as_finite_vector(value, name="result")
    if out.size != n:
        raise ValueError(f"result must contain {n} values")
    if np.any((out < 0.0) | (out >= TWO_PI + 1e-12)):
        raise ValueError("result phases must lie in [0, 2*pi)")
    return out % TWO_PI
