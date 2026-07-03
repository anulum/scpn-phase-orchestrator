# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - swarmalator backend validation contracts

"""Shared validation for direct swarmalator accelerator backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde._validation_common import (
    contains_boolean_alias,
)

__all__ = ["validate_swarmalator_inputs", "validate_swarmalator_output"]

FloatArray: TypeAlias = NDArray[np.float64]
ValidatedSwarmalatorInputs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
]


def _as_real_array(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real array, else raise."""
    if contains_boolean_alias(value):
        raise ValueError(f"{name} must be real-valued, not boolean")
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued, not complex")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    out = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_real_vector(value: Any, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real vector, else raise."""
    out = _as_real_array(value, name=name)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return out


def _as_position_matrix(value: Any, *, n: int, dim: int, name: str) -> FloatArray:
    """Return ``value`` as a validated finite position matrix, else raise."""
    out = _as_real_array(value, name=name)
    expected = n * dim
    if out.shape == (n, dim):
        return out
    if out.ndim == 1 and out.size == expected:
        return np.ascontiguousarray(out.reshape(n, dim), dtype=np.float64)
    raise ValueError(f"{name} must have shape {(n, dim)} or {expected} values")


def _as_positive_int(value: Any, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise ValueError(f"{name} must be a non-boolean integer")
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1")
    return out


def _as_finite_real(value: Any, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if contains_boolean_alias(value):
        raise ValueError(f"{name} must be finite real, not boolean")
    if not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite real")
    return out


def _as_positive_real(value: Any, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    out = _as_finite_real(value, name=name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def validate_swarmalator_inputs(
    pos: Any,
    phases: Any,
    omegas: Any,
    n: Any,
    dim: Any,
    a: Any,
    b: Any,
    j: Any,
    k: Any,
    dt: Any,
) -> ValidatedSwarmalatorInputs:
    """Validate direct swarmalator backend inputs."""
    n_i = _as_positive_int(n, name="n")
    dim_i = _as_positive_int(dim, name="dim")
    p = _as_position_matrix(pos, n=n_i, dim=dim_i, name="pos")
    ph = _as_real_vector(phases, name="phases")
    if ph.size != n_i:
        raise ValueError("phases length must match n")
    om = _as_real_vector(omegas, name="omegas")
    if om.size != n_i:
        raise ValueError("omegas length must match n")
    return (
        p,
        ph,
        om,
        n_i,
        dim_i,
        _as_finite_real(a, name="a"),
        _as_finite_real(b, name="b"),
        _as_finite_real(j, name="j"),
        _as_finite_real(k, name="k"),
        _as_positive_real(dt, name="dt"),
    )


def validate_swarmalator_output(
    pos: Any,
    phases: Any,
    *,
    n: int,
    dim: int,
) -> tuple[FloatArray, FloatArray]:
    """Validate direct backend positions and torus phases before returning."""
    p = _as_position_matrix(pos, n=n, dim=dim, name="swarmalator output positions")
    ph = _as_real_vector(phases, name="swarmalator output phases")
    if ph.size != n:
        raise ValueError(f"swarmalator output phases must have length {n}")
    if np.any(ph < 0.0) or np.any(ph >= TWO_PI):
        raise ValueError("swarmalator output phases must be in [0, 2*pi)")
    return p, np.ascontiguousarray(ph, dtype=np.float64)
