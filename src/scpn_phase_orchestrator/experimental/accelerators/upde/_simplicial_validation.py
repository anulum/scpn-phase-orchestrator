# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - simplicial backend validation contracts

"""Shared validation for direct simplicial Kuramoto accelerator backends."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["validate_simplicial_inputs", "validate_simplicial_output"]

FloatArray: TypeAlias = NDArray[np.float64]
ValidatedSimplicialInputs: TypeAlias = tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    int,
    float,
    float,
    float,
    float,
    int,
]


def _as_real_vector(value: Any, *, name: str) -> FloatArray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if array.dtype == np.bool_ or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be real-valued, not boolean")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued, not complex")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    out = np.ascontiguousarray(array, dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _as_square_flat(value: Any, *, name: str, n: int) -> FloatArray:
    array = _as_real_vector(value, name=name)
    expected = n * n
    if array.size != expected:
        raise ValueError(f"{name} must contain {expected} flattened values")
    return array


def _as_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer")
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1")
    return out


def _as_non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer")
    out = int(value)
    if out < 0:
        raise ValueError(f"{name} must be >= 0")
    return out


def _as_finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite real")
    return out


def _as_positive_real(value: Any, *, name: str) -> float:
    out = _as_finite_real(value, name=name)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


def _as_non_negative_real(value: Any, *, name: str) -> float:
    out = _as_finite_real(value, name=name)
    if out < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return out


def _validate_lengths(
    phases: FloatArray,
    omegas: FloatArray,
    *,
    n: int,
) -> None:
    if phases.size != n:
        raise ValueError("phases length must match n")
    if omegas.size != n:
        raise ValueError("omegas length must match n")


def validate_simplicial_inputs(
    phases: Any,
    omegas: Any,
    knm_flat: Any,
    alpha_flat: Any,
    n: Any,
    zeta: Any,
    psi: Any,
    sigma2: Any,
    dt: Any,
    n_steps: Any,
) -> ValidatedSimplicialInputs:
    """Validate direct pairwise-plus-simplicial backend inputs."""

    n_i = _as_positive_int(n, name="n")
    p = _as_real_vector(phases, name="phases")
    o = _as_real_vector(omegas, name="omegas")
    _validate_lengths(p, o, n=n_i)
    k = _as_square_flat(knm_flat, name="knm_flat", n=n_i)
    if np.any(np.diag(k.reshape(n_i, n_i)) != 0.0):
        raise ValueError("knm_flat diagonal must be exactly zero")
    a = _as_square_flat(alpha_flat, name="alpha_flat", n=n_i)
    return (
        p,
        o,
        k,
        a,
        n_i,
        _as_finite_real(zeta, name="zeta"),
        _as_finite_real(psi, name="psi"),
        _as_non_negative_real(sigma2, name="sigma2"),
        _as_positive_real(dt, name="dt"),
        _as_non_negative_int(n_steps, name="n_steps"),
    )


def validate_simplicial_output(value: Any, *, n: int) -> FloatArray:
    """Validate direct backend torus phases before returning them."""

    out = _as_real_vector(value, name="simplicial backend output")
    if out.size != n:
        raise ValueError(f"simplicial backend output must have length {n}")
    if np.any(out < 0.0) or np.any(out >= TWO_PI):
        raise ValueError("simplicial backend output phases must be in [0, 2*pi)")
    return np.ascontiguousarray(out, dtype=np.float64)
