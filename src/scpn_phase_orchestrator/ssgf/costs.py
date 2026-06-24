# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF cost terms

"""Validated SSGF total-cost terms for phase and coupling geometry states.

``compute_ssgf_costs`` combines synchronization deficit, spectral gap,
sparsity, and symmetry terms into a weighted objective. The public entry point
rejects boolean, non-numeric, non-finite, non-vector phase inputs, non-square
or non-finite coupling matrices, and invalid weight tuples before dispatching
to Rust or Python. This keeps accelerated and fallback paths aligned on the
same physical dimensions and cost semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.spectral import fiedler_value
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

try:
    from spo_kernel import (
        compute_ssgf_costs_rust as _loaded_rust_costs,
    )

    _rust_costs: Callable[..., object] | None = cast(
        "Callable[..., object]",
        _loaded_rust_costs,
    )
    _HAS_RUST = True
except ImportError:
    _rust_costs = None
    _HAS_RUST = False

__all__ = ["SSGFCosts", "compute_ssgf_costs"]

FloatArray: TypeAlias = NDArray[np.float64]


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
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


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
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.ravel())


def _validate_weights(weights: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Return the validated cost weights, else raise."""
    if not isinstance(weights, tuple) or len(weights) != 4:
        raise ValueError("weights must be a tuple of four finite non-negative reals")
    parsed: list[float] = []
    for weight in weights:
        if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, Real):
            raise ValueError("weights must contain finite non-negative real values")
        value = float(weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("weights must contain finite non-negative real values")
        parsed.append(value)
    return parsed[0], parsed[1], parsed[2], parsed[3]


def _validate_phases(phases: object) -> FloatArray:
    """Return the phases as a validated finite array, else raise."""
    if _contains_boolean_alias(phases):
        raise ValueError("phases must not contain boolean values")
    raw = np.asarray(phases)
    if np.iscomplexobj(raw) or _contains_complex_alias(phases):
        raise ValueError("phases must be real-valued")
    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be numeric") from exc
    if values.ndim != 1:
        raise ValueError("phases must be a one-dimensional vector")
    if values.shape[0] < 1:
        raise ValueError("phases must contain at least one oscillator")
    if not np.all(np.isfinite(values)):
        raise ValueError("phases must contain only finite values")
    return values


def _validate_weight_matrix(W: object) -> FloatArray:
    """Return the validated weight matrix, else raise."""
    if _contains_boolean_alias(W):
        raise ValueError("W must not contain boolean values")
    raw = np.asarray(W)
    if np.iscomplexobj(raw) or _contains_complex_alias(W):
        raise ValueError("W must be real-valued")
    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("W must be numeric") from exc
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("W must be a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("W must contain only finite values")
    return values


def _validate_cost_scalar(value: object, *, name: str) -> float:
    """Return ``value`` as a validated cost scalar, else raise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-boolean real")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_rust_costs(
    value: object,
    *,
    weights: tuple[float, float, float, float],
) -> SSGFCosts:
    """Return the Rust backend costs matching the reference, else raise."""
    if not isinstance(value, tuple) or len(value) != 5:
        raise ValueError("Rust SSGF costs output must contain five cost terms")
    c1 = _validate_cost_scalar(value[0], name="synchronisation deficit")
    c2 = _validate_cost_scalar(value[1], name="spectral cost")
    c3 = _validate_cost_scalar(value[2], name="sparsity cost")
    c4 = _validate_cost_scalar(value[3], name="symmetry cost")
    u_total = _validate_cost_scalar(value[4], name="weighted total")

    tolerance = 1e-10
    if c1 < -tolerance or c1 > 1.0 + tolerance:
        raise ValueError("synchronisation deficit must stay in [0, 1]")
    if c2 > tolerance:
        raise ValueError("spectral cost must be non-positive")
    if c3 < -tolerance:
        raise ValueError("sparsity cost must be non-negative")
    if c4 < -tolerance:
        raise ValueError("symmetry cost must be non-negative")

    w1, w2, w3, w4 = weights
    expected_total = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4
    if not np.isclose(u_total, expected_total, rtol=1e-10, atol=1e-10):
        raise ValueError("weighted total must equal the weighted SSGF cost terms")

    return SSGFCosts(
        c1_sync=float(np.clip(c1, 0.0, 1.0)),
        c2_spectral_gap=0.0 if abs(c2) <= tolerance else c2,
        c3_sparsity=max(c3, 0.0),
        c4_symmetry=max(c4, 0.0),
        u_total=u_total,
    )


@dataclass
class SSGFCosts:
    """Individual SSGF cost terms plus the weighted total objective."""

    c1_sync: float
    c2_spectral_gap: float
    c3_sparsity: float
    c4_symmetry: float
    u_total: float


def compute_ssgf_costs(
    W: FloatArray,
    phases: FloatArray,
    weights: tuple[float, ...] = (1.0, 0.5, 0.1, 0.1),
) -> SSGFCosts:
    """Compute SSGF cost terms for geometry W given current phases.

    C1: 1 - R (synchronization deficit)
    C2: -λ₂(L(W)) (negative algebraic connectivity — maximize λ₂)
    C3: ||W||₁ / N² (sparsity regularizer — prevent dense coupling)
    C4: ||W - W^T||_F / N (symmetry deviation)

    U_total = w1·C1 + w2·C2 + w3·C3 + w4·C4

    Parameters
    ----------
    W : FloatArray
        Weight matrix.
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    weights : tuple[float, ...]
        The weights.

    Returns
    -------
    SSGFCosts
        SSGF cost terms for geometry W given current phases.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    RuntimeError
        If the operation fails.
    """
    w1, w2, w3, w4 = _validate_weights(weights)
    phases = _validate_phases(phases)
    W_array = _validate_weight_matrix(W)
    if phases.shape[0] != W_array.shape[0]:
        raise ValueError("phases length must match W dimensions")
    n = W_array.shape[0]

    if _HAS_RUST:
        if _rust_costs is None:
            raise RuntimeError("Rust SSGF backend unavailable")
        w_flat: FloatArray = np.ascontiguousarray(W_array.ravel())
        p: FloatArray = np.ascontiguousarray(phases, dtype=np.float64)
        return _validate_rust_costs(
            _rust_costs(w_flat, p, n, w1, w2, w3, w4),
            weights=(w1, w2, w3, w4),
        )

    R, _ = compute_order_parameter(phases)
    c1 = 1.0 - R

    lam2 = fiedler_value(W_array)
    c2 = -lam2  # minimize → maximize algebraic connectivity

    c3 = float(np.sum(np.abs(W_array))) / (n * n) if n > 0 else 0.0

    c4 = float(np.linalg.norm(W_array - W_array.T, "fro")) / n if n > 0 else 0.0

    u_total = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4

    return SSGFCosts(
        c1_sync=c1,
        c2_spectral_gap=c2,
        c3_sparsity=c3,
        c4_symmetry=c4,
        u_total=u_total,
    )
