# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF cost terms

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.spectral import fiedler_value
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

try:
    from spo_kernel import (
        compute_ssgf_costs_rust as _rust_costs,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["SSGFCosts", "compute_ssgf_costs"]

FloatArray: TypeAlias = NDArray[np.float64]


def _validate_weights(weights: tuple[float, ...]) -> tuple[float, float, float, float]:
    if not isinstance(weights, tuple) or len(weights) != 4:
        raise ValueError("weights must be a tuple of four finite non-negative reals")
    parsed: list[float] = []
    for weight in weights:
        if isinstance(weight, bool) or not isinstance(weight, Real):
            raise ValueError("weights must contain finite non-negative real values")
        value = float(weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("weights must contain finite non-negative real values")
        parsed.append(value)
    return parsed[0], parsed[1], parsed[2], parsed[3]


def _validate_phases(phases: FloatArray) -> FloatArray:
    raw = np.asarray(phases)
    if raw.dtype == np.bool_:
        raise ValueError("phases must not contain boolean values")
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


def _validate_weight_matrix(W: FloatArray) -> FloatArray:
    raw = np.asarray(W)
    if raw.dtype == np.bool_:
        raise ValueError("W must not contain boolean values")
    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("W must be numeric") from exc
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("W must be a square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("W must contain only finite values")
    return values


@dataclass
class SSGFCosts:
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
    """
    w1, w2, w3, w4 = _validate_weights(weights)
    phases = _validate_phases(phases)
    W_array = _validate_weight_matrix(W)
    if phases.shape[0] != W_array.shape[0]:
        raise ValueError("phases length must match W dimensions")
    n = W_array.shape[0]

    if _HAS_RUST:
        w_flat: FloatArray = np.ascontiguousarray(W_array.ravel())
        p: FloatArray = np.ascontiguousarray(phases, dtype=np.float64)
        c1, c2, c3, c4, ut = _rust_costs(w_flat, p, n, w1, w2, w3, w4)
        return SSGFCosts(
            c1_sync=c1,
            c2_spectral_gap=c2,
            c3_sparsity=c3,
            c4_symmetry=c4,
            u_total=ut,
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
