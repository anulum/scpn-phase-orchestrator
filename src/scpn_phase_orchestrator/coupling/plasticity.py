# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Three-factor Hebbian plasticity for coupling

"""Validated three-factor plasticity updates for coupling matrices.

The module computes pairwise phase eligibility traces and applies a
modulator-gated Hebbian update to `K_nm`. Public functions reject boolean,
non-numeric, non-finite, non-vector, non-square, and shape-mismatched inputs so
plasticity cannot corrupt coupling state silently. The update preserves the
Kuramoto coupling contract by requiring non-negative zero-diagonal `K_nm`,
bounded zero-diagonal eligibility traces, and finite real scalar controls.
"""

from __future__ import annotations

from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["compute_eligibility", "three_factor_update"]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _validate_phase_vector(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 1-D phase vector") from exc
    if phases.ndim != 1:
        raise ValueError(f"{name} must be a finite 1-D phase vector")
    if not np.all(np.isfinite(phases)):
        raise ValueError(f"{name} must contain only finite values")
    return phases


def _validate_square_matrix(value: object, *, name: str) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a finite square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _validate_coupling_matrix(value: object) -> FloatArray:
    knm = _validate_square_matrix(value, name="knm")
    if np.any(knm < -1e-12):
        raise ValueError("knm must be non-negative")
    if not np.allclose(np.diag(knm), 0.0, atol=1e-12):
        raise ValueError("knm diagonal must be zero")
    result: FloatArray = np.maximum(knm, 0.0)
    return result


def _validate_eligibility_matrix(value: object) -> FloatArray:
    eligibility = _validate_square_matrix(value, name="eligibility")
    if not np.allclose(np.diag(eligibility), 0.0, atol=1e-12):
        raise ValueError("eligibility diagonal must be zero")
    if np.any(eligibility < -1.0 - 1e-12) or np.any(eligibility > 1.0 + 1e-12):
        raise ValueError("eligibility values must lie in [-1, 1]")
    result: FloatArray = np.clip(eligibility, -1.0, 1.0)
    return result


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real value")
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validate_learning_rate(value: object) -> float:
    lr = _validate_finite_real(value, name="lr")
    if lr < 0.0:
        raise ValueError("lr must be non-negative")
    return lr


def compute_eligibility(phases: FloatArray) -> FloatArray:
    """Pairwise Hebbian eligibility trace: cos(theta_j - theta_i).

    Returns shape (n, n) with zero diagonal.
    """
    phases = _validate_phase_vector(phases, name="phases")
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    elig = np.cos(diffs)
    np.fill_diagonal(elig, 0.0)
    result: FloatArray = elig
    return result


def three_factor_update(
    knm: FloatArray,
    eligibility: FloatArray,
    modulator: float,
    phase_gate: bool,
    lr: float = 0.01,
) -> FloatArray:
    """Three-factor plasticity rule: K_ij += lr * eligibility_ij * M * gate.

    Factors:
        1. eligibility — pairwise phase correlation (Hebbian trace)
        2. modulator — scalar reward/error signal from L16 director
        3. phase_gate — boolean from TCBO consciousness boundary

    Friston 2005, Philos. Trans. R. Soc. B
    360:815-836 (free energy & synaptic plasticity).

    Args:
        knm: current coupling matrix, shape (n, n).
        eligibility: Hebbian trace, shape (n, n).
        modulator: scalar neuromodulatory signal.
        phase_gate: if False, no update occurs (TCBO below consciousness threshold).
        lr: learning rate.

    Returns:
        Updated coupling matrix (new array, does not mutate input).
    """
    knm = _validate_coupling_matrix(knm)
    eligibility = _validate_eligibility_matrix(eligibility)
    if eligibility.shape != knm.shape:
        raise ValueError(
            "eligibility shape "
            f"{eligibility.shape} does not match knm shape {knm.shape}"
        )
    modulator = _validate_finite_real(modulator, name="modulator")
    if not isinstance(phase_gate, bool):
        raise TypeError("phase_gate must be a bool")
    lr = _validate_learning_rate(lr)
    if not phase_gate:
        return knm.copy()
    delta = lr * eligibility * modulator
    updated = np.maximum(knm + delta, 0.0)
    np.fill_diagonal(updated, 0.0)
    result: FloatArray = updated
    return result
