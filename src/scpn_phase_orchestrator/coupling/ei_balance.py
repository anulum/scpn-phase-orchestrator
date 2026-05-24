# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Excitatory/Inhibitory balance

"""Excitatory/inhibitory balance summaries and adjustment helpers.

The module measures mean outgoing coupling from caller-specified excitatory and
inhibitory index sets, then optionally rescales inhibitory rows toward a target
ratio. Rust acceleration is used when available; the NumPy fallback preserves
the same shape and summary contract for examples and deterministic tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        adjust_ei_ratio_rust as _rust_adjust,
    )
    from spo_kernel import (
        compute_ei_balance_rust as _rust_ei,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["EIBalance", "compute_ei_balance", "adjust_ei_ratio"]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool) for item in raw.ravel())


def _validate_knm(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("knm must not contain boolean values")
    try:
        knm = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("knm must be a finite square matrix") from exc
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError("knm must be a finite square matrix")
    if not np.all(np.isfinite(knm)):
        raise ValueError("knm must contain only finite values")
    return np.ascontiguousarray(knm, dtype=np.float64)


def _validate_target_ratio(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("target_ratio must be a finite positive real")
    target_ratio = float(value)
    if not np.isfinite(target_ratio) or target_ratio <= 0.0:
        raise ValueError("target_ratio must be a finite positive real")
    return target_ratio


@dataclass
class EIBalance:
    """Summary of excitatory and inhibitory coupling balance."""

    ratio: float
    excitatory_strength: float
    inhibitory_strength: float
    is_balanced: bool


def _validate_indices(indices: list[int], n: int, name: str) -> list[int]:
    valid: list[int] = []
    for idx in indices:
        if not isinstance(idx, int) or isinstance(idx, bool):
            msg = f"{name} indices must be integers, got {idx!r}"
            raise ValueError(msg)
        if idx < 0:
            msg = f"{name} indices must be non-negative, got {idx}"
            raise ValueError(msg)
        if idx < n:
            valid.append(idx)
    return valid


def compute_ei_balance(
    knm: FloatArray,
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
) -> EIBalance:
    """Compute E/I balance from coupling matrix and layer typing.

    Kuroki & Mizuseki 2025, Neural Computation — E/I balance is the
    critical parameter for synchronization, not K or D.

    ratio > 1: excitation-dominated (hypersynchrony risk)
    ratio < 1: inhibition-dominated (desynchronization risk)
    ratio ≈ 1: balanced (optimal for metastability)
    """
    knm = _validate_knm(knm)
    n = knm.shape[0]
    excitatory_indices = _validate_indices(excitatory_indices, n, "excitatory")
    inhibitory_indices = _validate_indices(inhibitory_indices, n, "inhibitory")

    if _HAS_RUST:
        k_flat = np.ascontiguousarray(knm.ravel())
        e_arr = np.array(excitatory_indices, dtype=np.int64)
        i_arr = np.array(inhibitory_indices, dtype=np.int64)
        ratio, e_str, i_str, balanced = _rust_ei(k_flat, n, e_arr, i_arr)
        return EIBalance(
            ratio=float(ratio),
            excitatory_strength=float(e_str),
            inhibitory_strength=float(i_str),
            is_balanced=bool(balanced),
        )

    e_mask = np.zeros(n, dtype=bool)
    i_mask = np.zeros(n, dtype=bool)
    for idx in excitatory_indices:
        e_mask[idx] = True
    for idx in inhibitory_indices:
        i_mask[idx] = True

    # Excitatory strength: mean coupling FROM excitatory oscillators
    e_strength = float(np.mean(knm[e_mask, :])) if np.any(e_mask) else 0.0
    # Inhibitory strength: mean coupling FROM inhibitory oscillators
    i_strength = float(np.mean(knm[i_mask, :])) if np.any(i_mask) else 0.0

    if i_strength < 1e-15:
        ratio = float("inf") if e_strength > 0 else 1.0
    else:
        ratio = e_strength / i_strength

    return EIBalance(
        ratio=ratio,
        excitatory_strength=e_strength,
        inhibitory_strength=i_strength,
        is_balanced=0.8 <= ratio <= 1.2,
    )


def adjust_ei_ratio(
    knm: FloatArray,
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
    target_ratio: float = 1.0,
) -> FloatArray:
    """Scale inhibitory coupling to achieve target E/I ratio.

    Returns modified knm with inhibitory rows scaled so that
    E_strength / I_strength ≈ target_ratio.
    """
    knm = _validate_knm(knm)
    target_ratio = _validate_target_ratio(target_ratio)
    n = knm.shape[0]
    excitatory_indices = _validate_indices(excitatory_indices, n, "excitatory")
    inhibitory_indices = _validate_indices(inhibitory_indices, n, "inhibitory")

    if _HAS_RUST:
        k_flat = np.ascontiguousarray(knm.ravel())
        e_arr = np.array(excitatory_indices, dtype=np.int64)
        i_arr = np.array(inhibitory_indices, dtype=np.int64)
        result_flat: FloatArray = np.asarray(
            _rust_adjust(k_flat, n, e_arr, i_arr, target_ratio),
        )
        return result_flat.reshape(n, n)

    balance = compute_ei_balance(knm, excitatory_indices, inhibitory_indices)
    if balance.inhibitory_strength < 1e-15 or balance.excitatory_strength < 1e-15:
        return knm.copy()

    current_ratio = balance.ratio
    if abs(current_ratio - target_ratio) < 1e-10:
        return knm.copy()

    # Scale inhibitory rows: I_new = I_old * (current_ratio / target_ratio)
    scale = current_ratio / target_ratio
    result: FloatArray = knm.copy()
    for idx in inhibitory_indices:
        result[idx, :] *= scale
    return result
