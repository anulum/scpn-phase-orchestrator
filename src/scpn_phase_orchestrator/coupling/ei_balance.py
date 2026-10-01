# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Excitatory/Inhibitory balance

"""Excitatory/inhibitory balance summaries and adjustment helpers.

The module measures signed mean outgoing coupling from caller-specified source
sets. Repeated indices count once. Rust acceleration and the genuinely
kernel-absent NumPy path share the same arithmetic and copy-preserving adjustment
contract. These summaries are engineering diagnostics, not a physiological
validation or a universal synchronisation criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values

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
    """Return whether the value contains any boolean alias."""
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _validate_knm(value: object) -> FloatArray:
    """Return the coupling as a validated finite square matrix, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError("knm must not contain boolean values")
    try:
        require_real_values(value, name="knm", allow_object=True)
        knm = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("knm must be a finite square matrix") from exc
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError("knm must be a finite square matrix")
    if not np.all(np.isfinite(knm)):
        raise ValueError("knm must contain only finite values")
    return np.ascontiguousarray(knm, dtype=np.float64)


def _validate_target_ratio(value: object) -> float:
    """Return the validated target excitation/inhibition ratio, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("target_ratio must be a finite positive real")
    require_real_values(value, name="target_ratio")
    target_ratio = float(value)
    if not np.isfinite(target_ratio) or target_ratio <= 0.0:
        raise ValueError("target_ratio must be a finite positive real")
    return target_ratio


@dataclass
class EIBalance:
    """Summary of excitatory and inhibitory coupling balance.

    ``excitatory_strength`` / ``inhibitory_strength`` aggregate the mean
    coupling from each source group over all targets, and ``ratio`` is their
    signed quotient. The four ``*_to_*`` means describe directed source-to-target
    blocks, including diagonal entries. ``is_balanced`` is the configured
    numerical interval ``[0.8, 1.2]``, not an empirical regime classification.
    """

    ratio: float
    excitatory_strength: float
    inhibitory_strength: float
    is_balanced: bool
    e_to_e: float
    e_to_i: float
    i_to_e: float
    i_to_i: float


def _validate_indices(indices: list[int], n: int, name: str) -> list[int]:
    """Return the validated excitatory/inhibitory indices, else raise."""
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
    return sorted(set(valid))


def _mean(values: FloatArray) -> float:
    """Return a scaled arithmetic mean without overflowing a finite sum.

    Parameters
    ----------
    values : FloatArray
        Finite selected coupling entries.

    Returns
    -------
    float
        Compensated mean, or zero for empty and all-zero selections.
    """
    if values.size == 0:
        return 0.0
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    return fsum(float(v) / scale for v in values.ravel()) / values.size * scale


def _block_mean(
    knm: FloatArray,
    source_mask: NDArray[np.bool_],
    target_mask: NDArray[np.bool_],
) -> float:
    """Mean coupling from ``source_mask`` rows to ``target_mask`` columns.

    Returns ``0.0`` when either group is empty.
    """
    if not (np.any(source_mask) and np.any(target_mask)):
        return 0.0
    return _mean(knm[np.ix_(source_mask, target_mask)])


def compute_ei_balance(
    knm: FloatArray,
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
) -> EIBalance:
    """Compute E/I balance from coupling matrix and layer typing.

    Means retain coupling signs and include diagonal entries. Duplicate indices
    count once; non-negative out-of-range indices are ignored. Groups can overlap
    and need not cover all oscillators. Empty groups have zero strength.

    When the inhibitory mean has magnitude below ``1e-15``, the ratio is
    infinity for positive excitation and one otherwise. Other ratios are signed
    quotients and may overflow to infinity. Balance means ``0.8 <= ratio <= 1.2``.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    excitatory_indices : list[int]
        Indices of the excitatory oscillators.
    inhibitory_indices : list[int]
        Indices of the inhibitory oscillators.

    Returns
    -------
    EIBalance
        The E/I balance summary derived from the coupling typing.

    Raises
    ------
    ValueError
        If coupling is not a finite square real matrix or indices are negative,
        boolean, or non-integral.
    """
    knm = _validate_knm(knm)
    n = knm.shape[0]
    excitatory_indices = _validate_indices(excitatory_indices, n, "excitatory")
    inhibitory_indices = _validate_indices(inhibitory_indices, n, "inhibitory")

    if _HAS_RUST:
        k_flat = np.ascontiguousarray(knm.ravel())
        e_arr = np.array(excitatory_indices, dtype=np.int64)
        i_arr = np.array(inhibitory_indices, dtype=np.int64)
        ratio, e_str, i_str, balanced, e_to_e, e_to_i, i_to_e, i_to_i = _rust_ei(
            k_flat, n, e_arr, i_arr
        )
        if np.isnan(ratio) or not np.all(
            np.isfinite([e_str, i_str, e_to_e, e_to_i, i_to_e, i_to_i])
        ):
            raise ValueError("E/I means must remain finite and ratio must not be NaN")
        return EIBalance(
            ratio=float(ratio),
            excitatory_strength=float(e_str),
            inhibitory_strength=float(i_str),
            is_balanced=bool(balanced),
            e_to_e=float(e_to_e),
            e_to_i=float(e_to_i),
            i_to_e=float(i_to_e),
            i_to_i=float(i_to_i),
        )

    e_mask = np.zeros(n, dtype=bool)
    i_mask = np.zeros(n, dtype=bool)
    for idx in excitatory_indices:
        e_mask[idx] = True
    for idx in inhibitory_indices:
        i_mask[idx] = True

    # Excitatory strength: mean coupling FROM excitatory oscillators
    e_strength = _mean(knm[e_mask, :])
    # Inhibitory strength: mean coupling FROM inhibitory oscillators
    i_strength = _mean(knm[i_mask, :])

    if abs(i_strength) < 1e-15:
        ratio = float("inf") if e_strength > 0 else 1.0
    else:
        ratio = e_strength / i_strength

    return EIBalance(
        ratio=ratio,
        excitatory_strength=e_strength,
        inhibitory_strength=i_strength,
        is_balanced=0.8 <= ratio <= 1.2,
        e_to_e=_block_mean(knm, e_mask, e_mask),
        e_to_i=_block_mean(knm, e_mask, i_mask),
        i_to_e=_block_mean(knm, i_mask, e_mask),
        i_to_i=_block_mean(knm, i_mask, i_mask),
    )


def adjust_ei_ratio(
    knm: FloatArray,
    excitatory_indices: list[int],
    inhibitory_indices: list[int],
    target_ratio: float = 1.0,
) -> FloatArray:
    """Scale inhibitory coupling to achieve target E/I ratio.

    Scale each inhibitory row once by ``current_ratio / target_ratio``.
    The source is preserved and every successful return is an independent copy.
    The scale must be finite and non-zero in float64; underflow to either signed
    zero raises instead of silently erasing inhibitory coupling.
    Magnitudes below ``1e-15`` in either source mean give an unchanged copy;
    so does a ratio within ``1e-10`` of target. Signed inputs remain supported.

    Target attainment also requires disjoint source groups, adequate float64
    precision and an adjusted inhibitory mean with magnitude at least ``1e-15``.
    Below that threshold the summary uses its silent convention. Overlapping
    groups remain admissible, but scaling shared rows changes both source means.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    excitatory_indices : list[int]
        Indices of the excitatory oscillators.
    inhibitory_indices : list[int]
        Indices of the inhibitory oscillators.
    target_ratio : float
        Target excitatory/inhibitory coupling ratio.

    Returns
    -------
    FloatArray
        The finite coupling matrix with each inhibitory row scaled once.

    Raises
    ------
    TypeError
        If target_ratio is not a non-boolean real scalar.
    ValueError
        If matrix or indices are invalid, target_ratio is non-positive or
        non-finite, the scale underflows to zero, or the scale or an adjusted
        element is non-finite.
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
        return _validate_knm(result_flat.reshape(n, n)).copy()

    balance = compute_ei_balance(knm, excitatory_indices, inhibitory_indices)
    if (
        abs(balance.inhibitory_strength) < 1e-15
        or abs(balance.excitatory_strength) < 1e-15
    ):
        return knm.copy()

    current_ratio = balance.ratio
    if abs(current_ratio - target_ratio) < 1e-10:
        return knm.copy()

    # Scale inhibitory rows: I_new = I_old * (current_ratio / target_ratio)
    scale = current_ratio / target_ratio
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError("E/I adjustment must remain finite with a non-zero scale")
    result: FloatArray = knm.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for idx in inhibitory_indices:
            result[idx, :] *= scale
    if not np.all(np.isfinite(result)):
        raise ValueError("E/I adjustment must remain finite")
    return result
