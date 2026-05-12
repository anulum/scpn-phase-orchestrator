# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Partial Information Decomposition for phase groups

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = ["redundancy", "synergy"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]

# Williams & Beer 2010, arXiv:1004.2515 — PID framework
# Circular MI estimate via binned phase histograms

_DEFAULT_BINS = 32


def _validate_n_bins(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    if value < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return int(value)


def _validate_phases(value: object) -> FloatArray:
    try:
        phases = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite 1-D phase vector") from exc
    if phases.ndim != 1:
        raise ValueError("phases must be a finite 1-D phase vector")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases must contain only finite values")
    return phases


def _validate_group_indices(value: object, *, name: str, n_phases: int) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1-D integer index array")
    if raw.dtype == np.bool_:
        raise TypeError(f"{name} must contain integer indices, not booleans")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer indices") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integer indices")
    if not np.all(numeric == np.floor(numeric)):
        raise TypeError(f"{name} must contain integer indices")
    indices = numeric.astype(np.intp)
    if indices.size > 0 and (np.any(indices < 0) or np.any(indices >= n_phases)):
        raise IndexError(f"{name} indices must be within [0, {n_phases})")
    return indices


def _circular_entropy(phases: FloatArray, n_bins: int = _DEFAULT_BINS) -> float:
    """Shannon entropy of a circular phase distribution via histogram."""
    if len(phases) == 0:
        return 0.0
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    counts, _ = np.histogram(phases % (2 * np.pi), bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def _joint_entropy_2d(
    phases_a: FloatArray, phases_b: FloatArray, n_bins: int = _DEFAULT_BINS
) -> float:
    """Joint entropy H(A, B) from paired circular observations."""
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    a_wrapped = phases_a % (2 * np.pi)
    b_wrapped = phases_b % (2 * np.pi)
    hist, _, _ = np.histogram2d(a_wrapped, b_wrapped, bins=[bins, bins])
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist.ravel() / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def _mutual_information_paired(
    phases_a: FloatArray, phases_b: FloatArray, n_bins: int = _DEFAULT_BINS
) -> float:
    """MI(A; B) = H(A) + H(B) - H(A, B) for paired circular samples."""
    if len(phases_a) != len(phases_b) or len(phases_a) == 0:
        return 0.0
    ha = _circular_entropy(phases_a, n_bins)
    hb = _circular_entropy(phases_b, n_bins)
    hab = _joint_entropy_2d(phases_a, phases_b, n_bins)
    return max(0.0, ha + hb - hab)


def redundancy(
    phases: FloatArray,
    group_a: list[int] | IntArray,
    group_b: list[int] | IntArray,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """Redundant information: shared by both groups about the global phase.

    I_red = min(MI(A; global), MI(B; global))

    Williams & Beer 2010 minimum-MI redundancy.
    """
    bin_count = _validate_n_bins(n_bins)
    phase_values = _validate_phases(phases)
    n = len(phase_values)
    if n == 0:
        return 0.0
    group_a_idx = _validate_group_indices(group_a, name="group_a", n_phases=n)
    group_b_idx = _validate_group_indices(group_b, name="group_b", n_phases=n)
    if len(group_a_idx) == 0 or len(group_b_idx) == 0:
        return 0.0

    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import pid_redundancy as _rust_red

        return float(
            _rust_red(
                np.ascontiguousarray(phase_values.ravel()),
                group_a_idx.tolist(),
                group_b_idx.tolist(),
                bin_count,
            )
        )

    global_phase = float(np.angle(np.mean(np.exp(1j * phase_values))))
    global_a: FloatArray = np.full(len(group_a_idx), global_phase)
    global_b: FloatArray = np.full(len(group_b_idx), global_phase)

    mi_a = _mutual_information_paired(phase_values[group_a_idx], global_a, bin_count)
    mi_b = _mutual_information_paired(phase_values[group_b_idx], global_b, bin_count)
    return min(mi_a, mi_b)


def synergy(
    phases: FloatArray,
    group_a: list[int] | IntArray,
    group_b: list[int] | IntArray,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """Synergistic information: present only in the joint (A, B).

    I_syn = MI(A+B; global) - MI(A; global) - MI(B; global) + I_red

    Positive synergy means the combined group carries information
    about the global state that neither subgroup carries alone.
    """
    bin_count = _validate_n_bins(n_bins)
    phase_values = _validate_phases(phases)
    n = len(phase_values)
    if n == 0:
        return 0.0
    group_a_idx = _validate_group_indices(group_a, name="group_a", n_phases=n)
    group_b_idx = _validate_group_indices(group_b, name="group_b", n_phases=n)
    if len(group_a_idx) == 0 or len(group_b_idx) == 0:
        return 0.0

    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import pid_synergy as _rust_syn

        return float(
            _rust_syn(
                np.ascontiguousarray(phase_values.ravel()),
                group_a_idx.tolist(),
                group_b_idx.tolist(),
                bin_count,
            )
        )

    global_phase = float(np.angle(np.mean(np.exp(1j * phase_values))))

    joint_indices: IntArray = np.concatenate([group_a_idx, group_b_idx])
    global_joint: FloatArray = np.full(len(joint_indices), global_phase)
    global_a: FloatArray = np.full(len(group_a_idx), global_phase)
    global_b: FloatArray = np.full(len(group_b_idx), global_phase)

    mi_joint = _mutual_information_paired(
        phase_values[joint_indices], global_joint, bin_count
    )
    mi_a = _mutual_information_paired(phase_values[group_a_idx], global_a, bin_count)
    mi_b = _mutual_information_paired(phase_values[group_b_idx], global_b, bin_count)
    i_red = min(mi_a, mi_b)

    return max(0.0, mi_joint - mi_a - mi_b + i_red)
