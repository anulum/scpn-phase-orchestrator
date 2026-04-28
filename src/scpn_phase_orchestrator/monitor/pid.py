# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Partial Information Decomposition for phase groups

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = ["redundancy", "synergy"]

# Williams & Beer 2010, arXiv:1004.2515 — PID framework
# Circular MI estimate via binned phase histograms

_DEFAULT_BINS = 32


def _circular_entropy(phases: NDArray, n_bins: int = _DEFAULT_BINS) -> float:
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
    phases_a: NDArray, phases_b: NDArray, n_bins: int = _DEFAULT_BINS
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
    phases_a: NDArray, phases_b: NDArray, n_bins: int = _DEFAULT_BINS
) -> float:
    """MI(A; B) = H(A) + H(B) - H(A, B) for paired circular samples."""
    if len(phases_a) != len(phases_b) or len(phases_a) == 0:
        return 0.0
    ha = _circular_entropy(phases_a, n_bins)
    hb = _circular_entropy(phases_b, n_bins)
    hab = _joint_entropy_2d(phases_a, phases_b, n_bins)
    return max(0.0, ha + hb - hab)


def redundancy(
    phases: NDArray,
    group_a: list[int] | NDArray,
    group_b: list[int] | NDArray,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """Redundant information: shared by both groups about the global phase.

    I_red = min(MI(A; global), MI(B; global))

    Williams & Beer 2010 minimum-MI redundancy.
    """
    n = len(phases)
    if n == 0:
        return 0.0
    group_a = np.asarray(group_a, dtype=np.intp)
    group_b = np.asarray(group_b, dtype=np.intp)
    if len(group_a) == 0 or len(group_b) == 0:
        return 0.0

    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import pid_redundancy as _rust_red

        return float(
            _rust_red(
                np.ascontiguousarray(phases.ravel()),
                group_a.tolist(),
                group_b.tolist(),
                n_bins,
            )
        )

    global_phase = float(np.angle(np.mean(np.exp(1j * phases))))
    global_a = np.full(len(group_a), global_phase)
    global_b = np.full(len(group_b), global_phase)

    mi_a = _mutual_information_paired(phases[group_a], global_a, n_bins)
    mi_b = _mutual_information_paired(phases[group_b], global_b, n_bins)
    return min(mi_a, mi_b)


def synergy(
    phases: NDArray,
    group_a: list[int] | NDArray,
    group_b: list[int] | NDArray,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """Synergistic information: present only in the joint (A, B).

    I_syn = MI(A+B; global) - MI(A; global) - MI(B; global) + I_red

    Positive synergy means the combined group carries information
    about the global state that neither subgroup carries alone.
    """
    n = len(phases)
    if n == 0:
        return 0.0
    group_a = np.asarray(group_a, dtype=np.intp)
    group_b = np.asarray(group_b, dtype=np.intp)
    if len(group_a) == 0 or len(group_b) == 0:
        return 0.0

    if _HAS_RUST:  # pragma: no cover
        from spo_kernel import pid_synergy as _rust_syn

        return float(
            _rust_syn(
                np.ascontiguousarray(phases.ravel()),
                group_a.tolist(),
                group_b.tolist(),
                n_bins,
            )
        )

    global_phase = float(np.angle(np.mean(np.exp(1j * phases))))

    joint_indices = np.concatenate([group_a, group_b])
    global_joint = np.full(len(joint_indices), global_phase)
    global_a = np.full(len(group_a), global_phase)
    global_b = np.full(len(group_b), global_phase)

    mi_joint = _mutual_information_paired(phases[joint_indices], global_joint, n_bins)
    mi_a = _mutual_information_paired(phases[group_a], global_a, n_bins)
    mi_b = _mutual_information_paired(phases[group_b], global_b, n_bins)
    i_red = min(mi_a, mi_b)

    return max(0.0, mi_joint - mi_a - mi_b + i_red)
