# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence analysis for phase dynamics

"""Recurrence analysis: recurrence matrix, RQA measures, cross-recurrence.

Recurrence Quantification Analysis (RQA) extracts dynamical invariants
from phase trajectories without requiring stationarity or long time series.

References:
    Eckmann, Kamphorst & Ruelle 1987, Europhys. Lett. 4:973-977.
    Zbilut & Webber 1992, Phys. Lett. A 171:199-203.
    Marwan et al. 2007, Phys. Reports 438:237-329.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RQAResult",
    "recurrence_matrix",
    "rqa",
    "cross_recurrence_matrix",
    "cross_rqa",
]

try:
    from spo_kernel import (  # type: ignore[import-untyped]
        cross_recurrence_matrix_rust as _rust_cross_rm,
    )
    from spo_kernel import (
        recurrence_matrix_rust as _rust_rm,
    )
    from spo_kernel import (
        rqa_rust as _rust_rqa,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


@dataclass
class RQAResult:
    """Recurrence Quantification Analysis measures.

    Attributes:
        recurrence_rate: Fraction of recurrence points (density of R).
        determinism: Fraction of recurrence points forming diagonal lines
            of length >= l_min. High DET → deterministic dynamics.
        avg_diagonal: Mean diagonal line length. Related to prediction
            horizon (Tredicce et al. 2000).
        max_diagonal: Longest diagonal line. Inversely related to
            max Lyapunov exponent: L_max ~ -1/dt * ln(1/L_max_diag).
        entropy_diagonal: Shannon entropy of diagonal line length
            distribution. Measures complexity of deterministic structure.
        laminarity: Fraction of recurrence points forming vertical lines
            of length >= v_min. High LAM → laminar (trapped) states.
        trapping_time: Mean vertical line length. Average time the
            system stays in a laminar state.
        max_vertical: Longest vertical line.
    """

    recurrence_rate: float
    determinism: float
    avg_diagonal: float
    max_diagonal: int
    entropy_diagonal: float
    laminarity: float
    trapping_time: float
    max_vertical: int


def recurrence_matrix(
    trajectory: NDArray,
    epsilon: float,
    metric: str = "euclidean",
) -> NDArray:
    """Compute binary recurrence matrix R_ij = Θ(ε - ||x_i - x_j||).

    Args:
        trajectory: (T, d) state-space trajectory. For phase oscillators,
            d=N (one phase per oscillator) or d=1 (single oscillator).
        epsilon: Recurrence threshold. States closer than ε are recurrent.
        metric: Distance metric. "euclidean" or "angular" (for phases
            on the circle, uses chord distance).

    Returns:
        (T, T) boolean recurrence matrix.
    """
    traj = np.asarray(trajectory)
    if traj.ndim == 1:
        traj = traj[:, np.newaxis]

    if _HAS_RUST:
        t, d = traj.shape
        flat = np.ascontiguousarray(traj.ravel(), dtype=np.float64)
        r_flat = np.asarray(_rust_rm(flat, t, d, epsilon, metric == "angular"))
        return r_flat.reshape(t, t).astype(bool)

    if metric == "angular":
        # Chord distance on circle: 2*sin(|Δθ|/2) for each dimension
        diff = traj[:, np.newaxis, :] - traj[np.newaxis, :, :]
        dist = np.sqrt(np.sum(4 * np.sin(diff / 2) ** 2, axis=2))
    else:
        diff = traj[:, np.newaxis, :] - traj[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

    result: NDArray = dist <= epsilon
    return result


def _diagonal_lines(R: NDArray, l_min: int = 2) -> list[int]:
    """Extract diagonal line lengths from recurrence matrix (exclude main diagonal)."""
    T = R.shape[0]
    lengths: list[int] = []

    # Scan all diagonals (positive offsets only — R is symmetric)
    for k in range(1, T):
        diag = np.diag(R, k)
        count = 0
        for val in diag:
            if val:
                count += 1
            else:
                if count >= l_min:
                    lengths.append(count)
                count = 0
        if count >= l_min:
            lengths.append(count)

    return lengths


def _vertical_lines(R: NDArray, v_min: int = 2) -> list[int]:
    """Extract vertical line lengths from recurrence matrix."""
    T = R.shape[0]
    lengths: list[int] = []

    for col in range(T):
        count = 0
        for row in range(T):
            if R[row, col]:
                count += 1
            else:
                if count >= v_min:
                    lengths.append(count)
                count = 0
        if count >= v_min:
            lengths.append(count)

    return lengths


def rqa(
    trajectory: NDArray,
    epsilon: float,
    l_min: int = 2,
    v_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult:
    """Full Recurrence Quantification Analysis.

    Args:
        trajectory: (T, d) state-space trajectory.
        epsilon: Recurrence threshold.
        l_min: Minimum diagonal line length for DET computation.
        v_min: Minimum vertical line length for LAM computation.
        metric: "euclidean" or "angular".

    Returns:
        RQAResult with all standard RQA measures.
    """
    if _HAS_RUST:
        traj = np.asarray(trajectory)
        if traj.ndim == 1:
            traj = traj[:, np.newaxis]
        t_len, d = traj.shape
        flat = np.ascontiguousarray(traj.ravel(), dtype=np.float64)
        r_flat = np.asarray(
            _rust_rm(flat, t_len, d, epsilon, metric == "angular"),
        )
        rr, det, avg_d, max_d, ent_d, lam, tt, max_v = _rust_rqa(
            r_flat, t_len, l_min, v_min, True,
        )
        return RQAResult(
            recurrence_rate=rr,
            determinism=det,
            avg_diagonal=avg_d,
            max_diagonal=int(max_d),
            entropy_diagonal=ent_d,
            laminarity=lam,
            trapping_time=tt,
            max_vertical=int(max_v),
        )

    R = recurrence_matrix(trajectory, epsilon, metric)
    T = R.shape[0]
    total_points = T * T

    # Recurrence rate (exclude main diagonal)
    np.fill_diagonal(R, False)
    n_recurrent = int(np.sum(R))
    off_diag_total = total_points - T
    rr = n_recurrent / off_diag_total if off_diag_total > 0 else 0.0

    # Diagonal lines
    diag_lengths = _diagonal_lines(R, l_min)
    if diag_lengths:
        diag_points = sum(diag_lengths)
        det = diag_points / n_recurrent if n_recurrent > 0 else 0.0
        avg_diag = float(np.mean(diag_lengths))
        max_diag = max(diag_lengths)
        # Shannon entropy of length distribution
        counts = np.bincount(diag_lengths)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        ent_diag = float(-np.sum(probs * np.log(probs)))
    else:
        det = 0.0
        avg_diag = 0.0
        max_diag = 0
        ent_diag = 0.0

    # Vertical lines
    vert_lengths = _vertical_lines(R, v_min)
    if vert_lengths:
        vert_points = sum(vert_lengths)
        lam = vert_points / n_recurrent if n_recurrent > 0 else 0.0
        trap_time = float(np.mean(vert_lengths))
        max_vert = max(vert_lengths)
    else:
        lam = 0.0
        trap_time = 0.0
        max_vert = 0

    return RQAResult(
        recurrence_rate=rr,
        determinism=det,
        avg_diagonal=avg_diag,
        max_diagonal=max_diag,
        entropy_diagonal=ent_diag,
        laminarity=lam,
        trapping_time=trap_time,
        max_vertical=max_vert,
    )


def cross_recurrence_matrix(
    traj_a: NDArray,
    traj_b: NDArray,
    epsilon: float,
    metric: str = "euclidean",
) -> NDArray:
    """Cross-recurrence matrix CR_ij = Θ(ε - ||x_i - y_j||).

    Measures when two systems visit similar states at possibly
    different times. Useful for detecting phase synchronization
    between oscillator groups.

    Args:
        traj_a: (T, d) first trajectory.
        traj_b: (T, d) second trajectory (same T and d).
        epsilon: Recurrence threshold.
        metric: "euclidean" or "angular".

    Returns:
        (T, T) boolean cross-recurrence matrix.
    """
    a = np.atleast_2d(traj_a)
    b = np.atleast_2d(traj_b)
    if a.ndim == 1:
        a = a[:, np.newaxis]
    if b.ndim == 1:
        b = b[:, np.newaxis]

    if _HAS_RUST:
        t, d = a.shape
        a_flat = np.ascontiguousarray(a.ravel(), dtype=np.float64)
        b_flat = np.ascontiguousarray(b.ravel(), dtype=np.float64)
        cr_flat = np.asarray(
            _rust_cross_rm(a_flat, b_flat, t, d, epsilon, metric == "angular"),
        )
        return cr_flat.reshape(t, t).astype(bool)

    if metric == "angular":
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        dist = np.sqrt(np.sum(4 * np.sin(diff / 2) ** 2, axis=2))
    else:
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))

    result: NDArray = dist <= epsilon
    return result


def cross_rqa(
    traj_a: NDArray,
    traj_b: NDArray,
    epsilon: float,
    l_min: int = 2,
    metric: str = "euclidean",
) -> RQAResult:
    """Cross-Recurrence Quantification Analysis between two trajectories.

    Args:
        traj_a: (T, d) first trajectory.
        traj_b: (T, d) second trajectory.
        epsilon: Recurrence threshold.
        l_min: Minimum diagonal line length.
        metric: "euclidean" or "angular".

    Returns:
        RQAResult (laminarity/trapping computed on vertical lines of CR).
    """
    CR = cross_recurrence_matrix(traj_a, traj_b, epsilon, metric)
    T = CR.shape[0]
    total_points = T * T

    n_recurrent = int(np.sum(CR))
    rr = n_recurrent / total_points if total_points > 0 else 0.0

    diag_lengths = _diagonal_lines(CR, l_min)
    if diag_lengths:
        diag_points = sum(diag_lengths)
        det = diag_points / n_recurrent if n_recurrent > 0 else 0.0
        avg_diag = float(np.mean(diag_lengths))
        max_diag = max(diag_lengths)
        counts = np.bincount(diag_lengths)
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        ent_diag = float(-np.sum(probs * np.log(probs)))
    else:
        det = 0.0
        avg_diag = 0.0
        max_diag = 0
        ent_diag = 0.0

    vert_lengths = _vertical_lines(CR, l_min)
    if vert_lengths:
        vert_points = sum(vert_lengths)
        lam = vert_points / n_recurrent if n_recurrent > 0 else 0.0
        trap_time = float(np.mean(vert_lengths))
        max_vert = max(vert_lengths)
    else:
        lam = 0.0
        trap_time = 0.0
        max_vert = 0

    return RQAResult(
        recurrence_rate=rr,
        determinism=det,
        avg_diagonal=avg_diag,
        max_diagonal=max_diag,
        entropy_diagonal=ent_diag,
        laminarity=lam,
        trapping_time=trap_time,
        max_vertical=max_vert,
    )
