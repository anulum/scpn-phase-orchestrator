# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare section analysis

"""Poincare section crossings and return time analysis.

Detects when a trajectory crosses a hyperplane, extracts the
crossing points (Poincare map), and computes return time statistics.

For phase oscillators, the natural Poincare section is the plane
where one oscillator's phase crosses a reference value (e.g. 0 or π).

References:
    Poincare 1899, "Les methodes nouvelles de la mecanique celeste".
    Strogatz 2015, "Nonlinear Dynamics and Chaos", Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        phase_poincare_rust as _rust_phase_poincare,
    )
    from spo_kernel import (
        poincare_section_rust as _rust_poincare_section,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = [
    "PoincareResult",
    "poincare_section",
    "return_times",
    "phase_poincare",
]


@dataclass
class PoincareResult:
    """Result of Poincare section analysis.

    Attributes:
        crossings: (M, d) coordinates of section crossings.
        crossing_times: (M,) time indices of crossings.
        return_times: (M-1,) time intervals between consecutive crossings.
        mean_return_time: Mean return time (related to dominant frequency).
        std_return_time: Std of return times (0 for periodic, >0 for chaotic).
    """

    crossings: NDArray
    crossing_times: NDArray
    return_times: NDArray
    mean_return_time: float
    std_return_time: float


def poincare_section(
    trajectory: NDArray,
    normal: NDArray,
    offset: float = 0.0,
    direction: str = "positive",
) -> PoincareResult:
    """Detect crossings of a hyperplane n·x = offset.

    Args:
        trajectory: (T, d) state-space trajectory.
        normal: (d,) normal vector defining the section plane.
        offset: Plane offset (n·x = offset).
        direction: "positive" (crossing from negative to positive side),
            "negative", or "both".

    Returns:
        PoincareResult with crossing coordinates and return times.
    """
    traj = np.atleast_2d(np.asarray(trajectory, dtype=np.float64))
    T, d = traj.shape
    norm_vec = np.asarray(normal, dtype=np.float64)

    if _HAS_RUST:
        flat = np.ascontiguousarray(traj.ravel())
        n_arr = np.ascontiguousarray(norm_vec / np.linalg.norm(norm_vec))
        cr_flat, ct, n_cr = _rust_poincare_section(flat, T, d, n_arr, offset, direction)
        cr_flat = np.asarray(cr_flat)
        ct = np.asarray(ct)
        if n_cr == 0:
            return PoincareResult(
                crossings=np.empty((0, d)),
                crossing_times=np.array([]),
                return_times=np.array([]),
                mean_return_time=0.0,
                std_return_time=0.0,
            )
        crossings_arr = cr_flat.reshape(n_cr, d)
        rt = np.diff(ct)
        return PoincareResult(
            crossings=crossings_arr,
            crossing_times=ct,
            return_times=rt,
            mean_return_time=float(np.mean(rt)) if len(rt) > 0 else 0.0,
            std_return_time=float(np.std(rt)) if len(rt) > 0 else 0.0,
        )

    n = norm_vec / np.linalg.norm(norm_vec)

    # Signed distance from plane
    signed_dist = traj @ n - offset

    crossings = []
    crossing_times = []

    for i in range(len(signed_dist) - 1):
        d0, d1 = signed_dist[i], signed_dist[i + 1]

        is_crossing = (
            (direction == "positive" and d0 < 0 and d1 >= 0)
            or (direction == "negative" and d0 > 0 and d1 <= 0)
            or (direction == "both" and d0 * d1 < 0)
        )
        if not is_crossing:
            continue

        # Linear interpolation for crossing point
        alpha = -d0 / (d1 - d0) if abs(d1 - d0) > 1e-15 else 0.5

        point = traj[i] + alpha * (traj[i + 1] - traj[i])
        crossings.append(point)
        crossing_times.append(i + alpha)

    if not crossings:
        return PoincareResult(
            crossings=np.empty((0, traj.shape[1])),
            crossing_times=np.array([]),
            return_times=np.array([]),
            mean_return_time=0.0,
            std_return_time=0.0,
        )

    crossings_arr = np.array(crossings)
    times_arr = np.array(crossing_times)
    rt = np.diff(times_arr)

    return PoincareResult(
        crossings=crossings_arr,
        crossing_times=times_arr,
        return_times=rt,
        mean_return_time=float(np.mean(rt)) if len(rt) > 0 else 0.0,
        std_return_time=float(np.std(rt)) if len(rt) > 0 else 0.0,
    )


def return_times(
    trajectory: NDArray,
    normal: NDArray,
    offset: float = 0.0,
) -> NDArray:
    """Shortcut: return only the return-time sequence."""
    result = poincare_section(trajectory, normal, offset, direction="positive")
    return result.return_times


def phase_poincare(
    phases: NDArray,
    oscillator_idx: int = 0,
    section_phase: float = 0.0,
) -> PoincareResult:
    """Poincare section for phase oscillator trajectories.

    Takes a phase trajectory (T, N) and detects when oscillator
    oscillator_idx crosses section_phase (mod 2π).

    Args:
        phases: (T, N) phase time series.
        oscillator_idx: Which oscillator defines the section.
        section_phase: Phase value for the section.

    Returns:
        PoincareResult. Crossings contain the full phase vector at each
        crossing time.
    """
    phases = np.atleast_2d(np.asarray(phases, dtype=np.float64))
    T, N = phases.shape

    if _HAS_RUST:
        flat = np.ascontiguousarray(phases.ravel())
        cr_flat, ct, n_cr = _rust_phase_poincare(
            flat,
            T,
            N,
            oscillator_idx,
            section_phase,
        )
        cr_flat = np.asarray(cr_flat)
        ct = np.asarray(ct)
        if n_cr == 0:
            return PoincareResult(
                crossings=np.empty((0, N)),
                crossing_times=np.array([]),
                return_times=np.array([]),
                mean_return_time=0.0,
                std_return_time=0.0,
            )
        crossings_arr = cr_flat.reshape(n_cr, N)
        rt = np.diff(ct)
        return PoincareResult(
            crossings=crossings_arr,
            crossing_times=ct,
            return_times=rt,
            mean_return_time=float(np.mean(rt)) if len(rt) > 0 else 0.0,
            std_return_time=float(np.std(rt)) if len(rt) > 0 else 0.0,
        )

    # Unwrap the target oscillator's phase for crossing detection
    target = np.unwrap(phases[:, oscillator_idx])
    # Shift so crossings are at multiples of 2π + section_phase
    shifted = (target - section_phase) % (2 * np.pi)

    crossings = []
    crossing_times = []

    for i in range(T - 1):
        # Detect 2π wrapping (crossing from near 2π to near 0)
        if shifted[i] > np.pi and shifted[i + 1] < np.pi:
            alpha = shifted[i] / (shifted[i] - shifted[i + 1] + 2 * np.pi)
            alpha = min(max(alpha, 0.0), 1.0)
            point = phases[i] + alpha * (phases[i + 1] - phases[i])
            crossings.append(point)
            crossing_times.append(i + alpha)

    if not crossings:
        return PoincareResult(
            crossings=np.empty((0, N)),
            crossing_times=np.array([]),
            return_times=np.array([]),
            mean_return_time=0.0,
            std_return_time=0.0,
        )

    crossings_arr = np.array(crossings)
    times_arr = np.array(crossing_times)
    rt = np.diff(times_arr)

    return PoincareResult(
        crossings=crossings_arr,
        crossing_times=times_arr,
        return_times=rt,
        mean_return_time=float(np.mean(rt)) if len(rt) > 0 else 0.0,
        std_return_time=float(np.std(rt)) if len(rt) > 0 else 0.0,
    )
