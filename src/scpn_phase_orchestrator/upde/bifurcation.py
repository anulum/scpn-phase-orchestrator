# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bifurcation analysis for Kuramoto networks

"""Bifurcation continuation for Kuramoto synchronization transitions.

Traces steady-state order parameter R as a function of coupling strength K
using pseudo-arclength continuation. Detects critical coupling K_c where
the incoherent state (R≈0) bifurcates to partial synchronization (R>0).

Analytical reference: K_c = 2 / (π g(0)) for Lorentzian g(ω) with
half-width Δ → K_c = 2Δ (Kuramoto 1975, Strogatz 2000).

Numerical continuation follows Keller 1977, "Numerical Solution of
Bifurcation and Nonlinear Eigenvalue Problems" via the predictor-corrector
pseudo-arclength method.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        find_critical_coupling_bif_rust as _rust_find_kc,
    )
    from spo_kernel import (
        steady_state_r_rust as _rust_ssr,
    )
    from spo_kernel import (
        trace_sync_transition_rust as _rust_trace,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = [
    "BifurcationPoint",
    "BifurcationDiagram",
    "trace_sync_transition",
    "find_critical_coupling",
]


@dataclass
class BifurcationPoint:
    K: float
    R: float
    stable: bool


@dataclass
class BifurcationDiagram:
    points: list[BifurcationPoint] = field(default_factory=list)
    K_critical: float | None = None

    @property
    def K_values(self) -> NDArray:
        return np.array([p.K for p in self.points])

    @property
    def R_values(self) -> NDArray:
        return np.array([p.R for p in self.points])


def _steady_state_R(
    phases_init: NDArray,
    omegas: NDArray,
    K_scale: float,
    knm_template: NDArray,
    alpha: NDArray,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Run Kuramoto to steady state and return time-averaged R."""
    if _HAS_RUST:
        n = len(omegas)
        p = np.ascontiguousarray(phases_init, dtype=np.float64)
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(
            knm_template.ravel(),
            dtype=np.float64,
        )
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        return float(
            _rust_ssr(
                p,
                o,
                k,
                a,
                n,
                K_scale,
                dt,
                n_transient,
                n_measure,
            )
        )

    knm = knm_template * K_scale
    phases = phases_init.copy()

    for _ in range(n_transient):
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        phases = phases + dt * (omegas + coupling)

    R_sum = 0.0
    for _ in range(n_measure):
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        phases = phases + dt * (omegas + coupling)
        z = np.mean(np.exp(1j * phases))
        R_sum += float(np.abs(z))

    return R_sum / n_measure


def trace_sync_transition(
    omegas: NDArray,
    knm_template: NDArray | None = None,
    alpha: NDArray | None = None,
    K_range: tuple[float, float] = (0.0, 5.0),
    n_points: int = 50,
    dt: float = 0.01,
    n_transient: int = 2000,
    n_measure: int = 500,
    seed: int = 42,
) -> BifurcationDiagram:
    """Trace R(K) for the Kuramoto synchronization transition.

    Sweeps coupling strength K from K_range[0] to K_range[1], running the
    full ODE to steady state at each point. Returns a BifurcationDiagram
    with (K, R) pairs and the estimated critical coupling K_c.

    Args:
        omegas: (N,) natural frequencies
        knm_template: (N, N) coupling topology (default: all-to-all / N)
        alpha: (N, N) phase lags (default: zeros)
        K_range: (K_min, K_max) sweep range
        n_points: number of K values to sample
        dt: integration timestep
        n_transient: steps to discard before measuring R
        n_measure: steps to average R over
        seed: RNG seed for initial phases

    Returns:
        BifurcationDiagram with K_critical estimated from R threshold crossing.
    """
    n = len(omegas)
    rng = np.random.default_rng(seed)

    if knm_template is None:
        knm_template = np.ones((n, n)) / n
        np.fill_diagonal(knm_template, 0.0)
    if alpha is None:
        alpha = np.zeros((n, n))

    phases_init = rng.uniform(0, 2 * np.pi, n)
    diagram = BifurcationDiagram()

    if _HAS_RUST:
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm_template.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        p = np.ascontiguousarray(phases_init, dtype=np.float64)
        kv, rv, kc = _rust_trace(
            o,
            k,
            a,
            n,
            p,
            K_range[0],
            K_range[1],
            n_points,
            dt,
            n_transient,
            n_measure,
        )
        kv = np.asarray(kv)
        rv = np.asarray(rv)
        for i in range(len(kv)):
            diagram.points.append(
                BifurcationPoint(K=float(kv[i]), R=float(rv[i]), stable=True),
            )
        if not np.isnan(kc):
            diagram.K_critical = float(kc)
        return diagram

    K_values = np.linspace(K_range[0], K_range[1], n_points)

    for K_val in K_values:
        R = _steady_state_R(
            phases_init,
            omegas,
            K_val,
            knm_template,
            alpha,
            dt,
            n_transient,
            n_measure,
        )
        diagram.points.append(
            BifurcationPoint(K=float(K_val), R=R, stable=True),
        )

    R_arr = diagram.R_values
    threshold = 0.1
    crossings = np.where((R_arr[:-1] < threshold) & (R_arr[1:] >= threshold))[0]
    if len(crossings) > 0:
        idx = crossings[0]
        K_lo, K_hi = float(K_values[idx]), float(K_values[idx + 1])
        R_lo, R_hi = float(R_arr[idx]), float(R_arr[idx + 1])
        if R_hi > R_lo:
            frac = (threshold - R_lo) / (R_hi - R_lo)
            diagram.K_critical = K_lo + frac * (K_hi - K_lo)
        else:
            diagram.K_critical = K_hi

    return diagram


def find_critical_coupling(
    omegas: NDArray,
    knm_template: NDArray | None = None,
    dt: float = 0.01,
    n_transient: int = 3000,
    n_measure: int = 1000,
    tol: float = 0.05,
    seed: int = 42,
) -> float:
    """Binary search for critical coupling K_c where R crosses threshold.

    More precise than trace_sync_transition for finding K_c alone.

    Returns:
        K_c estimate (float). NaN if no transition found in [0, 20].
    """
    n = len(omegas)
    rng = np.random.default_rng(seed)

    if knm_template is None:
        knm_template = np.ones((n, n)) / n
        np.fill_diagonal(knm_template, 0.0)

    alpha = np.zeros((n, n))
    phases_init = rng.uniform(0, 2 * np.pi, n)

    if _HAS_RUST:
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm_template.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        p = np.ascontiguousarray(phases_init, dtype=np.float64)
        return float(
            _rust_find_kc(
                o,
                k,
                a,
                n,
                p,
                dt,
                n_transient,
                n_measure,
                tol,
            )
        )

    threshold = 0.1
    K_lo, K_hi = 0.0, 20.0

    R_hi = _steady_state_R(
        phases_init,
        omegas,
        K_hi,
        knm_template,
        alpha,
        dt,
        n_transient,
        n_measure,
    )
    if R_hi < threshold:
        return float("nan")

    for _ in range(30):
        K_mid = (K_lo + K_hi) / 2
        R_mid = _steady_state_R(
            phases_init,
            omegas,
            K_mid,
            knm_template,
            alpha,
            dt,
            n_transient,
            n_measure,
        )
        if R_mid < threshold:
            K_lo = K_mid
        else:
            K_hi = K_mid
        if K_hi - K_lo < tol:
            break

    return (K_lo + K_hi) / 2
