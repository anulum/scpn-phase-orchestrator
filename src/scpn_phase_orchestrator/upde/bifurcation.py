# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bifurcation analysis for Kuramoto networks

"""Bifurcation continuation for Kuramoto synchronisation transitions.

Traces steady-state order parameter ``R`` as a function of coupling
strength ``K`` using pseudo-arclength continuation (Keller 1977).
Detects critical coupling ``K_c`` where the incoherent state
``R ≈ 0`` bifurcates to partial synchronisation ``R > 0``.

Analytical reference: ``K_c = 2 / (π g(0))`` for Lorentzian ``g(ω)``
with half-width ``Δ`` → ``K_c = 2Δ`` (Kuramoto 1975, Strogatz 2000).

5-backend chain via delegation
------------------------------
The single-trial kernel ``steady_state_r(phases, omegas, knm,
alpha, k_scale, dt, n_transient, n_measure) → R`` is already
dispatched across Rust / Mojo / Julia / Go / Python in
:mod:`scpn_phase_orchestrator.upde.basin_stability`. This module
delegates to it rather than re-implementing the Euler trial
integrator, which means every ``trace_sync_transition`` /
``find_critical_coupling`` call in the Python-composite branch
inherits the full fallback chain for free.

The two composite Rust kernels — ``trace_sync_transition_rust``
(batched K-sweep) and ``find_critical_coupling_bif_rust`` (binary
search inside Rust) — are preserved as one-shot fast paths: a
single FFI call amortises the per-K boundary overhead better than
the N_points × dispatch-call path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.basin_stability import (
    steady_state_r as _dispatched_steady_state_r,
)

try:
    from spo_kernel import (
        find_critical_coupling_bif_rust as _rust_find_kc,
    )
    from spo_kernel import (
        trace_sync_transition_rust as _rust_trace,
    )

    _HAS_COMPOSITE_RUST = True
except ImportError:
    _HAS_COMPOSITE_RUST = False

__all__ = [
    "BifurcationDiagram",
    "BifurcationPoint",
    "find_critical_coupling",
    "trace_sync_transition",
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


def _steady_state_R_dispatch(
    phases_init: NDArray,
    omegas: NDArray,
    K_scale: float,
    knm_template: NDArray,
    alpha: NDArray,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Thin wrapper around the 5-backend-dispatched kernel.

    Shipped as a module-private helper so the ``trace_*`` /
    ``find_*`` functions stay Python-only at the Python level;
    the multi-language work happens inside
    ``basin_stability.steady_state_r``.
    """
    return _dispatched_steady_state_r(
        phases_init, omegas, knm_template, alpha=alpha,
        k_scale=K_scale, dt=dt,
        n_transient=n_transient, n_measure=n_measure,
    )


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
    """Trace R(K) for the Kuramoto synchronisation transition.

    Sweeps coupling strength ``K`` from ``K_range[0]`` to
    ``K_range[1]``, running the ODE to steady state at each point,
    and returns a :class:`BifurcationDiagram` with the ``(K, R)``
    pairs plus the estimated critical coupling ``K_c``.

    When the Rust composite kernel is available, the whole sweep
    is batched into a single FFI call. Otherwise the function
    loops in Python and each trial is dispatched through the
    5-backend chain inherited from
    :func:`basin_stability.steady_state_r`.
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

    if _HAS_COMPOSITE_RUST:
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm_template.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        p = np.ascontiguousarray(phases_init, dtype=np.float64)
        kv, rv, kc = _rust_trace(
            o, k, a, n, p,
            K_range[0], K_range[1], n_points,
            dt, n_transient, n_measure,
        )
        kv = np.asarray(kv)
        rv = np.asarray(rv)
        for i in range(len(kv)):
            diagram.points.append(
                BifurcationPoint(
                    K=float(kv[i]), R=float(rv[i]), stable=True,
                ),
            )
        if not np.isnan(kc):
            diagram.K_critical = float(kc)
        return diagram

    # Composite Rust unavailable — loop in Python, each trial
    # dispatched through the basin_stability 5-backend chain.
    K_values = np.linspace(K_range[0], K_range[1], n_points)
    for K_val in K_values:
        R = _steady_state_R_dispatch(
            phases_init, omegas, K_val, knm_template, alpha,
            dt, n_transient, n_measure,
        )
        diagram.points.append(
            BifurcationPoint(K=float(K_val), R=R, stable=True),
        )

    R_arr = diagram.R_values
    threshold = 0.1
    crossings = np.where(
        (R_arr[:-1] < threshold) & (R_arr[1:] >= threshold),
    )[0]
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
    """Binary search for the critical coupling ``K_c`` at which
    ``R`` crosses a 0.1 threshold.

    More precise than :func:`trace_sync_transition` when only
    ``K_c`` is needed. Returns ``nan`` if no transition is found
    in ``[0, 20]``.
    """
    n = len(omegas)
    rng = np.random.default_rng(seed)

    if knm_template is None:
        knm_template = np.ones((n, n)) / n
        np.fill_diagonal(knm_template, 0.0)

    alpha = np.zeros((n, n))
    phases_init = rng.uniform(0, 2 * np.pi, n)

    if _HAS_COMPOSITE_RUST:
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm_template.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        p = np.ascontiguousarray(phases_init, dtype=np.float64)
        return float(
            _rust_find_kc(
                o, k, a, n, p, dt, n_transient, n_measure, tol,
            ),
        )

    threshold = 0.1
    K_lo, K_hi = 0.0, 20.0

    R_hi = _steady_state_R_dispatch(
        phases_init, omegas, K_hi, knm_template, alpha,
        dt, n_transient, n_measure,
    )
    if R_hi < threshold:
        return float("nan")

    for _ in range(30):
        K_mid = (K_lo + K_hi) / 2
        R_mid = _steady_state_R_dispatch(
            phases_init, omegas, K_mid, knm_template, alpha,
            dt, n_transient, n_measure,
        )
        if R_mid < threshold:
            K_lo = K_mid
        else:
            K_hi = K_mid
        if K_hi - K_lo < tol:
            break

    return (K_lo + K_hi) / 2
