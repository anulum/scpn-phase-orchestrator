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
from numbers import Integral, Real
from typing import TypeAlias

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
FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in arr.flat)


@dataclass
class BifurcationPoint:
    """One sampled point on a Kuramoto synchronisation branch."""

    K: float
    R: float
    stable: bool

    def __post_init__(self) -> None:
        k_value = _validate_finite_float(self.K, name="K")
        if k_value < 0.0:
            raise ValueError(f"K must be non-negative, got {self.K!r}")
        r_value = _validate_unit_interval(self.R, name="R")
        if not isinstance(self.stable, bool):
            raise ValueError(f"stable must be a boolean flag, got {self.stable!r}")

        self.K = k_value
        self.R = r_value


@dataclass
class BifurcationDiagram:
    """Ordered bifurcation samples plus optional critical coupling."""

    points: list[BifurcationPoint] = field(default_factory=list)
    K_critical: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.points, list):
            raise ValueError("points must be a list of BifurcationPoint records")
        for idx, point in enumerate(self.points):
            if not isinstance(point, BifurcationPoint):
                raise ValueError(
                    f"points[{idx}] must be a BifurcationPoint, got {point!r}"
                )
        if self.K_critical is not None:
            k_critical = _validate_finite_float(self.K_critical, name="K_critical")
            if k_critical < 0.0:
                raise ValueError(
                    f"K_critical must be non-negative, got {self.K_critical!r}"
                )
            self.K_critical = k_critical

    @property
    def K_values(self) -> FloatArray:
        """Return continuation coupling strengths in diagram order."""

        return np.array([p.K for p in self.points])

    @property
    def R_values(self) -> FloatArray:
        """Return continuation order parameters in diagram order."""

        return np.array([p.R for p in self.points])


def _validate_integral(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_positive_float(value: object, *, name: str) -> float:
    coerced = _validate_finite_float(value, name=name)
    if coerced <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return coerced


def _validate_unit_interval(value: object, *, name: str) -> float:
    coerced = _validate_finite_float(value, name=name)
    if coerced < 0.0 or coerced > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return coerced


def _validate_omegas(value: object) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError("omegas must not contain boolean values")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("omegas must be a finite one-dimensional array") from exc
    if arr.ndim != 1:
        raise ValueError(f"omegas shape {arr.shape} must be one-dimensional")
    if arr.size < 1:
        raise ValueError("omegas must contain at least one oscillator")
    if not np.all(np.isfinite(arr)):
        raise ValueError("omegas must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_matrix(
    value: object,
    *,
    name: str,
    n: int,
    require_zero_diagonal: bool = False,
) -> FloatArray:
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float matrix") from exc
    if arr.shape != (n, n):
        raise ValueError(f"{name} shape {arr.shape} does not match ({n}, {n})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if require_zero_diagonal and not np.allclose(np.diag(arr), 0.0, atol=1e-12):
        raise ValueError(
            f"{name} diagonal must be zero; self-coupling K_ii is not physical"
        )
    return np.ascontiguousarray(arr, dtype=np.float64)


def _default_coupling(n: int) -> FloatArray:
    knm_template = np.ones((n, n), dtype=np.float64) / n
    np.fill_diagonal(knm_template, 0.0)
    return knm_template


def _validate_k_range(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("K_range must contain exactly two finite values")
    start, stop = value
    start = _validate_finite_float(start, name="K_range")
    stop = _validate_finite_float(stop, name="K_range")
    if start < 0.0:
        raise ValueError("K_range start must be non-negative")
    if stop <= start:
        raise ValueError("K_range stop must be greater than start")
    return start, stop


def _steady_state_R_dispatch(
    phases_init: FloatArray,
    omegas: FloatArray,
    K_scale: float,
    knm_template: FloatArray,
    alpha: FloatArray,
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
        phases_init,
        omegas,
        knm_template,
        alpha=alpha,
        k_scale=K_scale,
        dt=dt,
        n_transient=n_transient,
        n_measure=n_measure,
    )


def trace_sync_transition(
    omegas: FloatArray,
    knm_template: FloatArray | None = None,
    alpha: FloatArray | None = None,
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
    omegas = _validate_omegas(omegas)
    n = int(omegas.shape[0])
    K_range = _validate_k_range(K_range)
    n_points = _validate_integral(n_points, name="n_points", minimum=2)
    dt = _validate_positive_float(dt, name="dt")
    n_transient = _validate_integral(n_transient, name="n_transient", minimum=0)
    n_measure = _validate_integral(n_measure, name="n_measure", minimum=0)
    seed = _validate_integral(seed, name="seed", minimum=0)

    if knm_template is None:
        knm_template = _default_coupling(n)
    else:
        knm_template = _validate_matrix(
            knm_template,
            name="knm_template",
            n=n,
            require_zero_diagonal=True,
        )
    if alpha is None:
        alpha = np.zeros((n, n), dtype=np.float64)
    else:
        alpha = _validate_matrix(alpha, name="alpha", n=n)

    rng = np.random.default_rng(seed)
    phases_init = rng.uniform(0, 2 * np.pi, n)
    diagram = BifurcationDiagram()

    if _HAS_COMPOSITE_RUST:
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
                BifurcationPoint(
                    K=float(kv[i]),
                    R=float(rv[i]),
                    stable=True,
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
    omegas: FloatArray,
    knm_template: FloatArray | None = None,
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
    omegas = _validate_omegas(omegas)
    n = int(omegas.shape[0])
    dt = _validate_positive_float(dt, name="dt")
    n_transient = _validate_integral(n_transient, name="n_transient", minimum=0)
    n_measure = _validate_integral(n_measure, name="n_measure", minimum=0)
    tol = _validate_positive_float(tol, name="tol")
    seed = _validate_integral(seed, name="seed", minimum=0)

    if knm_template is None:
        knm_template = _default_coupling(n)
    else:
        knm_template = _validate_matrix(
            knm_template,
            name="knm_template",
            n=n,
            require_zero_diagonal=True,
        )

    alpha = np.zeros((n, n))
    rng = np.random.default_rng(seed)
    phases_init = rng.uniform(0, 2 * np.pi, n)

    if _HAS_COMPOSITE_RUST:
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
            ),
        )

    threshold = 0.1
    K_lo, K_hi = 0.0, 20.0

    R_hi = _steady_state_R_dispatch(
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
        R_mid = _steady_state_R_dispatch(
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
