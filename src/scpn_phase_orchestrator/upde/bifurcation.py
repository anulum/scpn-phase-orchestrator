# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bifurcation analysis for Kuramoto networks

"""Independent finite-horizon coupling sweeps and threshold searches.

Every grid point starts from the same NumPy-seeded phases and uses explicit Euler.
This is not pseudo-arclength continuation. The reported crossing is the first
sampled upcrossing of R=0.1; binary search assumes a monotone response on [0,20].
Neither crossing certifies a dynamical bifurcation. The classical continuum
Kuramoto critical-coupling formula is not a finite-network acceptance oracle.

Automatic dispatch uses the batched Rust composites when installed, otherwise
the basin trial preference chain. Named owners forbid fallback. The historical
``stable=True`` field is a compatibility marker, not a measured stability result.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Complex, Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.basin_stability import (
    steady_state_r as _dispatched_steady_state_r,
)

FloatArray: TypeAlias = NDArray[np.float64]
NativeTraceKernel: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        FloatArray,
        float,
        float,
        int,
        float,
        int,
        int,
    ],
    tuple[object, object, object],
]
NativeSearchKernel: TypeAlias = Callable[
    [FloatArray, FloatArray, FloatArray, int, FloatArray, float, int, int, float],
    object,
]

try:
    _native_module = importlib.import_module("spo_kernel")
    _trace_function = getattr(_native_module, "trace_sync_transition_rust", None)
    _search_function = getattr(_native_module, "find_critical_coupling_bif_rust", None)
    if not callable(_trace_function) or not callable(_search_function):
        raise ImportError("Rust kernel lacks the composite coupling entry points")
    _rust_trace = cast("NativeTraceKernel", _trace_function)
    _rust_find_kc = cast("NativeSearchKernel", _search_function)
    _HAS_COMPOSITE_RUST = True
except ImportError:
    _HAS_COMPOSITE_RUST = False

__all__ = [
    "BifurcationDiagram",
    "BifurcationPoint",
    "find_critical_coupling",
    "trace_sync_transition",
]


def _as_real_numeric_array(value: object, *, name: str) -> FloatArray:
    """Return a real numeric array without coercing string or complex aliases."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    object_values = raw.dtype == np.object_
    if raw.dtype == np.bool_ or (
        object_values and any(isinstance(item, (bool, np.bool_)) for item in raw.flat)
    ):
        raise ValueError(f"{name} must be real-valued, not boolean")
    if np.iscomplexobj(raw) or (
        object_values
        and any(
            isinstance(item, Complex) and not isinstance(item, Real)
            for item in raw.flat
        )
    ):
        raise ValueError(f"{name} must be real-valued, not complex")
    numeric_object = object_values and all(isinstance(item, Real) for item in raw.flat)
    if not np.issubdtype(raw.dtype, np.number) and not numeric_object:
        raise ValueError(f"{name} must be numeric")
    try:
        return np.ascontiguousarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


@dataclass
class BifurcationPoint:
    """One independent finite-window coupling sample.

    Attributes
    ----------
    K : float
        Sampled nonnegative coupling multiplier.
    R : float
        Mean post-step order parameter in the finite measurement window.
    stable : bool
        Historical compatibility flag. Generated samples set this to True;
        it is not a measured stability result or a stability certificate.
    """

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
        """Return sampled coupling strengths in diagram order.

        Returns
        -------
        FloatArray
            Return sampled coupling strengths in diagram order.
        """
        return np.array([p.K for p in self.points])

    @property
    def R_values(self) -> FloatArray:
        """Return finite-window order parameters in diagram order.

        Returns
        -------
        FloatArray
            Return finite-window order parameters in diagram order.
        """
        return np.array([p.R for p in self.points])


def _validate_integral(value: object, *, name: str, minimum: int) -> int:
    """Return ``value`` as a validated integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    coerced = _validate_finite_float(value, name=name)
    if coerced <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return coerced


def _validate_unit_interval(value: object, *, name: str) -> float:
    """Return ``value`` as a float in [0, 1], else raise ``ValueError``."""
    coerced = _validate_finite_float(value, name=name)
    if coerced < 0.0 or coerced > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return coerced


def _validate_omegas(value: object) -> FloatArray:
    """Return the natural frequencies as a validated finite array, else raise."""
    arr = _as_real_numeric_array(value, name="omegas")
    if arr.ndim != 1:
        raise ValueError(f"omegas shape {arr.shape} must be one-dimensional")
    if arr.size < 1:
        raise ValueError("omegas must contain at least one oscillator")
    if not np.all(np.isfinite(arr)):
        raise ValueError("omegas must contain only finite values")
    return arr


def _validate_matrix(
    value: object,
    *,
    name: str,
    n: int,
    require_zero_diagonal: bool = False,
) -> FloatArray:
    """Return the value as a validated finite matrix, else raise."""
    arr = _as_real_numeric_array(value, name=name)
    if arr.shape != (n, n):
        raise ValueError(f"{name} shape {arr.shape} does not match ({n}, {n})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    if require_zero_diagonal and not np.allclose(np.diag(arr), 0.0, atol=1e-12):
        raise ValueError(
            f"{name} diagonal must be zero; self-coupling K_ii is not physical"
        )
    return arr


def _default_coupling(n: int) -> FloatArray:
    """Return the default all-to-all coupling matrix for the size."""
    knm_template = np.ones((n, n), dtype=np.float64) / n
    np.fill_diagonal(knm_template, 0.0)
    return knm_template


def _validate_k_range(value: object) -> tuple[float, float]:
    """Return the validated coupling-strength sweep range, else raise."""
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


def _validate_rust_trace_result(
    K_values: object,
    R_values: object,
    *,
    n_points: int,
    K_range: tuple[float, float],
) -> tuple[FloatArray, FloatArray]:
    """Return a Rust independent-grid trace matching the reference, else raise."""
    k_arr = _as_real_numeric_array(
        K_values,
        name="Rust bifurcation trace K values",
    )
    r_arr = _as_real_numeric_array(
        R_values,
        name="Rust bifurcation trace R values",
    )
    if k_arr.shape != (n_points,) or r_arr.shape != (n_points,):
        raise ValueError(
            "Rust bifurcation trace returned arrays with unexpected shape "
            f"{k_arr.shape} and {r_arr.shape}; expected ({n_points},)"
        )
    if not np.all(np.isfinite(k_arr)) or not np.all(np.isfinite(r_arr)):
        raise ValueError("Rust bifurcation trace returned non-finite values")
    if np.any(r_arr < 0.0) or np.any(r_arr > 1.0):
        raise ValueError("Rust bifurcation trace returned R outside [0, 1]")
    if np.any(np.diff(k_arr) < -1e-12):
        raise ValueError("Rust bifurcation trace returned non-monotone K values")
    start, stop = K_range
    if np.any(k_arr < start - 1e-12) or np.any(k_arr > stop + 1e-12):
        raise ValueError("Rust bifurcation trace returned K outside K_range")
    expected = np.linspace(start, stop, n_points)
    if not np.allclose(k_arr, expected, rtol=0.0, atol=1e-12):
        raise ValueError("Rust bifurcation trace returned a different coupling grid")
    return k_arr, r_arr


def _first_upcrossing(k_values: FloatArray, r_values: FloatArray) -> float | None:
    """Interpolate the first sampled R=0.1 upcrossing, or return no crossing."""
    crossings = np.flatnonzero((r_values[:-1] < 0.1) & (r_values[1:] >= 0.1))
    if crossings.size == 0:
        return None
    index = int(crossings[0])
    fraction = (0.1 - r_values[index]) / (r_values[index + 1] - r_values[index])
    return float(k_values[index] + fraction * (k_values[index + 1] - k_values[index]))


def _validate_optional_critical_coupling(value: object) -> float | None:
    """Return the optional validated critical-coupling value, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Rust bifurcation trace returned invalid K_critical")
    critical = float(value)
    if np.isnan(critical):
        return None
    if not np.isfinite(critical) or critical < 0.0:
        raise ValueError("Rust bifurcation trace returned invalid K_critical")
    return critical


def _validate_find_critical_coupling_result(value: object) -> float:
    """Return the validated critical-coupling search result, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Rust critical-coupling search returned invalid K_c")
    critical = float(value)
    if np.isnan(critical):
        return float("nan")
    if not np.isfinite(critical) or critical < 0.0:
        raise ValueError("Rust critical-coupling search returned invalid K_c")
    return critical


def _steady_state_R_dispatch(
    phases_init: FloatArray,
    omegas: FloatArray,
    K_scale: float,
    knm_template: FloatArray,
    alpha: FloatArray,
    dt: float,
    n_transient: int,
    n_measure: int,
    backend: str | None = None,
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
        backend=backend,
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
    *,
    backend: str | None = None,
) -> BifurcationDiagram:
    """Trace R(K) for the Kuramoto synchronisation transition.

    Sweeps coupling strength ``K`` from ``K_range[0]`` to
    ``K_range[1]``, independently measuring a finite window at each point,
    and returns a :class:`BifurcationDiagram` with the ``(K, R)``
    pairs plus the estimated critical coupling ``K_c``.

    When the Rust composite kernel is available, the whole sweep
    is batched into a single FFI call. Otherwise the function
    loops in Python and each trial is dispatched through the
    5-backend chain inherited from
    :func:`basin_stability.steady_state_r`.

    Parameters
    ----------
    omegas : FloatArray
        Finite real numeric natural frequencies in rad/s, shape ``(N,)``.
        Boolean, complex, and numeric-string aliases are rejected.
    knm_template : FloatArray | None
        Unit coupling template scaled across the grid, or ``None`` for
        all-to-all. Must be finite, real numeric, and zero-diagonal.
    alpha : FloatArray | None
        Finite real numeric phase-lag matrix in radians, shape ``(N, N)``, or
        ``None`` for no lag.
    K_range : tuple[float, float]
        Inclusive ``(min, max)`` coupling-strength range to scan.
    n_points : int
        Number of coupling points sampled across the range.
    dt : float
        Integration step size.
    n_transient : int
        Number of transient steps discarded before measurement.
    n_measure : int
        Number of steps averaged to measure the order parameter.
    seed : int
        Seed for the deterministic RNG.
    backend : str | None
        Named numerical owner, or automatic batched Rust / trial preference.
        An unavailable named owner raises ImportError without fallback.

    Returns
    -------
    BifurcationDiagram
        The traced ``R(K)`` bifurcation diagram.

    Raises
    ------
    ValueError
        If measurements, controls, or numerical owner outputs are invalid.
    ImportError
        If the explicitly requested numerical owner is unavailable.
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

    if backend is not None and backend not in ("rust", "mojo", "julia", "go", "python"):
        raise ValueError(f"unknown basin backend: {backend!r}")
    if _HAS_COMPOSITE_RUST and backend in (None, "rust"):
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
        kv, rv = _validate_rust_trace_result(
            kv,
            rv,
            n_points=n_points,
            K_range=K_range,
        )
        critical = _validate_optional_critical_coupling(kc)
        expected_critical = _first_upcrossing(kv, rv)
        if (critical is None) != (expected_critical is None):
            raise ValueError("Rust K_critical disagrees with the sampled upcrossing")
        if (
            critical is not None
            and expected_critical is not None
            and not np.isclose(critical, expected_critical, rtol=1e-12, atol=1e-12)
        ):
            raise ValueError("Rust K_critical disagrees with the sampled interpolation")
        for i in range(len(kv)):
            diagram.points.append(
                BifurcationPoint(
                    K=float(kv[i]),
                    R=float(rv[i]),
                    stable=True,
                ),
            )
        if critical is not None:
            diagram.K_critical = critical
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
            backend,
        )
        diagram.points.append(
            BifurcationPoint(K=float(K_val), R=R, stable=True),
        )

    diagram.K_critical = _first_upcrossing(K_values, diagram.R_values)
    return diagram


def find_critical_coupling(
    omegas: FloatArray,
    knm_template: FloatArray | None = None,
    dt: float = 0.01,
    n_transient: int = 3000,
    n_measure: int = 1000,
    tol: float = 0.05,
    seed: int = 42,
    *,
    backend: str | None = None,
) -> float:
    """Bisect a finite-window R=0.1 classification response on [0,20].

    Assumes a monotone finite-horizon R response; it is not a stability test.
    Returns NaN only when the upper endpoint has R below 0.1. The lower
    endpoint is not measured. If its response is already above threshold,
    bisection can return a small positive lower-bracket midpoint without
    any transition. This historical interval-return convention is retained.

    Parameters
    ----------
    omegas : FloatArray
        Finite real numeric natural frequencies in rad/s, shape ``(N,)``.
        Boolean, complex, and numeric-string aliases are rejected.
    knm_template : FloatArray | None
        Unit coupling template scaled across the grid, or ``None`` for
        all-to-all. Must be finite, real numeric, and zero-diagonal.
    dt : float
        Integration step size.
    n_transient : int
        Number of transient steps discarded before measurement.
    n_measure : int
        Number of steps averaged to measure the order parameter.
    tol : float
        Positive stopping tolerance for the coupling interval width.
    seed : int
        Seed for the deterministic RNG.
    backend : str | None
        Named numerical owner, or automatic batched Rust / trial preference.
        An unavailable named owner raises ImportError without fallback.

    Returns
    -------
    float
        Final interval midpoint, or NaN when R at K=20 is below 0.1.
        A midpoint is not evidence that the lower endpoint was subthreshold
        or that any physical transition occurred.

    Raises
    ------
    ValueError
        If measurements, controls, or numerical owner outputs are invalid.
    ImportError
        If the explicitly requested numerical owner is unavailable.
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

    if backend is not None and backend not in ("rust", "mojo", "julia", "go", "python"):
        raise ValueError(f"unknown basin backend: {backend!r}")
    if _HAS_COMPOSITE_RUST and backend in (None, "rust"):
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        k = np.ascontiguousarray(knm_template.ravel(), dtype=np.float64)
        a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
        p = np.ascontiguousarray(phases_init, dtype=np.float64)
        return _validate_find_critical_coupling_result(
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
        backend,
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
            backend,
        )
        if R_mid < threshold:
            K_lo = K_mid
        else:
            K_hi = K_mid
        if K_hi - K_lo < tol:
            break

    return (K_lo + K_hi) / 2
