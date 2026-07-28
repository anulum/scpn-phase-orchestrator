# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stochastic noise injection and optimal D*

"""Stochastic noise injection and noise-level sweeps for UPDE phase dynamics.

``StochasticInjector`` owns a local random generator and applies
Euler-Maruyama phase noise under validated non-negative diffusion and positive
time-step parameters. ``find_optimal_noise`` sweeps finite non-negative
candidate noise levels against a supplied UPDE engine and reports the best
coherence profile without changing the engine configuration or caller-provided
input arrays outside normal engine stepping.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Complex, Integral, Real
from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import i0, i1

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

if TYPE_CHECKING:
    from scpn_phase_orchestrator.upde.engine import UPDEEngine

__all__ = ["StochasticInjector", "NoiseProfile", "find_optimal_noise"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class NoiseProfile:
    """Validated noise-sweep result linking diffusion to bounded order."""

    D: float
    R_achieved: float
    R_deterministic: float

    def __post_init__(self) -> None:
        self.D = _validate_finite_non_negative(self.D, name="D")
        self.R_achieved = _validate_unit_interval(
            self.R_achieved,
            name="R_achieved",
        )
        self.R_deterministic = _validate_unit_interval(
            self.R_deterministic,
            name="R_deterministic",
        )


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


def _validate_finite_non_negative(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real, got {value!r}")
    value = float(value)
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real, got {value!r}")
    return value


def _validate_finite_positive(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    return value


def _validate_unit_interval(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real in ``[0, 1]``, else raise."""
    value = _validate_finite_non_negative(value, name=name)
    if value > 1.0:
        raise ValueError(f"{name} must be a finite real in [0, 1], got {value!r}")
    return value


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_seed(value: object, *, name: str = "seed") -> int:
    """Return a non-negative integer RNG seed, else raise ``ValueError``."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _validate_optional_seed(value: object) -> int | None:
    """Return ``None`` or a validated non-negative integer RNG seed."""
    if value is None:
        return None
    return _validate_seed(value)


def _validate_phases(value: object) -> FloatArray:
    """Return a finite one-dimensional real numeric phase array."""
    phases = _as_real_numeric_array(value, name="phases")
    if phases.ndim != 1:
        raise ValueError(f"phases shape {phases.shape} must be one-dimensional")
    if not np.all(np.isfinite(phases)):
        raise ValueError("phases must contain only finite values")
    return phases


def _validate_noise_range(value: FloatArray | None) -> FloatArray | None:
    """Return the validated ``(min, max)`` noise-level range, else raise."""
    if value is None:
        return None
    d_range = _as_real_numeric_array(value, name="D_range")
    if d_range.ndim != 1 or d_range.size == 0:
        raise ValueError("D_range must be a non-empty 1-D array")
    if not np.all(np.isfinite(d_range)) or np.any(d_range < 0.0):
        raise ValueError("D_range must contain only finite non-negative values")
    return d_range


class StochasticInjector:
    """Add calibrated noise to phase dynamics.

    Euler-Maruyama: θ_i(t+dt) = θ_i(t) + f(θ)*dt + √(2D*dt) * ξ_i
    where ξ_i ~ N(0,1) i.i.d.

    Tselios et al. 2025 — stochastic resonance in Kuramoto networks.
    """

    def __init__(self, D: float, seed: int | None = None):
        """Create an injector with finite ``D`` and an optional valid seed."""
        self._D = _validate_finite_non_negative(D, name="D")
        self._rng = np.random.default_rng(_validate_optional_seed(seed))

    @property
    def D(self) -> float:
        """Return the configured non-negative diffusion coefficient.

        Returns
        -------
        float
            Return the configured non-negative diffusion coefficient.
        """
        return self._D

    @D.setter
    def D(self, value: float) -> None:
        """Update the diffusion coefficient after finite non-negative validation.

        Parameters
        ----------
        value : float
            The new value to set.
        """
        self._D = _validate_finite_non_negative(value, name="D")

    def inject(self, phases: FloatArray, dt: float) -> FloatArray:
        """Add Wiener noise to phases: θ += √(2D*dt) * N(0,1).

        Parameters
        ----------
        phases : FloatArray
            Finite real numeric oscillator phases in radians, shape ``(N,)``.
            Boolean, complex, and numeric-string aliases are rejected.
        dt : float
            Integration step size.

        Returns
        -------
        FloatArray
            The phases with added Wiener noise.
        """
        dt = _validate_finite_positive(dt, name="dt")
        phases = _validate_phases(phases)
        if self._D == 0.0:
            return phases
        noise = self._rng.standard_normal(len(phases))
        result: FloatArray = (phases + np.sqrt(2.0 * self._D * dt) * noise) % TWO_PI
        return result


def _self_consistency_R(K: float, D: float) -> float:
    """Solve R = I₁(KR/D) / I₀(KR/D) self-consistency for R.

    Acebrón et al. 2005, Rev. Mod. Phys. 77(1):137-185, Eq. (12).
    """
    if D < 1e-15:
        return 1.0 if K > 0 else 0.0
    if K < 1e-15:
        return 0.0
    R = 0.5
    for _ in range(100):
        x = K * R / D
        R_new = 1.0 - 0.5 / x if x > 500 else float(i1(x) / i0(x))
        if abs(R_new - R) < 1e-10:
            return R_new
        R = 0.7 * R_new + 0.3 * R
    return R


def optimal_D(K: float, R_det: float) -> float:
    """Estimate optimal noise for stochastic resonance.

    D* ≈ K·R_det/2 (common noise case).
    Tselios et al. 2025.
    """
    return K * R_det / 2.0


def find_optimal_noise(
    engine: UPDEEngine,
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    D_range: FloatArray | None = None,
    n_steps: int = 500,
    seed: int = 42,
) -> NoiseProfile:
    """Sweep noise levels, return D that maximizes R.

    Uses the engine to simulate n_steps at each D value.

    Parameters
    ----------
    engine : UPDEEngine
        The UPDE engine used to integrate each trial.
    phases_init : FloatArray
        Initial oscillator phases in radians, shape ``(N,)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    D_range : FloatArray | None
        Finite non-negative real numeric diffusion coefficients to sweep, or
        ``None`` for the default range. Coercive aliases are rejected.
    n_steps : int
        Number of integration steps to run.
    seed : int
        Non-negative non-boolean seed for the deterministic RNG.

    Returns
    -------
    NoiseProfile
        The noise profile whose diffusion ``D`` maximises ``R``.
    """
    n_steps = _validate_positive_int(n_steps, name="n_steps")
    seed = _validate_seed(seed)
    D_range = _validate_noise_range(D_range)
    if D_range is None:
        K_mean = float(np.mean(knm[knm > 0])) if np.any(knm > 0) else 1.0
        D_range = np.linspace(0.0, K_mean, 11, dtype=np.float64)

    best_D = 0.0
    best_R = 0.0
    R_det = 0.0

    for i, D in enumerate(D_range):
        phases = phases_init.copy()
        injector = StochasticInjector(D, seed=seed + i)
        for _ in range(n_steps):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
            if D > 0:
                phases = injector.inject(phases, engine._dt)
        R, _ = compute_order_parameter(phases)
        if i == 0:
            R_det = R
        if best_R < R:
            best_R = R
            best_D = float(D)

    return NoiseProfile(D=best_D, R_achieved=best_R, R_deterministic=R_det)
