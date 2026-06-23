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
from numbers import Real
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
    """Noise-sweep result linking diffusion strength to observed order."""

    D: float
    R_achieved: float
    R_deterministic: float


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


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_noise_range(value: FloatArray | None) -> FloatArray | None:
    """Return the validated ``(min, max)`` noise-level range, else raise."""
    if value is None:
        return None
    d_range = np.asarray(value, dtype=np.float64)
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
        self._D = _validate_finite_non_negative(D, name="D")
        self._rng = np.random.default_rng(seed)

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
            Oscillator phases in radians, shape ``(N,)``.
        dt : float
            Integration step size.

        Returns
        -------
        FloatArray
            The phases with added Wiener noise.
        """
        dt = _validate_finite_positive(dt, name="dt")
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
        Diffusion coefficients to sweep, or ``None`` for the default range.
    n_steps : int
        Number of integration steps to run.
    seed : int
        Seed for the deterministic RNG.

    Returns
    -------
    NoiseProfile
        The noise profile whose diffusion ``D`` maximises ``R``.
    """
    n_steps = _validate_positive_int(n_steps, name="n_steps")
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
