# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stochastic noise injection and optimal D*

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.special import i0, i1  # type: ignore[import-untyped]

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

if TYPE_CHECKING:
    from scpn_phase_orchestrator.upde.engine import UPDEEngine

__all__ = ["StochasticInjector", "NoiseProfile", "find_optimal_noise"]


@dataclass
class NoiseProfile:
    D: float
    R_achieved: float
    R_deterministic: float


class StochasticInjector:
    """Add calibrated noise to phase dynamics.

    Euler-Maruyama: θ_i(t+dt) = θ_i(t) + f(θ)*dt + √(2D*dt) * ξ_i
    where ξ_i ~ N(0,1) i.i.d.

    Tselios et al. 2025 — stochastic resonance in Kuramoto networks.
    """

    def __init__(self, D: float, seed: int | None = None):
        if D < 0:
            raise ValueError(f"noise strength D must be non-negative, got {D}")
        self._D = D
        self._rng = np.random.default_rng(seed)

    @property
    def D(self) -> float:
        return self._D

    @D.setter
    def D(self, value: float) -> None:
        if value < 0:
            raise ValueError(f"D must be non-negative, got {value}")
        self._D = value

    def inject(self, phases: NDArray, dt: float) -> NDArray:
        """Add Wiener noise to phases: θ += √(2D*dt) * N(0,1)."""
        if self._D == 0.0:
            return phases
        noise = self._rng.standard_normal(len(phases))
        result: NDArray = (phases + np.sqrt(2.0 * self._D * dt) * noise) % TWO_PI
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
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    D_range: NDArray | None = None,
    n_steps: int = 500,
    seed: int = 42,
) -> NoiseProfile:
    """Sweep noise levels, return D that maximizes R.

    Uses the engine to simulate n_steps at each D value.
    """
    if D_range is None:
        K_mean = float(np.mean(knm[knm > 0])) if np.any(knm > 0) else 1.0
        D_range = np.linspace(0.0, K_mean, 11)

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
