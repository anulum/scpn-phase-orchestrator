# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strang operator splitting integrator

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

try:
    from spo_kernel import (
        splitting_run_rust as _rust_splitting_run,
        PySplittingStepper as _SplittingStepper,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["SplittingEngine"]


class SplittingEngine:
    """Strang splitting: exact rotation for ω, RK4 for coupling.

    Splits dθ/dt = ω + K·sin(coupling) into:
      A: dθ/dt = ω          (exact: θ += ω·dt)
      B: dθ/dt = coupling    (RK4 on nonlinear part)

    Strang scheme: A(dt/2) → B(dt) → A(dt/2), second-order symmetric.

    Advantage over monolithic RK45: the linear part (rotation) is solved
    exactly, so phases stay on the circle without accumulating truncation
    error from integrating ω numerically.

    Hairer, Lubich & Wanner 2006, Geometric Numerical Integration, §II.5.
    """

    def __init__(self, n_oscillators: int, dt: float):
        self._n = n_oscillators
        self._dt = dt
        if _HAS_RUST:
            self._stepper = _SplittingStepper(n_oscillators, dt)
        else:
            self._stepper = None
        self._phase_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._sin_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._scratch = np.empty(n_oscillators, dtype=np.float64)

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """One Strang-split step: A(dt/2) -> B(dt) -> A(dt/2)."""
        dt = self._dt
        # A(dt/2): exact rotation
        p = (phases + 0.5 * dt * omegas) % TWO_PI
        # B(dt): RK4 on coupling-only derivative
        p = self._rk4_coupling(p, knm, zeta, psi, alpha, dt)
        # A(dt/2): exact rotation
        return (p + 0.5 * dt * omegas) % TWO_PI

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
        n_steps: int,
    ) -> NDArray:
        if _HAS_RUST:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
            a = np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
            return self._stepper.run(p, o, k, a, zeta, psi, n_steps)
        
        p = phases.copy()
        for _ in range(n_steps):
            # Slow Python fallback
            p = (p + 0.5 * self._dt * omegas) % TWO_PI
            d = self._derivative(p, knm, alpha, zeta, psi)
            # RK4 on coupling
            k1 = d
            k2 = self._derivative((p + 0.5 * self._dt * k1) % TWO_PI, knm, alpha, zeta, psi)
            k3 = self._derivative((p + 0.5 * self._dt * k2) % TWO_PI, knm, alpha, zeta, psi)
            k4 = self._derivative((p + self._dt * k3) % TWO_PI, knm, alpha, zeta, psi)
            p = (p + (self._dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % TWO_PI
            p = (p + 0.5 * self._dt * omegas) % TWO_PI
        return p

    def _coupling_deriv(
        self, theta: NDArray, knm: NDArray, zeta: float, psi: float, alpha: NDArray
    ) -> NDArray:
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        self._phase_diff -= alpha
        np.sin(self._phase_diff, out=self._sin_diff)
        np.sum(knm * self._sin_diff, axis=1, out=self._scratch)
        if zeta != 0.0:
            self._scratch += zeta * np.sin(psi - theta)
        return self._scratch.copy()

    def _rk4_coupling(
        self,
        phases: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
        dt: float,
    ) -> NDArray:
        k1 = self._coupling_deriv(phases, knm, zeta, psi, alpha)
        k2 = self._coupling_deriv(
            (phases + 0.5 * dt * k1) % TWO_PI, knm, zeta, psi, alpha
        )
        k3 = self._coupling_deriv(
            (phases + 0.5 * dt * k2) % TWO_PI, knm, zeta, psi, alpha
        )
        k4 = self._coupling_deriv((phases + dt * k3) % TWO_PI, knm, zeta, psi, alpha)
        result: NDArray = (phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % TWO_PI
        return result
