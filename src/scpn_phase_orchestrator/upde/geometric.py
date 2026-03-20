# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus-preserving geometric integrator

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["TorusEngine"]


class TorusEngine:
    """Geometric integrator on the N-torus T^N = (S¹)^N.

    Works in the Lie algebra of SO(2)^N: represents each phase as a unit
    complex number z_i = exp(iθ_i), computes the derivative in the tangent
    space, and maps back via the exponential map.

    This avoids the mod 2π discontinuity that causes subtle errors in
    standard integrators when phases cross 0/2π.

    Symplectic Euler on T^N:
      z_i(t+dt) = z_i(t) · exp(i · ω_eff_i · dt)
    where ω_eff_i is the full Kuramoto RHS.
    """

    def __init__(self, n_oscillators: int, dt: float):
        self._n = n_oscillators
        self._dt = dt

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """One step on the torus. Returns phases in [0, 2π)."""
        # Lift to complex unit circle
        z = np.exp(1j * phases)

        # Compute effective angular velocity (Kuramoto RHS)
        omega_eff = self._derivative(phases, omegas, knm, zeta, psi, alpha)

        # Exponential map: rotate each oscillator
        z_new = z * np.exp(1j * omega_eff * self._dt)

        # Project back to phases (angle extraction is exact, no mod needed)
        result: NDArray = np.angle(z_new) % (2 * np.pi)
        return result

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
        p = phases.copy()
        for _ in range(n_steps):
            p = self.step(p, omegas, knm, zeta, psi, alpha)
        return p

    def _derivative(
        self,
        theta: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        result = omegas + coupling
        if zeta != 0.0:
            result = result + zeta * np.sin(psi - theta)
        return result
