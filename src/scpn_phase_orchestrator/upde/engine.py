# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["UPDEEngine"]


class UPDEEngine:
    """Kuramoto UPDE integrator with pre-allocated scratch arrays.

    dtheta_i/dt = omega_i
                  + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
                  + zeta sin(Psi - theta_i)
    """

    def __init__(self, n_oscillators: int, dt: float, method: str = "euler"):
        self._n = n_oscillators
        self._dt = dt
        if method not in ("euler", "rk4"):
            raise ValueError(f"Unknown method {method!r}, expected 'euler' or 'rk4'")
        self._method = method

        self._rust = None
        if _HAS_RUST:
            from spo_kernel import PyUPDEStepper

            self._rust = PyUPDEStepper(n_oscillators, dt, method)

        self._phase_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._sin_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._scratch_dtheta = np.empty(n_oscillators, dtype=np.float64)

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """Advance phases by one timestep, return new phases in [0, 2*pi)."""
        if not (np.isfinite(zeta) and np.isfinite(psi)):
            raise ValueError("zeta and psi must be finite")
        n = self._n
        if phases.shape != (n,):
            raise ValueError(f"phases.shape={phases.shape}, expected ({n},)")
        if omegas.shape != (n,):
            raise ValueError(f"omegas.shape={omegas.shape}, expected ({n},)")
        if knm.shape != (n, n):
            raise ValueError(f"knm.shape={knm.shape}, expected ({n}, {n})")
        if alpha.shape != (n, n):
            raise ValueError(f"alpha.shape={alpha.shape}, expected ({n}, {n})")
        if self._rust is not None:
            result = self._rust.step(
                phases.ravel().tolist(),
                omegas.ravel().tolist(),
                knm.ravel().tolist(),
                float(zeta),
                float(psi),
                alpha.ravel().tolist(),
            )
            return np.asarray(result, dtype=np.float64)
        if self._method == "euler":
            return self._euler_step(phases, omegas, knm, zeta, psi, alpha)
        return self._rk4_step(phases, omegas, knm, zeta, psi, alpha)

    def compute_order_parameter(self, phases: NDArray) -> tuple[float, float]:
        """Kuramoto order parameter: R = |<exp(i*theta)>|, psi = arg(...)."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        return compute_order_parameter(phases)

    def _derivative(
        self,
        theta: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        # theta_j - theta_i - alpha_ij via outer subtraction
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        self._phase_diff -= alpha
        np.sin(self._phase_diff, out=self._sin_diff)

        # sum_j K_ij sin(theta_j - theta_i - alpha_ij)
        np.sum(knm * self._sin_diff, axis=1, out=self._scratch_dtheta)
        self._scratch_dtheta += omegas

        # external drive: zeta * sin(Psi - theta_i)
        if zeta != 0.0:
            self._scratch_dtheta += zeta * np.sin(psi - theta)

        return self._scratch_dtheta

    def _euler_step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        dtheta = self._derivative(phases, omegas, knm, zeta, psi, alpha)
        result: NDArray = (phases + self._dt * dtheta) % TWO_PI
        return result

    def _rk4_step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        dt = self._dt
        k1 = self._derivative(phases, omegas, knm, zeta, psi, alpha).copy()
        k2 = self._derivative(
            phases + 0.5 * dt * k1, omegas, knm, zeta, psi, alpha
        ).copy()
        k3 = self._derivative(
            phases + 0.5 * dt * k2, omegas, knm, zeta, psi, alpha
        ).copy()
        k4 = self._derivative(phases + dt * k3, omegas, knm, zeta, psi, alpha)
        weighted = k1 + 2.0 * k2 + 2.0 * k3 + k4
        result: NDArray = (phases + (dt / 6.0) * weighted) % TWO_PI
        return result
