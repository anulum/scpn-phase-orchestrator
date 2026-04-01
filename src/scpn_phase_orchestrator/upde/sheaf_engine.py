# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996-2026 Miroslav Šotek. All rights reserved.
# © Code 2020-2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Cellular Sheaf UPDE Engine

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["SheafUPDEEngine"]


class SheafUPDEEngine:
    """Cellular Sheaf UPDE integrator for multi-dimensional phase vectors.

    Phase per oscillator is a vector of dimension D.
    Restriction maps (coupling blocks) B_ij are D x D matrices mapping 
    the phase space of oscillator j into the space of oscillator i.

    Mathematics:
    d(theta_{i,d})/dt = omega_{i,d} 
                        + sum_j sum_k B_ij^{dk} sin(theta_{j,k} - theta_{i,d})
                        + zeta * sin(Psi_d - theta_{i,d})

    This enables complex cross-frequency coupling and opinion dynamics
    over multidimensional belief spaces.
    """

    def __init__(
        self,
        n_oscillators: int,
        d_dimensions: int,
        dt: float,
        method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        self._n = n_oscillators
        self._d = d_dimensions
        self._dt = dt
        if method not in ("euler", "rk4", "rk45"):
            msg = f"Unknown method {method!r}, expected 'euler', 'rk4', or 'rk45'"
            raise ValueError(msg)
        self._method = method
        self._atol = atol
        self._rtol = rtol
        self._last_dt = dt

        self._rust = None
        if _HAS_RUST:
            try:
                from spo_kernel import PySheafUPDEStepper
                self._rust = PySheafUPDEStepper(n_oscillators, d_dimensions, dt, method, atol=atol, rtol=rtol)
            except ImportError:
                pass

    @property
    def last_dt(self) -> float:
        return self._last_dt

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        restriction_maps: NDArray,
        zeta: float,
        psi: NDArray,
    ) -> NDArray:
        """Advance phases by one timestep.

        Args:
            phases: Current phase matrix [theta_i,d], shape (N, D).
            omegas: Natural frequency matrix [omega_i,d], shape (N, D).
            restriction_maps: Block matrix coupling [B_ij^{dk}], shape (N, N, D, D).
            zeta: External forcing strength (global scalar).
            psi: Reference phase target vector, shape (D,).

        Returns:
            New phase matrix, shape (N, D).
        """
        if self._rust is not None:
            res = self._rust.step(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                np.ascontiguousarray(restriction_maps.ravel(), dtype=np.float64),
                float(zeta),
                np.ascontiguousarray(psi.ravel(), dtype=np.float64),
            )
            return np.asarray(res).reshape((self._n, self._d))
        
        if self._method == "euler":
            return self._euler_step(phases, omegas, restriction_maps, zeta, psi)
        
        raise NotImplementedError(f"Method {self._method} sheaf fallback not implemented in Python")

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        restriction_maps: NDArray,
        zeta: float,
        psi: NDArray,
        n_steps: int,
    ) -> NDArray:
        """Run multiple steps in a batch, return final phases."""
        if self._rust is not None:
            res = self._rust.run(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                np.ascontiguousarray(restriction_maps.ravel(), dtype=np.float64),
                float(zeta),
                np.ascontiguousarray(psi.ravel(), dtype=np.float64),
                n_steps,
            )
            return np.asarray(res).reshape((self._n, self._d))
            
        p = phases.copy()
        for _ in range(n_steps):
            p = self.step(p, omegas, restriction_maps, zeta, psi)
        return p

    def _derivative(
        self,
        theta: NDArray,
        omegas: NDArray,
        restriction_maps: NDArray,
        zeta: float,
        psi: NDArray,
    ) -> NDArray:
        n, d = self._n, self._d
        dtheta = omegas.copy()
        for i in range(n):
            for dim in range(d):
                coupling_sum = 0.0
                for j in range(n):
                    for k in range(d):
                        b_val = restriction_maps[i, j, dim, k]
                        if b_val != 0.0:
                            coupling_sum += b_val * np.sin(theta[j, k] - theta[i, dim])
                dtheta[i, dim] += coupling_sum
                if zeta != 0.0:
                    dtheta[i, dim] += zeta * np.sin(psi[dim] - theta[i, dim])
        return dtheta

    def _euler_step(
        self,
        phases: NDArray,
        omegas: NDArray,
        restriction_maps: NDArray,
        zeta: float,
        psi: NDArray,
    ) -> NDArray:
        dtheta = self._derivative(phases, omegas, restriction_maps, zeta, psi)
        result: NDArray = (phases + self._dt * dtheta) % TWO_PI
        return result
