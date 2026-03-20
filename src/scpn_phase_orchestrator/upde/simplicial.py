# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Simplicial (higher-order) Kuramoto coupling

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["SimplicialEngine"]


class SimplicialEngine:
    """Kuramoto with pairwise + 3-body simplicial coupling.

    dθ_i/dt = ω_i
              + (σ₁/N) Σ_j A_ij sin(θ_j - θ_i)
              + (σ₂/N²) Σ_{j,k} B_ijk sin(θ_j + θ_k - 2θ_i)

    σ₁: pairwise coupling strength (standard Kuramoto)
    σ₂: 3-body coupling strength (simplicial)

    The 3-body term induces explosive (first-order) transitions and
    shrinks basins of attraction despite improving stability.

    Gambuzza et al. 2023, Nature Physics; Tang et al. 2025.
    """

    def __init__(self, n_oscillators: int, dt: float, sigma2: float = 0.0):
        self._n = n_oscillators
        self._dt = dt
        self._sigma2 = sigma2

    @property
    def sigma2(self) -> float:
        return self._sigma2

    @sigma2.setter
    def sigma2(self, value: float) -> None:
        self._sigma2 = value

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """Euler step with pairwise + 3-body coupling."""
        dtheta = self._derivative(phases, omegas, knm, zeta, psi, alpha)
        result: NDArray = (phases + self._dt * dtheta) % TWO_PI
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
        n = self._n
        # Pairwise: standard Kuramoto
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
        pairwise = np.sum(knm * np.sin(diff), axis=1)

        result = omegas + pairwise

        # 3-body simplicial term: σ₂/N² Σ_{j,k} sin(θ_j + θ_k - 2θ_i)
        if self._sigma2 != 0.0 and n >= 3:
            three_body = np.zeros(n, dtype=np.float64)
            # Vectorized: for each i, sum over all j,k pairs
            # θ_j + θ_k - 2θ_i = (θ_j - θ_i) + (θ_k - θ_i)
            diff_ji = theta[np.newaxis, :] - theta[:, np.newaxis]  # (n, n)
            for i in range(n):
                # sum_{j,k} sin(diff_ji[i,j] + diff_ji[i,k])
                # = sum_j sin(d_j) * sum_k cos(d_k) + sum_j cos(d_j) * sum_k sin(d_k)
                # where d_j = θ_j - θ_i
                d = diff_ji[i, :]
                S = np.sum(np.sin(d))
                C = np.sum(np.cos(d))
                three_body[i] = S * C + C * S  # = 2SC
            three_body *= self._sigma2 / (n * n)
            result = result + three_body

        if zeta != 0.0:
            result = result + zeta * np.sin(psi - theta)

        out: NDArray = result
        return out
