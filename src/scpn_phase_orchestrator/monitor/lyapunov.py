# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov stability monitor

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["LyapunovGuard", "LyapunovState", "lyapunov_spectrum"]


@dataclass
class LyapunovState:
    V: float
    dV_dt: float
    in_basin: bool
    max_phase_diff: float


class LyapunovGuard:
    """Lyapunov stability monitor for Kuramoto networks.

    V(θ) = -(K/2N) Σ_{i,j} A_ij cos(θ_i - θ_j)

    dV/dt ≤ 0 for gradient flow (Kuramoto is gradient on V).
    Basin of attraction: max|θ_i - θ_j| < π/2 for connected pairs.

    van Hemmen & Wreszinski 1993, J. Stat. Phys. 72:145-166.
    """

    def __init__(self, basin_threshold: float = np.pi / 2):
        self._basin_threshold = basin_threshold
        self._prev_V: float | None = None

    def evaluate(self, phases: NDArray, knm: NDArray) -> LyapunovState:
        """Compute Lyapunov function, its time derivative, and basin check."""
        n = len(phases)
        if n == 0:
            return LyapunovState(V=0.0, dV_dt=0.0, in_basin=True, max_phase_diff=0.0)

        diff = phases[:, np.newaxis] - phases[np.newaxis, :]
        cos_diff = np.cos(diff)

        # V(θ) = -(1/2N) Σ K_ij cos(θ_i - θ_j)
        V = -0.5 * float(np.sum(knm * cos_diff)) / n

        # Numerical dV/dt from consecutive calls
        dV_dt = 0.0
        if self._prev_V is not None:
            dV_dt = V - self._prev_V
        self._prev_V = V

        # Basin check: max phase difference between connected pairs
        connected = knm > 0
        if np.any(connected):
            abs_diff = np.abs(diff)
            # Wrap to [-π, π]
            abs_diff = np.minimum(abs_diff, 2 * np.pi - abs_diff)
            max_diff = float(np.max(abs_diff[connected]))
        else:
            max_diff = 0.0

        in_basin = max_diff < self._basin_threshold

        return LyapunovState(
            V=V,
            dV_dt=dV_dt,
            in_basin=in_basin,
            max_phase_diff=max_diff,
        )

    def reset(self) -> None:
        self._prev_V = None


def _kuramoto_jacobian(
    phases: NDArray, omegas: NDArray, knm: NDArray, alpha: NDArray
) -> NDArray:
    """Jacobian of the Kuramoto RHS: J_ij = K_ij cos(θ_j - θ_i - α_ij).

    Diagonal: J_ii = -Σ_{j≠i} K_ij cos(θ_j - θ_i - α_ij).
    """
    diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
    J = knm * np.cos(diff)
    np.fill_diagonal(J, 0.0)
    np.fill_diagonal(J, -J.sum(axis=1))
    return J


def lyapunov_spectrum(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float = 0.01,
    n_steps: int = 1000,
    qr_interval: int = 10,
    zeta: float = 0.0,
    psi: float = 0.0,
) -> NDArray:
    """Full Lyapunov spectrum (all N exponents) via QR decomposition.

    Evolves N perturbation vectors alongside the Kuramoto ODE. Every
    qr_interval steps, QR-reorthogonalizes and accumulates growth rates
    from the diagonal of R.

    Benettin et al. 1980, Meccanica 15:9-20.
    Shimada & Nagashima 1979, Prog. Theor. Phys. 61:1605-1616.

    Args:
        phases_init: (N,) initial phases
        omegas: (N,) natural frequencies
        knm: (N, N) coupling matrix
        alpha: (N, N) phase lag matrix
        dt: integration timestep
        n_steps: total integration steps
        qr_interval: steps between QR reorthogonalizations
        zeta: driver strength
        psi: target phase

    Returns:
        (N,) array of Lyapunov exponents, sorted descending.
    """
    n = len(phases_init)
    phases = phases_init.copy()
    Q = np.eye(n, dtype=np.float64)
    exponents = np.zeros(n, dtype=np.float64)
    n_qr = 0
    total_time = 0.0

    for step in range(n_steps):
        # Kuramoto Euler step
        diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        driving = zeta * np.sin(psi - phases) if zeta != 0 else 0.0
        dtheta = omegas + coupling + driving
        phases = phases + dt * dtheta
        total_time += dt

        # Evolve perturbation vectors: dQ/dt = J @ Q
        J = _kuramoto_jacobian(phases, omegas, knm, alpha)
        Q = Q + dt * (J @ Q)

        # QR reorthogonalization
        if (step + 1) % qr_interval == 0:
            Q, R = np.linalg.qr(Q)
            diag = np.abs(np.diag(R))
            diag = np.maximum(diag, 1e-300)
            exponents += np.log(diag)
            n_qr += 1

    if n_qr > 0:
        exponents /= total_time

    return np.sort(exponents)[::-1]
