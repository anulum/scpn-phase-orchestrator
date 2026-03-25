# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator dynamics

"""Swarmalator model: coupled spatial + phase dynamics.

Agents are simultaneously self-propelled particles AND phase oscillators.
Phase modulates spatial attraction; proximity modulates phase coupling.

Position dynamics:
    ẋ_i = (1/N) Σ_j [ (x_j - x_i)/|x_j - x_i| · (A + J cos(θ_j - θ_i))
                        - (x_j - x_i)/|x_j - x_i|³ · B ]

Phase dynamics:
    θ̇_i = ω_i + (K/N) Σ_j sin(θ_j - θ_i) / |x_j - x_i|

Parameters:
    A: spatial attraction strength
    B: spatial repulsion strength
    J: phase-dependent spatial modulation (-1 to 1)
    K: phase coupling strength

J > 0: phase-similar agents attract → static sync
J < 0: phase-similar agents repel → static async
J = 0: standard swarm + standard Kuramoto (decoupled)
K > 0: nearby agents synchronize → phase waves
K < 0: nearby agents desynchronize → static phase wave

O'Keeffe, Hong, Strogatz, Nature Communications 2017.
Experimental validation: Nature Communications Dec 2025 (colloidal system).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

TWO_PI = 2.0 * np.pi


class SwarmalatorEngine:
    """Swarmalator dynamics in D-dimensional space.

    State: positions (N, D) + phases (N,).
    """

    def __init__(
        self,
        n: int,
        dim: int = 2,
        dt: float = 0.01,
        A: float = 1.0,
        B: float = 1.0,
        J: float = 0.5,
        K: float = 1.0,
    ) -> None:
        self._n = n
        self._dim = dim
        self._dt = dt
        self.A = A
        self.B = B
        self.J = J
        self.K = K

    def step(
        self,
        positions: NDArray,
        phases: NDArray,
        omegas: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Single Euler step of swarmalator dynamics.

        Args:
            positions: (N, D) spatial positions
            phases: (N,) oscillator phases
            omegas: (N,) natural frequencies

        Returns:
            Tuple of (new_positions, new_phases)
        """
        n = self._n
        dt = self._dt
        eps = 1e-6

        # Pairwise displacement: delta[i,j] = x_j - x_i, shape (N, N, D)
        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        # Pairwise distance: (N, N)
        dist = np.sqrt(np.sum(delta**2, axis=2) + eps)
        # Unit direction: (N, N, D)
        direction = delta / dist[:, :, np.newaxis]

        # Phase difference: cos(θ_j - θ_i), shape (N, N)
        phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
        cos_diff = np.cos(phase_diff)
        sin_diff = np.sin(phase_diff)

        # Position dynamics
        attract = direction * (self.A + self.J * cos_diff[:, :, np.newaxis])
        repulse = delta / (dist**3 + eps)[:, :, np.newaxis] * self.B
        dx = np.sum(attract - repulse, axis=1) / n

        # Phase dynamics: coupling weighted by 1/distance
        inv_dist = 1.0 / dist
        np.fill_diagonal(inv_dist, 0.0)
        dtheta = omegas + self.K / n * np.sum(sin_diff * inv_dist, axis=1)

        new_pos = positions + dt * dx
        new_phases = (phases + dt * dtheta) % TWO_PI
        return new_pos, new_phases

    def run(
        self,
        positions: NDArray,
        phases: NDArray,
        omegas: NDArray,
        n_steps: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        """Run n_steps, returning final state and trajectories.

        Returns:
            (final_pos, final_phases, pos_traj, phase_traj)
            pos_traj: (n_steps, N, D), phase_traj: (n_steps, N)
        """
        n, dim = self._n, self._dim
        pos_traj = np.empty((n_steps, n, dim))
        phase_traj = np.empty((n_steps, n))

        pos, ph = positions.copy(), phases.copy()
        for i in range(n_steps):
            pos, ph = self.step(pos, ph, omegas)
            pos_traj[i] = pos
            phase_traj[i] = ph

        return pos, ph, pos_traj, phase_traj

    def spatial_coherence(self, positions: NDArray) -> float:
        """Mean pairwise distance (spatial compactness measure)."""
        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dists = np.sqrt(np.sum(delta**2, axis=2))
        n = positions.shape[0]
        mean_dist: float = float(np.sum(dists) / (n * (n - 1)))
        return mean_dist

    def phase_coherence(self, phases: NDArray) -> float:
        """Kuramoto order parameter R."""
        z = np.exp(1j * phases)
        return float(np.abs(np.mean(z)))

    def phase_spatial_correlation(self, positions: NDArray, phases: NDArray) -> float:
        """Correlation between spatial distance and phase difference.

        Positive = nearby agents have similar phases (phase wave).
        Negative = nearby agents have different phases (anti-phase wave).
        Near zero = no spatial-phase relationship (decoupled).
        """
        delta = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        dist = np.sqrt(np.sum(delta**2, axis=2))
        phase_diff = np.abs(phases[np.newaxis, :] - phases[:, np.newaxis])
        phase_diff = np.minimum(phase_diff, TWO_PI - phase_diff)

        idx = np.triu_indices(self._n, k=1)
        d_flat = dist[idx]
        p_flat = phase_diff[idx]

        if len(d_flat) < 2:
            return 0.0

        d_c = d_flat - np.mean(d_flat)
        p_c = p_flat - np.mean(p_flat)
        num = np.sum(d_c * p_c)
        denom = np.sqrt(np.sum(d_c**2) * np.sum(p_c**2) + 1e-10)
        return float(num / denom)
