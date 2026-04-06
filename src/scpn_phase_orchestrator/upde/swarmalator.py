# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator dynamics

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scpn_phase_orchestrator._compat import TWO_PI

try:
    from spo_kernel import (
        swarmalator_run_rust as _rust_run,
        PySwarmalatorStepper as _SwarmalatorStepper,
    )
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["SwarmalatorEngine"]

class SwarmalatorEngine:
    def __init__(self, n_agents: int, dim: int = 2, dt: float = 0.01):
        self._n = n_agents
        self._dim = dim
        self._dt = dt
        if _HAS_RUST:
            self._stepper = _SwarmalatorStepper(n_agents, dim, dt)
        else:
            self._stepper = None

    def step(self, pos: NDArray, phases: NDArray, omegas: NDArray, a: float = 1.0, b: float = 1.0, j: float = 1.0, k: float = 1.0) -> tuple[NDArray, NDArray]:
        if _HAS_RUST:
            p_pos = np.ascontiguousarray(pos.ravel(), dtype=np.float64)
            p_phases = np.ascontiguousarray(phases, dtype=np.float64)
            p_omegas = np.ascontiguousarray(omegas, dtype=np.float64)
            new_pos, new_phases = self._stepper.step(p_pos, p_phases, p_omegas, a, b, j, k)
            return np.asarray(new_pos).reshape(self._n, self._dim), np.asarray(new_phases)
        
        # Python fallback (O(N^2))
        n, dim, dt = self._n, self._dim, self._dt
        new_pos = pos.copy()
        new_phases = phases.copy()
        for i in range(n):
            diff = pos - pos[i]
            dist = np.sqrt(np.sum(diff**2, axis=1) + 1e-6)
            cos_diff = np.cos(phases - phases[i])
            sin_diff = np.sin(phases - phases[i])
            
            attract = (a + j * cos_diff) / dist
            repulse = b / (dist**3 + 1e-6)
            
            vel = np.sum(diff * (attract - repulse)[:, np.newaxis], axis=0) / n
            new_pos[i] += dt * vel
            
            dth = omegas[i] + k * np.mean(sin_diff / dist)
            new_phases[i] = (phases[i] + dt * dth) % TWO_PI
            
        return new_pos, new_phases

    def run(self, pos: NDArray, phases: NDArray, omegas: NDArray, a: float = 1.0, b: float = 1.0, j: float = 1.0, k: float = 1.0, n_steps: int = 100) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        curr_pos, curr_phases = pos.copy(), phases.copy()
        pos_traj = np.empty((n_steps, self._n, self._dim))
        phase_traj = np.empty((n_steps, self._n))
        for i in range(n_steps):
            curr_pos, curr_phases = self.step(curr_pos, curr_phases, omegas, a, b, j, k)
            pos_traj[i] = curr_pos
            phase_traj[i] = curr_phases
        return curr_pos, curr_phases, pos_traj, phase_traj

    def order_parameter(self, phases: NDArray) -> float:
        return float(np.abs(np.mean(np.exp(1j * phases))))
