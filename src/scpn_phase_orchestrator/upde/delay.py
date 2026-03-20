# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Time-delayed Kuramoto coupling

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["DelayBuffer", "DelayedEngine"]


class DelayBuffer:
    """Circular buffer storing phase history for delayed coupling.

    Stores last `max_delay_steps` snapshots. Retrieves phases from
    `delay_steps` steps ago.
    """

    def __init__(self, n_oscillators: int, max_delay_steps: int):
        if max_delay_steps < 1:
            raise ValueError(f"max_delay_steps must be >= 1, got {max_delay_steps}")
        self._n = n_oscillators
        self._max = max_delay_steps
        self._buffer: deque[NDArray] = deque(maxlen=max_delay_steps)

    def push(self, phases: NDArray) -> None:
        self._buffer.append(phases.copy())

    def get_delayed(self, delay_steps: int) -> NDArray | None:
        """Return phases from `delay_steps` ago, or None if not enough history."""
        if delay_steps < 1 or delay_steps > len(self._buffer):
            return None
        return self._buffer[-delay_steps]

    @property
    def length(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


class DelayedEngine:
    """Kuramoto with time-delayed coupling.

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t-τ) - θ_i(t) - α_ij)

    Time delay generates effective higher-order interactions for free
    (Ciszak et al. 2025). Uses circular buffer for delayed phase lookup.
    Falls back to instantaneous coupling when delay buffer is not full.
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        delay_steps: int = 1,
    ):
        self._n = n_oscillators
        self._dt = dt
        self._delay_steps = delay_steps
        self._buffer = DelayBuffer(n_oscillators, max_delay_steps=delay_steps + 1)

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        delayed = self._buffer.get_delayed(self._delay_steps)
        self._buffer.push(phases)
        if delayed is None:
            # Not enough history yet — use instantaneous coupling
            coupling_phases = phases
        else:
            coupling_phases = delayed

        # dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t-τ) - θ_i(t) - α_ij)
        diff = coupling_phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        dtheta = omegas + coupling

        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - phases)

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
