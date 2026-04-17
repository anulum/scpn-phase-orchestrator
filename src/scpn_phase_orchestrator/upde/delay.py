# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

try:
    from spo_kernel import (
        delayed_kuramoto_run_rust as _rust_delayed_run,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["DelayBuffer", "DelayedEngine"]


class DelayBuffer:
    """Circular buffer storing phase history for delayed coupling.

    Stores last `max_delay_steps` snapshots. Retrieves phases from
    `delay_steps` steps ago.
    """

    def __init__(self, n_oscillators: int, max_delay_steps: int):
        if n_oscillators < 1:
            raise ValueError(f"n_oscillators must be >= 1, got {n_oscillators}")
        if max_delay_steps < 1:
            raise ValueError(f"max_delay_steps must be >= 1, got {max_delay_steps}")
        self._n = n_oscillators
        self._max = max_delay_steps
        self._buffer: deque[NDArray] = deque(maxlen=max_delay_steps)

    def push(self, phases: NDArray) -> None:
        """Append a phase snapshot to the buffer."""
        self._buffer.append(phases.copy())

    def get_delayed(self, delay_steps: int) -> NDArray | None:
        """Return phases from `delay_steps` ago, or None if not enough history."""
        if delay_steps < 1 or delay_steps > len(self._buffer):
            return None
        return self._buffer[-delay_steps]

    @property
    def length(self) -> int:
        """Number of snapshots currently stored."""
        return len(self._buffer)

    def clear(self) -> None:
        """Discard all stored phase snapshots."""
        self._buffer.clear()


class DelayedEngine:
    """Kuramoto with time-delayed coupling.

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t-τ) - θ_i(t) - α_ij)
    """

    def __init__(self, n_oscillators: int, dt: float, delay_steps: int = 1):
        if n_oscillators < 1:
            raise ValueError(f"n_oscillators must be >= 1, got {n_oscillators}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if delay_steps < 1:
            raise ValueError(f"delay_steps must be >= 1, got {delay_steps}")
        self._n = n_oscillators
        self._dt = dt
        self._delay_steps = delay_steps
        self._buffer: deque[NDArray] = deque(maxlen=delay_steps + 1)

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: NDArray | None = None,
        step_idx: int = 0,
    ) -> NDArray:
        self._buffer.append(phases.copy())
        delayed = self._buffer[0] if len(self._buffer) > self._delay_steps else phases
        if alpha is None:
            alpha = np.zeros((self._n, self._n))
        diff = delayed[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        dtheta = omegas + coupling
        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - phases)
        step_out: NDArray = (phases + self._dt * dtheta) % TWO_PI
        return step_out

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: NDArray | None = None,
        n_steps: int = 100,
    ) -> NDArray:
        if _HAS_RUST:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
            a = np.ascontiguousarray(
                alpha.ravel() if alpha is not None else np.zeros(self._n * self._n),
                dtype=np.float64,
            )
            result: NDArray = np.asarray(
                _rust_delayed_run(
                    p,
                    o,
                    k,
                    a,
                    self._n,
                    zeta,
                    psi,
                    self._dt,
                    self._delay_steps,
                    n_steps,
                )
            )
            return result
        p = phases.copy()
        for i in range(n_steps):
            p = self.step(p, omegas, knm, zeta, psi, alpha, step_idx=i)
        return p
