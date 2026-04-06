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

try:
    from spo_kernel import (
        PyDelayedStepper as _DelayedStepper,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["DelayedEngine"]


class DelayedEngine:
    def __init__(self, n_oscillators: int, dt: float, delay_steps: int = 1):
        self._n = n_oscillators
        self._dt = dt
        self._delay_steps = delay_steps
        if _HAS_RUST:
            self._stepper = _DelayedStepper(n_oscillators, delay_steps, dt)
        else:
            self._stepper = None
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
        if _HAS_RUST:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            k = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
            a = np.ascontiguousarray(
                alpha.ravel() if alpha is not None else np.zeros(self._n * self._n),
                dtype=np.float64,
            )
            result: NDArray = np.asarray(
                self._stepper.step(p, o, k, a, zeta, psi, step_idx)
            )
            return result

        self._buffer.append(phases.copy())
        delayed = self._buffer[0] if len(self._buffer) > self._delay_steps else phases
        if alpha is None:
            alpha = np.zeros((self._n, self._n))
        diff = delayed[np.newaxis, :] - phases[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        dtheta = omegas + coupling
        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - phases)
        step_result: NDArray = (phases + self._dt * dtheta) % TWO_PI
        return step_result

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
            result_run: NDArray = np.asarray(
                self._stepper.run(p, o, k, a, zeta, psi, n_steps)
            )
            return result_run
        p = phases.copy()
        for i in range(n_steps):
            p = self.step(p, omegas, knm, zeta, psi, alpha, step_idx=i)
        return p
