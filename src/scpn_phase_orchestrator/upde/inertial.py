# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Second-order inertial Kuramoto

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        PyInertialStepper as _InertialStepper,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

TWO_PI = 2.0 * np.pi


class InertialKuramotoEngine:
    def __init__(self, n: int, dt: float = 0.01) -> None:
        self._n = n
        self._dt = dt
        if _HAS_RUST:
            self._stepper = _InertialStepper(n, dt)
        else:
            self._stepper = None

    def step(
        self,
        theta: NDArray,
        omega_dot: NDArray,
        power: NDArray,
        knm: NDArray,
        inertia: NDArray,
        damping: NDArray,
    ) -> tuple[NDArray, NDArray]:
        if _HAS_RUST:
            th = np.ascontiguousarray(theta, dtype=np.float64)
            od = np.ascontiguousarray(omega_dot, dtype=np.float64)
            pw = np.ascontiguousarray(power, dtype=np.float64)
            km = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
            in_ = np.ascontiguousarray(inertia, dtype=np.float64)
            dm = np.ascontiguousarray(damping, dtype=np.float64)
            res_th, res_od = self._stepper.step(th, od, pw, km, in_, dm)
            return np.asarray(res_th), np.asarray(res_od)

        dt = self._dt

        def deriv(th: NDArray, od: NDArray) -> tuple[NDArray, NDArray]:
            diff = th[np.newaxis, :] - th[:, np.newaxis]
            coupling = np.sum(knm * np.sin(diff), axis=1)
            accel = (power + coupling - damping * od) / inertia
            return od, accel

        k1t, k1o = deriv(theta, omega_dot)
        k2t, k2o = deriv(theta + 0.5 * dt * k1t, omega_dot + 0.5 * dt * k1o)
        k3t, k3o = deriv(theta + 0.5 * dt * k2t, omega_dot + 0.5 * dt * k2o)
        k4t, k4o = deriv(theta + dt * k3t, omega_dot + dt * k3o)
        new_theta = (theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)) % TWO_PI
        new_omega = omega_dot + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        return new_theta, new_omega

    def run(
        self,
        theta: NDArray,
        omega_dot: NDArray,
        power: NDArray,
        knm: NDArray,
        inertia: NDArray,
        damping: NDArray,
        n_steps: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        if _HAS_RUST:
            # Fallback to loop for now as PyInertialStepper doesnt have run() yet
            # but we can add it later if needed.
            th, od = theta.copy(), omega_dot.copy()
            theta_traj = np.empty((n_steps, self._n))
            omega_traj = np.empty((n_steps, self._n))
            for i in range(n_steps):
                th, od = self.step(th, od, power, knm, inertia, damping)
                theta_traj[i] = th
                omega_traj[i] = od
            return th, od, theta_traj, omega_traj

        theta_traj = np.empty((n_steps, self._n))
        omega_traj = np.empty((n_steps, self._n))
        th, od = theta.copy(), omega_dot.copy()
        for i in range(n_steps):
            th, od = self.step(th, od, power, knm, inertia, damping)
            theta_traj[i] = th
            omega_traj[i] = od
        return th, od, theta_traj, omega_traj

    def frequency_deviation(self, omega_dot: NDArray) -> float:
        return float(np.max(np.abs(omega_dot)) / TWO_PI)

    def coherence(self, theta: NDArray) -> float:
        return float(np.abs(np.mean(np.exp(1j * theta))))
