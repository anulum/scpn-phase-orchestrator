# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Second-order inertial Kuramoto

"""Second-order Kuramoto model with inertia for power grid dynamics.

m_i θ̈_i + d_i θ̇_i = P_i + Σ_j K_ij sin(θ_j - θ_i)

where m_i is inertia (rotating mass), d_i is damping, P_i is power
injection (positive = generator, negative = load), and K_ij is the
transmission line susceptance.

The swing equation is the standard model for power system transient
stability. Desynchronization → cascading blackout (Iberian Peninsula
blackout, April 2025: 31 GW disconnected).

Filatrella et al. 2008; Dörfler & Bullo 2014; PRX Energy 2024.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        inertial_run_rust as _rust_run,
    )
    from spo_kernel import (
        inertial_step_rust as _rust_step,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

TWO_PI = 2.0 * np.pi


class InertialKuramotoEngine:
    """Second-order Kuramoto with inertia and damping.

    State: (theta, omega_dot) where omega_dot = dtheta/dt.
    """

    def __init__(self, n: int, dt: float = 0.01) -> None:
        self._n = n
        self._dt = dt

    def step(
        self,
        theta: NDArray,
        omega_dot: NDArray,
        power: NDArray,
        knm: NDArray,
        inertia: NDArray,
        damping: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Single RK4 step of the swing equation.

        Args:
            theta: (N,) rotor angles
            omega_dot: (N,) angular velocities (deviation from nominal)
            power: (N,) power injection (P_i, positive = generation)
            knm: (N, N) coupling (transmission susceptance)
            inertia: (N,) inertia constants (m_i)
            damping: (N,) damping coefficients (d_i)

        Returns:
            Tuple of (new_theta, new_omega_dot)
        """
        if _HAS_RUST:
            n = self._n
            th = np.ascontiguousarray(theta, dtype=np.float64)
            od = np.ascontiguousarray(omega_dot, dtype=np.float64)
            pw = np.ascontiguousarray(power, dtype=np.float64)
            km = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
            in_ = np.ascontiguousarray(inertia, dtype=np.float64)
            dm = np.ascontiguousarray(damping, dtype=np.float64)
            new_th, new_od = _rust_step(
                th,
                od,
                pw,
                km,
                in_,
                dm,
                n,
                self._dt,
            )
            return np.asarray(new_th), np.asarray(new_od)

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

        new_theta = theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
        new_omega = omega_dot + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)

        new_theta = new_theta % TWO_PI
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
        """Run n_steps, returning final state and trajectories.

        Returns:
            (final_theta, final_omega, theta_traj, omega_traj)
            where trajectories are (n_steps, N)
        """
        n = self._n

        if _HAS_RUST:
            th = np.ascontiguousarray(theta, dtype=np.float64)
            od = np.ascontiguousarray(omega_dot, dtype=np.float64)
            pw = np.ascontiguousarray(power, dtype=np.float64)
            km = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
            in_ = np.ascontiguousarray(inertia, dtype=np.float64)
            dm = np.ascontiguousarray(damping, dtype=np.float64)
            f_th, f_od, t_th, t_od = _rust_run(
                th,
                od,
                pw,
                km,
                in_,
                dm,
                n,
                self._dt,
                n_steps,
            )
            return (
                np.asarray(f_th),
                np.asarray(f_od),
                np.asarray(t_th).reshape(n_steps, n),
                np.asarray(t_od).reshape(n_steps, n),
            )

        theta_traj = np.empty((n_steps, n))
        omega_traj = np.empty((n_steps, n))

        th, od = theta.copy(), omega_dot.copy()
        for i in range(n_steps):
            th, od = self.step(th, od, power, knm, inertia, damping)
            theta_traj[i] = th
            omega_traj[i] = od

        return th, od, theta_traj, omega_traj

    def frequency_deviation(self, omega_dot: NDArray) -> float:
        """Max absolute frequency deviation from nominal (Hz).

        Power grids operate at 50/60 Hz. Deviation > 0.5 Hz triggers
        load shedding; > 2 Hz triggers emergency disconnection.
        """
        return float(np.max(np.abs(omega_dot)) / TWO_PI)

    def coherence(self, theta: NDArray) -> float:
        """Phase coherence R (order parameter) of rotor angles."""
        z = np.exp(1j * theta)
        return float(np.abs(np.mean(z)))
