# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["UPDEEngine"]


class UPDEEngine:
    """Kuramoto UPDE integrator with pre-allocated scratch arrays.

    dtheta_i/dt = omega_i
                  + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
                  + zeta sin(Psi - theta_i)
    """

    # Dormand-Prince RK45 Butcher tableau coefficients
    _DP_A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
        ]
    )
    _DP_B4 = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100]
    )
    _DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    _DP_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        self._n = n_oscillators
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
            from spo_kernel import PyUPDEStepper

            self._rust = PyUPDEStepper(n_oscillators, dt, method, atol=atol, rtol=rtol)

        self._phase_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._sin_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._scratch_dtheta = np.empty(n_oscillators, dtype=np.float64)

        if method == "rk45":
            self._ks = [np.empty(n_oscillators, dtype=np.float64) for _ in range(6)]
            self._err_buf = np.empty(n_oscillators, dtype=np.float64)

    @property
    def last_dt(self) -> float:
        """Actual dt used on the last accepted step (relevant for rk45)."""
        return self._last_dt

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """Advance phases by one timestep, return new phases in [0, 2*pi)."""
        if not (np.isfinite(zeta) and np.isfinite(psi)):
            raise ValueError("zeta and psi must be finite")
        n = self._n
        if phases.shape != (n,):
            raise ValueError(f"phases.shape={phases.shape}, expected ({n},)")
        if omegas.shape != (n,):
            raise ValueError(f"omegas.shape={omegas.shape}, expected ({n},)")
        if knm.shape != (n, n):
            raise ValueError(f"knm.shape={knm.shape}, expected ({n}, {n})")
        if alpha.shape != (n, n):
            raise ValueError(f"alpha.shape={alpha.shape}, expected ({n}, {n})")
        if self._rust is not None:
            result = self._rust.step(
                phases.ravel().tolist(),
                omegas.ravel().tolist(),
                knm.ravel().tolist(),
                float(zeta),
                float(psi),
                alpha.ravel().tolist(),
            )
            return np.asarray(result, dtype=np.float64)
        if self._method == "euler":
            return self._euler_step(phases, omegas, knm, zeta, psi, alpha)
        if self._method == "rk45":
            return self._rk45_step(phases, omegas, knm, zeta, psi, alpha)
        return self._rk4_step(phases, omegas, knm, zeta, psi, alpha)

    def compute_order_parameter(self, phases: NDArray) -> tuple[float, float]:
        """Kuramoto order parameter: R = |<exp(i*theta)>|, psi = arg(...)."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        return compute_order_parameter(phases)

    def _derivative(
        self,
        theta: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        # theta_j - theta_i - alpha_ij via outer subtraction
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        self._phase_diff -= alpha
        np.sin(self._phase_diff, out=self._sin_diff)

        # sum_j K_ij sin(theta_j - theta_i - alpha_ij)
        np.sum(knm * self._sin_diff, axis=1, out=self._scratch_dtheta)
        self._scratch_dtheta += omegas

        # external drive: zeta * sin(Psi - theta_i)
        if zeta != 0.0:
            self._scratch_dtheta += zeta * np.sin(psi - theta)

        return self._scratch_dtheta

    def _euler_step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        dtheta = self._derivative(phases, omegas, knm, zeta, psi, alpha)
        result: NDArray = (phases + self._dt * dtheta) % TWO_PI
        return result

    def _rk4_step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        dt = self._dt
        k1 = self._derivative(phases, omegas, knm, zeta, psi, alpha).copy()
        k2 = self._derivative(
            phases + 0.5 * dt * k1, omegas, knm, zeta, psi, alpha
        ).copy()
        k3 = self._derivative(
            phases + 0.5 * dt * k2, omegas, knm, zeta, psi, alpha
        ).copy()
        k4 = self._derivative(phases + dt * k3, omegas, knm, zeta, psi, alpha)
        weighted = k1 + 2.0 * k2 + 2.0 * k3 + k4
        result: NDArray = (phases + (dt / 6.0) * weighted) % TWO_PI
        return result

    def _rk45_step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """Dormand-Prince RK45 with embedded error estimation and adaptive dt."""
        dt = self._last_dt
        A = self._DP_A
        ks = self._ks
        max_reject = 3

        for _ in range(max_reject + 1):
            ks[0][:] = self._derivative(phases, omegas, knm, zeta, psi, alpha)
            for i in range(1, 6):
                stage = phases + dt * np.dot(A[i, :i], np.array(ks[:i]))
                ks[i][:] = self._derivative(stage, omegas, knm, zeta, psi, alpha)

            ks_arr = np.array(ks)
            y5 = phases + dt * np.dot(self._DP_B5, ks_arr)
            y4 = phases + dt * np.dot(self._DP_B4, ks_arr)

            np.subtract(y5, y4, out=self._err_buf)
            np.abs(self._err_buf, out=self._err_buf)
            scale = self._atol + self._rtol * np.maximum(np.abs(phases), np.abs(y5))
            err_norm = float(np.max(self._err_buf / scale))

            if err_norm <= 1.0:
                factor = min(5.0, 0.9 * err_norm ** (-0.2)) if err_norm > 0.0 else 5.0
                self._last_dt = min(dt * factor, self._dt * 10.0)
                result: NDArray = y5 % TWO_PI
                return result

            # Reject — shrink dt and retry
            factor = max(0.2, 0.9 * err_norm ** (-0.25))
            dt = dt * factor

        # Exhausted retries, accept current result
        self._last_dt = dt
        result_fallback: NDArray = y5 % TWO_PI
        return result_fallback
