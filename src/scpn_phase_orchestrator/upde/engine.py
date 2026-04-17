# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE integration engine

from __future__ import annotations

import threading

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

    # Dormand-Prince (1980) Butcher tableau — 7-stage, FSAL property
    # Coefficients from Table 5.2 of Hairer, Norsett & Wanner, vol. I
    _DP_A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
    )
    # B4: 4th-order weights for error estimation (embedded pair)
    _DP_B4 = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    # B5: 5th-order weights — the accepted solution
    _DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    # C: stage time fractions c_i, row sums of A
    _DP_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

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
        if _HAS_RUST:  # pragma: no cover
            from spo_kernel import PyUPDEStepper

            self._rust = PyUPDEStepper(n_oscillators, dt, method, atol=atol, rtol=rtol)

        self._phase_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._sin_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._scratch_dtheta = np.empty(n_oscillators, dtype=np.float64)

        if method == "rk45":
            self._ks = [np.empty(n_oscillators, dtype=np.float64) for _ in range(7)]
            self._err_buf = np.empty(n_oscillators, dtype=np.float64)

        # Serialise concurrent step()/run() callers on this instance — the
        # pre-allocated scratch arrays above are not safe to share. Reentrant
        # so that run() → step() within the same thread does not deadlock.
        self._lock = threading.RLock()

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
        if not np.all(np.isfinite(phases)):
            raise ValueError("phases contain NaN/Inf")
        if not np.all(np.isfinite(omegas)):
            raise ValueError("omegas contain NaN/Inf")
        if not np.all(np.isfinite(knm)):
            raise ValueError("knm contains NaN/Inf")
        if not np.all(np.isfinite(alpha)):
            raise ValueError("alpha contains NaN/Inf")
        with self._lock:
            if self._rust is not None:  # pragma: no cover
                return np.asarray(
                    self._rust.step(
                        np.ascontiguousarray(phases.ravel()),
                        np.ascontiguousarray(omegas.ravel()),
                        np.ascontiguousarray(knm.ravel()),
                        float(zeta),
                        float(psi),
                        np.ascontiguousarray(alpha.ravel()),
                    )
                )
            if self._method == "euler":
                return self._euler_step(phases, omegas, knm, zeta, psi, alpha)
            if self._method == "rk45":
                return self._rk45_step(phases, omegas, knm, zeta, psi, alpha)
            return self._rk4_step(phases, omegas, knm, zeta, psi, alpha)

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
        """Run n_steps, return final phases. Uses Rust batch API when available."""
        with self._lock:
            if self._rust is not None:  # pragma: no cover
                return np.asarray(
                    self._rust.run(
                        np.ascontiguousarray(phases.ravel()),
                        np.ascontiguousarray(omegas.ravel()),
                        np.ascontiguousarray(knm.ravel()),
                        float(zeta),
                        float(psi),
                        np.ascontiguousarray(alpha.ravel()),
                        n_steps,
                    )
                )
            p = phases.copy()
            for _ in range(n_steps):
                p = self.step(p, omegas, knm, zeta, psi, alpha)
            return p

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
        # Sakaguchi-Kuramoto coupling: K_ij sin(θ_j - θ_i - α_ij)
        # α_ij is the Sakaguchi phase-lag (Sakaguchi & Kuramoto 1986)
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        self._phase_diff -= alpha
        np.sin(self._phase_diff, out=self._sin_diff)

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
        # Mod 2π keeps phases on S¹ (circle topology)
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
        # Classic RK4: 4th-order Runge-Kutta (1/6, 1/3, 1/3, 1/6 weights)
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
            # Evaluate all 7 Dormand-Prince stages
            ks[0][:] = self._derivative(phases, omegas, knm, zeta, psi, alpha)
            for i in range(1, 7):
                # y_stage = y_n + h Σ a_ij k_j (Butcher row i)
                stage = phases + dt * np.dot(A[i, :i], np.array(ks[:i]))
                ks[i][:] = self._derivative(stage, omegas, knm, zeta, psi, alpha)

            ks_arr = np.array(ks)
            # 5th-order solution (accepted) and 4th-order (for error)
            y5 = phases + dt * np.dot(self._DP_B5, ks_arr)
            y4 = phases + dt * np.dot(self._DP_B4, ks_arr)

            # Local error estimate: |y5 - y4| / (atol + rtol*max(|y|,|y5|))
            # Hairer & Wanner mixed tolerance scaling
            np.subtract(y5, y4, out=self._err_buf)
            np.abs(self._err_buf, out=self._err_buf)
            scale = self._atol + self._rtol * np.maximum(np.abs(phases), np.abs(y5))
            err_norm = float(np.max(self._err_buf / scale))

            if err_norm <= 1.0:
                # PI step-size control: h_new = 0.9 * h * err^(-1/p)
                # p=5 for acceptance → exponent -0.2
                factor = min(5.0, 0.9 * err_norm ** (-0.2)) if err_norm > 0.0 else 5.0
                self._last_dt = min(dt * factor, self._dt * 10.0)
                result: NDArray = y5 % TWO_PI
                return result

            # Reject — shrink with safety factor, exponent -1/4 (order p+1)
            factor = max(0.2, 0.9 * err_norm ** (-0.25))  # pragma: no cover
            dt = dt * factor  # pragma: no cover

        # Exhausted retries, accept current result
        self._last_dt = dt  # pragma: no cover
        result_fallback: NDArray = y5 % TWO_PI  # pragma: no cover
        return result_fallback  # pragma: no cover
