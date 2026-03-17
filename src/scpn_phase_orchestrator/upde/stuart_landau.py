# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau oscillator model

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["StuartLandauEngine"]

# Type alias for the ODE parameter bundle passed to _derivative
_Params = tuple[NDArray, NDArray, NDArray, NDArray, float, float, NDArray, float]


class StuartLandauEngine:
    """Coupled Stuart-Landau (phase-amplitude) integrator.

    State vector layout: state[:n] = phases θ, state[n:] = amplitudes r.

    Phase ODE (Acebrón et al. 2005, Rev. Mod. Phys. 77(1)):
        dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j - θ_i - α_ij) + ζ sin(Ψ - θ_i)

    Amplitude ODE:
        dr_i/dt = (μ_i - r_i²)·r_i + ε Σ_j K^r_ij · r_j · cos(θ_j - θ_i)
    """

    # Dormand-Prince RK45 Butcher tableau
    _DP_A = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [
                9017 / 3168,
                -355 / 33,
                46732 / 5247,
                49 / 176,
                -5103 / 18656,
                0,
            ],
        ]
    )
    _DP_B4 = np.array(
        [
            5179 / 57600,
            0,
            7571 / 16695,
            393 / 640,
            -92097 / 339200,
            187 / 2100,
        ]
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
        if method not in ("euler", "rk4", "rk45"):
            msg = f"Unknown method {method!r}, expected 'euler', 'rk4', or 'rk45'"
            raise ValueError(msg)
        self._n = n_oscillators
        self._dt = dt
        self._method = method
        self._atol = atol
        self._rtol = rtol
        self._last_dt = dt

        self._use_rust = False
        try:  # pragma: no cover
            import spo_kernel  # noqa: PLC0415

            self._rust = spo_kernel.PyStuartLandauStepper(
                n_oscillators, dt=dt, method=method, n_substeps=1, atol=atol, rtol=rtol
            )
            self._use_rust = True
        except ImportError:  # pragma: no cover — Rust FFI optional
            pass

        n = n_oscillators
        self._phase_diff = np.empty((n, n), dtype=np.float64)
        self._sin_diff = np.empty((n, n), dtype=np.float64)
        self._cos_diff = np.empty((n, n), dtype=np.float64)
        self._scratch_dtheta = np.empty(n, dtype=np.float64)
        self._scratch_dr = np.empty(n, dtype=np.float64)
        self._scratch_deriv = np.empty(2 * n, dtype=np.float64)

        if method == "rk45":
            self._ks = [np.empty(2 * n, dtype=np.float64) for _ in range(6)]
            self._err_buf = np.empty(2 * n, dtype=np.float64)

    @property
    def last_dt(self) -> float:
        return self._last_dt

    def step(
        self,
        state: NDArray,
        omegas: NDArray,
        mu: NDArray,
        knm: NDArray,
        knm_r: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
        epsilon: float = 1.0,
    ) -> NDArray:
        """Advance (θ, r) by one timestep. Returns new state (2N,)."""
        self._validate(state, omegas, mu, knm, knm_r, zeta, psi, alpha)
        if self._use_rust:  # pragma: no cover
            result = np.asarray(
                self._rust.step(
                    np.ascontiguousarray(state),
                    np.ascontiguousarray(omegas),
                    np.ascontiguousarray(mu),
                    np.ascontiguousarray(knm.ravel()),
                    np.ascontiguousarray(knm_r.ravel()),
                    zeta,
                    psi,
                    np.ascontiguousarray(alpha.ravel()),
                    epsilon,
                )
            )
            self._last_dt = self._rust.last_dt
            return result
        p: _Params = (omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon)
        if self._method == "euler":
            return self._euler_step(state, p)
        if self._method == "rk45":
            return self._rk45_step(state, p)
        return self._rk4_step(state, p)

    def compute_order_parameter(self, state: NDArray) -> tuple[float, float]:
        """Amplitude-weighted Kuramoto: Z = mean(r_i · exp(i·θ_i))."""
        n = self._n
        z = np.mean(state[n:] * np.exp(1j * state[:n]))
        return float(np.abs(z)), float(np.angle(z) % TWO_PI)

    def compute_mean_amplitude(self, state: NDArray) -> float:
        return float(np.mean(state[self._n :]))

    def _validate(
        self,
        state: NDArray,
        omegas: NDArray,
        mu: NDArray,
        knm: NDArray,
        knm_r: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> None:
        n = self._n
        if state.shape != (2 * n,):
            raise ValueError(f"state.shape={state.shape}, expected ({2 * n},)")
        if omegas.shape != (n,):
            raise ValueError(f"omegas.shape={omegas.shape}, expected ({n},)")
        if mu.shape != (n,):
            raise ValueError(f"mu.shape={mu.shape}, expected ({n},)")
        if knm.shape != (n, n):
            raise ValueError(f"knm.shape={knm.shape}, expected ({n}, {n})")
        if knm_r.shape != (n, n):
            raise ValueError(f"knm_r.shape={knm_r.shape}, expected ({n}, {n})")
        if alpha.shape != (n, n):
            raise ValueError(f"alpha.shape={alpha.shape}, expected ({n}, {n})")
        if not (np.isfinite(zeta) and np.isfinite(psi)):
            raise ValueError("zeta and psi must be finite")
        if not np.all(np.isfinite(state)):
            raise ValueError("state contains NaN or Inf")

    def _derivative(self, state: NDArray, p: _Params) -> NDArray:
        omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon = p
        n = self._n
        theta = state[:n]
        r = state[n:]

        np.subtract(
            theta[np.newaxis, :],
            theta[:, np.newaxis],
            out=self._phase_diff,
        )

        np.sin(self._phase_diff - alpha, out=self._sin_diff)
        np.sum(knm * self._sin_diff, axis=1, out=self._scratch_dtheta)
        self._scratch_dtheta += omegas
        if zeta != 0.0:
            self._scratch_dtheta += zeta * np.sin(psi - theta)

        # Clamp r >= 0 for coupling: intermediate RK stages can produce
        # negative amplitudes that flip the coupling sign (P1-1 audit fix).
        r_clamped = np.maximum(r, 0.0)

        np.cos(self._phase_diff, out=self._cos_diff)
        np.sum(
            knm_r * self._cos_diff * r_clamped[np.newaxis, :],
            axis=1,
            out=self._scratch_dr,
        )
        self._scratch_dr *= epsilon
        self._scratch_dr += (mu - r * r) * r

        self._scratch_deriv[:n] = self._scratch_dtheta
        self._scratch_deriv[n:] = self._scratch_dr
        return self._scratch_deriv

    def _post_step(self, state: NDArray) -> NDArray:
        n = self._n
        state[:n] %= TWO_PI
        np.maximum(state[n:], 0.0, out=state[n:])
        return state

    def _euler_step(self, state: NDArray, p: _Params) -> NDArray:
        deriv = self._derivative(state, p)
        return self._post_step(state + self._dt * deriv)

    def _rk4_step(self, state: NDArray, p: _Params) -> NDArray:
        dt = self._dt
        k1 = self._derivative(state, p).copy()
        k2 = self._derivative(state + 0.5 * dt * k1, p).copy()
        k3 = self._derivative(state + 0.5 * dt * k2, p).copy()
        k4 = self._derivative(state + dt * k3, p)
        weighted = k1 + 2.0 * k2 + 2.0 * k3 + k4
        return self._post_step(state + (dt / 6.0) * weighted)

    def _rk45_step(self, state: NDArray, p: _Params) -> NDArray:
        dt = self._last_dt
        A = self._DP_A
        ks = self._ks
        max_reject = 3

        for _ in range(max_reject + 1):
            ks[0][:] = self._derivative(state, p)
            for i in range(1, 6):
                stage = state + dt * np.dot(A[i, :i], np.array(ks[:i]))
                ks[i][:] = self._derivative(stage, p)

            ks_arr = np.array(ks)
            y5 = state + dt * np.dot(self._DP_B5, ks_arr)
            y4 = state + dt * np.dot(self._DP_B4, ks_arr)

            np.subtract(y5, y4, out=self._err_buf)
            np.abs(self._err_buf, out=self._err_buf)
            scale = self._atol + self._rtol * np.maximum(np.abs(state), np.abs(y5))
            err_norm = float(np.max(self._err_buf / scale))

            if err_norm <= 1.0:
                factor = min(5.0, 0.9 * err_norm ** (-0.2)) if err_norm > 0.0 else 5.0
                self._last_dt = min(dt * factor, self._dt * 10.0)
                return self._post_step(y5.copy())

            factor = max(0.2, 0.9 * err_norm ** (-0.25))
            dt = dt * factor

        self._last_dt = dt
        return self._post_step(y5.copy())
