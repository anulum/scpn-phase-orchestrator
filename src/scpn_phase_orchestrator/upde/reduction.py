# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen mean-field reduction

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from spo_kernel import (
        fit_lorentzian_rust as _rust_fit_lorentzian,
    )
    from spo_kernel import (
        oa_run_rust as _rust_oa_run,
    )
    from spo_kernel import (
        steady_state_r_oa_rust as _rust_steady_state_r,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["OttAntonsenReduction", "OAState"]


@dataclass
class OAState:
    z: complex
    R: float
    psi: float
    K_c: float


class OttAntonsenReduction:
    """Ott-Antonsen mean-field reduction for globally-coupled Kuramoto.

    dz/dt = -(Δ + iω₀)z + (K/2)(z - |z|²z)

    Exact for Lorentzian g(ω) with half-width Δ, center ω₀.
    Critical coupling K_c = 2Δ.
    Steady-state: R_ss = √(1 - 2Δ/K) for K > K_c.

    Ott & Antonsen 2008, Chaos 18(3):037113.
    """

    def __init__(self, omega_0: float, delta: float, K: float, dt: float = 0.01):
        if delta < 0:
            raise ValueError(f"delta (half-width) must be non-negative, got {delta}")
        self._omega_0 = omega_0
        self._delta = delta
        self._K = K
        self._dt = dt

    @property
    def K_c(self) -> float:
        """Critical coupling: K_c = 2Δ."""
        return 2.0 * self._delta

    def steady_state_R(self) -> float:
        """Analytical steady-state order parameter R_ss = √(1 - 2Δ/K)."""
        if _HAS_RUST:
            return float(_rust_steady_state_r(self._delta, self._K))
        if self.K_c >= self._K:
            return 0.0
        return float(np.sqrt(1.0 - 2.0 * self._delta / self._K))

    def step(self, z: complex) -> complex:
        """RK4 integration of dz/dt = -(Δ+iω₀)z + (K/2)(z-|z|²z)."""
        dt = self._dt

        def f(zz: complex) -> complex:
            # OA manifold ODE: -(Δ+iω₀)z damps/rotates the mean field;
            # (K/2)(z - |z|²z) is the cubic self-coupling from the
            # Lorentzian g(ω) closure (Ott & Antonsen 2008, Eq. 9)
            return -(self._delta + 1j * self._omega_0) * zz + (self._K / 2.0) * (
                zz - abs(zz) ** 2 * zz
            )

        k1 = f(z)
        k2 = f(z + 0.5 * dt * k1)
        k3 = f(z + 0.5 * dt * k2)
        k4 = f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, z0: complex, n_steps: int) -> OAState:
        """Integrate n_steps, return final state."""
        if _HAS_RUST:
            z_re, z_im, R, psi = _rust_oa_run(
                z0.real,
                z0.imag,
                self._omega_0,
                self._delta,
                self._K,
                self._dt,
                n_steps,
            )
            return OAState(z=complex(z_re, z_im), R=R, psi=psi, K_c=self.K_c)
        z = z0
        for _ in range(n_steps):
            z = self.step(z)
        R = abs(z)
        psi = float(np.angle(z))
        return OAState(z=z, R=R, psi=psi, K_c=self.K_c)

    def predict_from_oscillators(self, omegas: NDArray, K: float) -> OAState:
        """Fit Lorentzian to omegas, run OA reduction as fast diagnostic.

        Uses median as ω₀ and IQR-based Δ estimate.
        """
        if _HAS_RUST:
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            omega_0, delta = _rust_fit_lorentzian(o)
        else:
            # Fit Lorentzian: median → centre ω₀, IQR/2 → half-width Δ
            # (IQR of a Lorentzian equals 2Δ, so IQR/2 ≈ Δ)
            omega_0 = float(np.median(omegas))
            q75, q25 = np.percentile(omegas, [75, 25])
            delta = (q75 - q25) / 2.0 if q75 > q25 else 0.01

        reducer = OttAntonsenReduction(omega_0, delta, K, dt=self._dt)
        # Small seed breaks symmetry; integrate ~10 time units to steady state
        z0 = complex(0.01, 0.0)
        return reducer.run(z0, n_steps=int(10.0 / self._dt))
