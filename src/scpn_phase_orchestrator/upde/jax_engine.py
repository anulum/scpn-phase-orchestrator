# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — JAX-accelerated UPDE engine

"""GPU-accelerated Kuramoto solver via JAX JIT compilation.

Raises ImportError if JAX is not installed. Check HAS_JAX before use.
Usage:
    from scpn_phase_orchestrator.upde.jax_engine import HAS_JAX
    if HAS_JAX:
        from scpn_phase_orchestrator.upde.jax_engine import JaxUPDEEngine
        engine = JaxUPDEEngine(n, dt=0.01)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

__all__ = ["JaxUPDEEngine", "HAS_JAX"]

TWO_PI = 2.0 * np.pi

try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

if TYPE_CHECKING:
    import jax.numpy as jnp


def _build_jax_step():  # type: ignore[no-untyped-def]
    """Build JIT-compiled Kuramoto step function."""

    @jit
    def _kuramoto_step(phases, omegas, knm, zeta, psi, alpha, dt):  # type: ignore[no-untyped-def]
        diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
        coupling = jnp.sum(knm * jnp.sin(diff - alpha), axis=1)
        dphi = omegas + coupling
        dphi = dphi + zeta * jnp.sin(psi - phases)
        new_phases = phases + dt * dphi
        return new_phases % (2.0 * jnp.pi)

    @jit
    def _kuramoto_rk4(phases, omegas, knm, zeta, psi, alpha, dt):  # type: ignore[no-untyped-def]
        def deriv(p):  # type: ignore[no-untyped-def]
            diff = p[jnp.newaxis, :] - p[:, jnp.newaxis]
            coupling = jnp.sum(knm * jnp.sin(diff - alpha), axis=1)
            return omegas + coupling + zeta * jnp.sin(psi - p)

        k1 = deriv(phases)
        k2 = deriv(phases + 0.5 * dt * k1)
        k3 = deriv(phases + 0.5 * dt * k2)
        k4 = deriv(phases + dt * k3)
        new_phases = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return new_phases % (2.0 * jnp.pi)

    return _kuramoto_step, _kuramoto_rk4


def _build_jax_sl_step():  # type: ignore[no-untyped-def]
    """Build JIT-compiled Stuart-Landau step function."""

    @jit
    def _sl_rk4(state, omegas, mu, knm, knm_r, zeta, psi, alpha, epsilon, dt):  # type: ignore[no-untyped-def]
        n = omegas.shape[0]

        def deriv(s):  # type: ignore[no-untyped-def]
            th, am = s[:n], s[n:]
            diff = th[jnp.newaxis, :] - th[:, jnp.newaxis]
            phase_coupling = jnp.sum(knm * jnp.sin(diff - alpha), axis=1)
            amp_coupling = jnp.sum(
                knm_r * jnp.maximum(am, 0.0)[jnp.newaxis, :] * jnp.cos(diff - alpha),
                axis=1,
            )
            dtheta = omegas + phase_coupling + zeta * jnp.sin(psi - th)
            dr = (mu - am * am) * am + epsilon * amp_coupling
            return jnp.concatenate([dtheta, dr])

        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        new_theta = new_state[:n] % (2.0 * jnp.pi)
        new_r = jnp.maximum(new_state[n:], 0.0)
        return jnp.concatenate([new_theta, new_r])

    return _sl_rk4


class JaxUPDEEngine:
    """JAX-accelerated Kuramoto/UPDE integrator.

    GPU-compiled via jax.jit. First call triggers XLA compilation
    (~1-3s), subsequent calls run at native speed.
    """

    def __init__(self, n: int, dt: float = 0.01, method: str = "rk4") -> None:
        if not HAS_JAX:
            msg = "JAX not installed. Install with: pip install jax jaxlib"
            raise ImportError(msg)
        self._n = n
        self._dt = dt
        self._method = method
        euler_fn, rk4_fn = _build_jax_step()
        self._euler = euler_fn
        self._rk4 = rk4_fn

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        jp = jnp.asarray(phases)
        jo = jnp.asarray(omegas)
        jk = jnp.asarray(knm)
        ja = jnp.asarray(alpha)

        if self._method == "rk4":
            result = self._rk4(jp, jo, jk, zeta, psi, ja, self._dt)
        else:
            result = self._euler(jp, jo, jk, zeta, psi, ja, self._dt)

        return np.asarray(result)


class JaxStuartLandauEngine:
    """JAX-accelerated Stuart-Landau integrator (RK4 only)."""

    def __init__(self, n: int, dt: float = 0.01) -> None:
        if not HAS_JAX:
            msg = "JAX not installed. Install with: pip install jax jaxlib"
            raise ImportError(msg)
        self._n = n
        self._dt = dt
        self._sl_rk4 = _build_jax_sl_step()

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
        js = jnp.asarray(state)
        result = self._sl_rk4(
            js,
            jnp.asarray(omegas),
            jnp.asarray(mu),
            jnp.asarray(knm),
            jnp.asarray(knm_r),
            zeta,
            psi,
            jnp.asarray(alpha),
            epsilon,
            self._dt,
        )
        return np.asarray(result)
