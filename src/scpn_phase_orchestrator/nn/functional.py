# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable Kuramoto functional API

"""Pure JAX functions for differentiable Kuramoto dynamics.

All functions are JIT-compilable, vmap-compatible, and differentiable
via JAX autodiff. No NumPy conversions — inputs and outputs stay as
JAX arrays for gradient flow.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi


def kuramoto_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single Euler step of the Kuramoto model.

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        dt: integration timestep

    Returns:
        (N,) updated phases, wrapped to [0, 2pi)
    """
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]
    coupling = jnp.sum(K * jnp.sin(diff), axis=1)
    return (phases + dt * (omegas + coupling)) % TWO_PI


def kuramoto_rk4_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single RK4 step of the Kuramoto model.

    Args:
        phases: (N,) oscillator phases in [0, 2pi)
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        dt: integration timestep

    Returns:
        (N,) updated phases, wrapped to [0, 2pi)
    """
    def deriv(p: jax.Array) -> jax.Array:
        diff = p[jnp.newaxis, :] - p[:, jnp.newaxis]
        return omegas + jnp.sum(K * jnp.sin(diff), axis=1)

    k1 = deriv(phases)
    k2 = deriv(phases + 0.5 * dt * k1)
    k3 = deriv(phases + 0.5 * dt * k2)
    k4 = deriv(phases + dt * k3)
    new = phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return new % TWO_PI


def kuramoto_forward(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    dt: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N Kuramoto steps, returning final state and trajectory.

    Uses jax.lax.scan for efficient compilation and autodiff.

    Args:
        phases: (N,) initial oscillator phases
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        dt: integration timestep
        n_steps: number of integration steps
        method: "rk4" or "euler"

    Returns:
        Tuple of:
            final: (N,) phases after n_steps
            trajectory: (n_steps, N) full phase trajectory
    """
    step_fn = kuramoto_rk4_step if method == "rk4" else kuramoto_step

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, omegas, K, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


def order_parameter(phases: jax.Array) -> jax.Array:
    """Kuramoto order parameter R = |<exp(i*phi)>|.

    Differentiable scalar measuring global synchronization.
    R=1 means perfect sync, R~0 means incoherent.

    Args:
        phases: (N,) or (T, N) oscillator phases

    Returns:
        Scalar R value (or (T,) if trajectory input)
    """
    z = jnp.exp(1j * phases)
    return jnp.abs(jnp.mean(z, axis=-1))


def plv(trajectory: jax.Array) -> jax.Array:
    """Phase-Locking Value matrix from a phase trajectory.

    PLV_ij = |<exp(i*(phi_i(t) - phi_j(t)))>_t|

    Args:
        trajectory: (T, N) phase trajectory

    Returns:
        (N, N) PLV matrix, values in [0, 1]
    """
    # (T, N, 1) - (T, 1, N) -> (T, N, N) phase differences
    diff = trajectory[:, :, jnp.newaxis] - trajectory[:, jnp.newaxis, :]
    return jnp.abs(jnp.mean(jnp.exp(1j * diff), axis=0))
