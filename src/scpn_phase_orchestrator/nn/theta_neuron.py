# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Theta neuron model (Type I excitability)

"""Theta neuron (Ermentrout-Kopell canonical model) for coupled excitable systems.

dθ_i/dt = (1 - cos(θ_i)) + (1 + cos(θ_i)) · (η_i + I_syn_i)

where I_syn_i = Σ_j K_ij · (1 - cos(θ_j)) is synaptic input.

The theta neuron is the canonical model for Type I neuronal excitability
(Ermentrout & Kopell 1986). Unlike Kuramoto oscillators which are always
oscillating, theta neurons can be excitable (η < 0) — they fire only
when driven by sufficient synaptic input.

Requires: jax>=0.4
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

TWO_PI = 2.0 * jnp.pi


def _theta_deriv(
    phases: jax.Array,
    eta: jax.Array,
    K: jax.Array,
) -> jax.Array:
    """Derivative for theta neuron network."""
    # Synaptic input: I_syn_i = Σ_j K_ij (1 - cos(θ_j))
    pulse = 1.0 - jnp.cos(phases)  # (N,)
    I_syn = K @ pulse  # (N,)
    return (1.0 - jnp.cos(phases)) + (1.0 + jnp.cos(phases)) * (eta + I_syn)


def theta_neuron_step(
    phases: jax.Array,
    eta: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single Euler step of the theta neuron model.

    Args:
        phases: (N,) neuron phases in [0, 2pi)
        eta: (N,) excitability parameters (η>0: oscillatory, η<0: excitable)
        K: (N, N) synaptic coupling matrix
        dt: integration timestep

    Returns:
        (N,) updated phases
    """
    dphi = _theta_deriv(phases, eta, K)
    return (phases + dt * dphi) % TWO_PI


def theta_neuron_rk4_step(
    phases: jax.Array,
    eta: jax.Array,
    K: jax.Array,
    dt: float,
) -> jax.Array:
    """Single RK4 step of the theta neuron model."""

    def deriv(p: jax.Array) -> jax.Array:
        return _theta_deriv(p, eta, K)

    k1 = deriv(phases)
    k2 = deriv(phases + 0.5 * dt * k1)
    k3 = deriv(phases + 0.5 * dt * k2)
    k4 = deriv(phases + dt * k3)
    return (phases + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)) % TWO_PI


def theta_neuron_forward(
    phases: jax.Array,
    eta: jax.Array,
    K: jax.Array,
    dt: float,
    n_steps: int,
    method: str = "rk4",
) -> tuple[jax.Array, jax.Array]:
    """Run N steps of theta neuron dynamics.

    Args:
        phases: (N,) initial phases
        eta: (N,) excitability parameters
        K: (N, N) synaptic coupling
        dt: timestep
        n_steps: integration steps
        method: "rk4" or "euler"

    Returns:
        (final, trajectory) where trajectory is (n_steps, N)
    """
    step_fn = theta_neuron_rk4_step if method == "rk4" else theta_neuron_step

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = step_fn(carry, eta, K, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


class ThetaNeuronLayer(eqx.Module):
    """Differentiable theta neuron layer.

    Learnable parameters:
        K: (n, n) synaptic coupling matrix
        eta: (n,) excitability parameters

    Static config:
        n_steps, dt
    """

    K: jax.Array
    eta: jax.Array
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        n_steps: int = 50,
        dt: float = 0.01,
        K_scale: float = 0.1,
        eta_mean: float = -0.5,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        raw = K_scale * jax.random.normal(k1, (n, n))
        self.K = (raw + raw.T) / 2.0
        # Default η<0 (excitable regime)
        self.eta = eta_mean + 0.1 * jax.random.normal(k2, (n,))
        self.n_steps = n_steps
        self.dt = dt
        self.n = n

    @eqx.filter_jit
    def __call__(self, phases: jax.Array) -> jax.Array:
        """Run theta neuron dynamics on input phases."""
        final, _ = theta_neuron_forward(
            phases,
            self.eta,
            self.K,
            self.dt,
            self.n_steps,
        )
        return final

    @eqx.filter_jit
    def forward_with_trajectory(
        self,
        phases: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Run dynamics and return full trajectory."""
        return theta_neuron_forward(
            phases,
            self.eta,
            self.K,
            self.dt,
            self.n_steps,
        )
