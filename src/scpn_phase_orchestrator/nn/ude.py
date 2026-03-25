# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Universal Differential Equation Kuramoto

"""UDE-Kuramoto: physics backbone + learned neural residual.

dθ_i/dt = ω_i + Σ_j K_ij · [sin(θ_j - θ_i) + NN_φ(θ_j - θ_i)]

The known Kuramoto structure provides the mechanistic backbone. A small
neural network NN_φ handles model mismatch: higher harmonics, asymmetric
coupling, amplitude-dependent effects. Trained end-to-end via JAX autodiff.

Rackauckas et al. 2020 (UDE framework); Frontiers Comp. Neuro. 2025.
First Python UDE implementation for oscillator networks.

Requires: jax>=0.4, equinox>=0.11
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .functional import TWO_PI, order_parameter


class CouplingResidual(eqx.Module):
    """Small MLP that learns the residual coupling function.

    Maps phase difference Δθ → correction to sin(Δθ).
    """

    layers: list

    def __init__(self, hidden: int = 16, *, key: jax.Array) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, 1, key=k3),
        ]

    def __call__(self, delta_theta: jax.Array) -> jax.Array:
        """Evaluate residual for a single phase difference scalar.

        Args:
            delta_theta: scalar phase difference

        Returns:
            scalar correction
        """
        x = delta_theta[jnp.newaxis]  # (1,)
        x = jnp.tanh(self.layers[0](x))
        x = jnp.tanh(self.layers[1](x))
        x = self.layers[2](x)
        return x[0]


def _ude_deriv(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    residual_fn: CouplingResidual,
) -> jax.Array:
    """Derivative for UDE-Kuramoto: sin backbone + learned residual."""
    diff = phases[jnp.newaxis, :] - phases[:, jnp.newaxis]  # (N, N)

    # Known backbone: K * sin(Δθ)
    backbone = K * jnp.sin(diff)

    # Learned residual: K * NN_φ(Δθ) — applied per-pair via vmap
    residual_scalar = jax.vmap(jax.vmap(residual_fn))(diff)  # (N, N)
    correction = K * residual_scalar

    coupling = jnp.sum(backbone + correction, axis=1)
    return omegas + coupling


def ude_kuramoto_step(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    residual_fn: CouplingResidual,
    dt: float,
) -> jax.Array:
    """Single Euler step of UDE-Kuramoto.

    Args:
        phases: (N,) oscillator phases
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        residual_fn: learned coupling correction
        dt: integration timestep

    Returns:
        (N,) updated phases
    """
    dphi = _ude_deriv(phases, omegas, K, residual_fn)
    return (phases + dt * dphi) % TWO_PI


def ude_kuramoto_forward(
    phases: jax.Array,
    omegas: jax.Array,
    K: jax.Array,
    residual_fn: CouplingResidual,
    dt: float,
    n_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Run N steps of UDE-Kuramoto, returning final state and trajectory.

    Args:
        phases: (N,) initial phases
        omegas: (N,) natural frequencies
        K: (N, N) coupling matrix
        residual_fn: learned coupling correction
        dt: integration timestep
        n_steps: number of steps

    Returns:
        Tuple of (final_phases, trajectory)
    """

    def body(carry: jax.Array, _: None) -> tuple[jax.Array, jax.Array]:
        p = ude_kuramoto_step(carry, omegas, K, residual_fn, dt)
        return p, p

    final, trajectory = jax.lax.scan(body, phases, None, length=n_steps)
    return final, trajectory


class UDEKuramotoLayer(eqx.Module):
    """UDE-Kuramoto layer: physics backbone + learned residual.

    Learnable parameters:
        K: (n, n) coupling matrix
        omegas: (n,) natural frequencies
        residual: CouplingResidual MLP
    """

    K: jax.Array
    omegas: jax.Array
    residual: CouplingResidual
    n_steps: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    n: int = eqx.field(static=True)

    def __init__(
        self,
        n: int,
        n_steps: int = 50,
        dt: float = 0.01,
        K_scale: float = 0.1,
        hidden: int = 16,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        raw = K_scale * jax.random.normal(k1, (n, n))
        self.K = (raw + raw.T) / 2.0
        self.omegas = jax.random.normal(k2, (n,))
        self.residual = CouplingResidual(hidden=hidden, key=k3)
        self.n_steps = n_steps
        self.dt = dt
        self.n = n

    def __call__(self, phases: jax.Array) -> jax.Array:
        final, _ = ude_kuramoto_forward(
            phases, self.omegas, self.K, self.residual, self.dt, self.n_steps
        )
        return final

    def forward_with_trajectory(self, phases: jax.Array) -> tuple[jax.Array, jax.Array]:
        return ude_kuramoto_forward(
            phases, self.omegas, self.K, self.residual, self.dt, self.n_steps
        )

    def sync_score(self, phases: jax.Array) -> jax.Array:
        return order_parameter(self(phases))
