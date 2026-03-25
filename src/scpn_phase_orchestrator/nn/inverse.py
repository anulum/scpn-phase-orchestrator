# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Gradient-based inverse Kuramoto

"""Infer coupling matrix K and natural frequencies ω from observed phases.

Given observed phase trajectories θ_i(t), optimize K and ω to minimize
the prediction error of the Kuramoto forward model. Supports L1 sparsity
on K for network topology discovery.

This is the inverse problem: data → model. The forward model is
differentiable via JAX autodiff, so we backpropagate through the ODE
solver to learn the parameters that produced the observed dynamics.

Requires: jax>=0.4
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .functional import kuramoto_forward


def inverse_loss(
    K: jax.Array,
    omegas: jax.Array,
    observed: jax.Array,
    dt: float,
    l1_weight: float = 0.0,
) -> jax.Array:
    """Loss for inverse Kuramoto: prediction error + optional L1 sparsity.

    Runs the forward model from observed[0] and compares the predicted
    trajectory against the observed trajectory.

    Args:
        K: (N, N) coupling matrix to optimize
        omegas: (N,) natural frequencies to optimize
        observed: (T, N) observed phase trajectory
        dt: integration timestep
        l1_weight: L1 penalty on K for sparsity (0 = no penalty)

    Returns:
        Scalar loss
    """
    n_steps = observed.shape[0] - 1
    initial = observed[0]

    _, predicted = kuramoto_forward(initial, omegas, K, dt, n_steps)

    # Phase-aware distance: use circular mean squared error
    # d(a, b) = 1 - cos(a - b), range [0, 2]
    diff = observed[1:] - predicted
    phase_error = jnp.mean(1.0 - jnp.cos(diff))

    loss = phase_error
    if l1_weight > 0.0:
        loss = loss + l1_weight * jnp.sum(jnp.abs(K))
    return loss


def infer_coupling(
    observed: jax.Array,
    dt: float,
    n_epochs: int = 200,
    lr: float = 0.01,
    l1_weight: float = 0.001,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, list[float]]:
    """Infer coupling matrix K and frequencies ω from observed phases.

    Gradient-based optimization of the forward Kuramoto model parameters
    to match observed phase trajectories.

    Args:
        observed: (T, N) observed phase trajectory
        dt: integration timestep used to generate the data
        n_epochs: optimization epochs
        lr: learning rate
        l1_weight: L1 sparsity penalty on K
        seed: random seed for initialization

    Returns:
        Tuple of (K, omegas, losses) where:
            K: (N, N) inferred coupling matrix
            omegas: (N,) inferred natural frequencies
            losses: list of loss values per epoch
    """
    N = observed.shape[1]
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    K = jax.random.normal(k1, (N, N)) * 0.1
    K = (K + K.T) / 2.0
    K = K.at[jnp.diag_indices(N)].set(0.0)
    omegas = jax.random.normal(k2, (N,))

    loss_and_grad = jax.value_and_grad(inverse_loss, argnums=(0, 1))
    losses: list[float] = []

    for _ in range(n_epochs):
        loss_val, (grad_K, grad_o) = loss_and_grad(K, omegas, observed, dt, l1_weight)
        K = K - lr * grad_K
        K = (K + K.T) / 2.0
        K = K.at[jnp.diag_indices(N)].set(0.0)
        omegas = omegas - lr * grad_o
        losses.append(float(loss_val))

    return K, omegas, losses


def coupling_correlation(K_true: jax.Array, K_inferred: jax.Array) -> jax.Array:
    """Pearson correlation between true and inferred coupling matrices.

    Args:
        K_true: (N, N) ground truth coupling
        K_inferred: (N, N) inferred coupling

    Returns:
        Scalar correlation in [-1, 1]
    """
    # Flatten upper triangle (exclude diagonal)
    N = K_true.shape[0]
    idx = jnp.triu_indices(N, k=1)
    a = K_true[idx]
    b = K_inferred[idx]
    a_centered = a - jnp.mean(a)
    b_centered = b - jnp.mean(b)
    num = jnp.sum(a_centered * b_centered)
    denom = jnp.sqrt(jnp.sum(a_centered**2) * jnp.sum(b_centered**2) + 1e-10)
    result: jax.Array = num / denom
    return result
