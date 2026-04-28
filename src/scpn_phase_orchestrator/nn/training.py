# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Training utilities for differentiable nn/ layers

"""Loss functions, training loops, and data generation for nn/ layers.

Integrates with optax for optimisation. All loss functions are
differentiable via JAX autodiff and compatible with equinox modules.

Requires: jax>=0.4, equinox>=0.11, optax>=0.2
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from .functional import kuramoto_forward, order_parameter

# ──────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────


def sync_loss(
    model: eqx.Module,
    phases: jax.Array,
    target_R: float = 1.0,
) -> jax.Array:
    """Drive oscillator layer toward a target synchronisation level.

    Args:
        model: equinox layer with __call__(phases) → final_phases
        phases: (N,) initial phases
        target_R: target order parameter R in [0, 1]

    Returns:
        Scalar loss (R - target_R)^2
    """
    final = model(phases)  # type: ignore[operator]
    R = order_parameter(final)
    return (R - target_R) ** 2


def trajectory_loss(
    model: eqx.Module,
    phases: jax.Array,
    observed: jax.Array,
) -> jax.Array:
    """Fit model trajectory to observed phase data.

    Uses circular distance (via cos) to handle 2pi wrapping.

    Args:
        model: equinox layer with forward_with_trajectory(phases) → (final, traj)
        phases: (N,) initial phases
        observed: (T, N) observed phase trajectory

    Returns:
        Scalar mean circular distance
    """
    _, predicted = model.forward_with_trajectory(phases)  # type: ignore[attr-defined]
    T = min(predicted.shape[0], observed.shape[0])
    pred = predicted[:T]
    obs = observed[:T]
    return jnp.mean(1.0 - jnp.cos(pred - obs))


def coupling_sparsity_loss(
    K: jax.Array,
    target_density: float = 0.1,
) -> jax.Array:
    """L1 penalty driving K toward target density.

    Args:
        K: (N, N) coupling matrix
        target_density: fraction of nonzero entries desired

    Returns:
        Scalar penalty: |mean(|K|) - target_density * mean(|K|_initial)|
    """
    return jnp.mean(jnp.abs(K)) - target_density * jnp.mean(jnp.abs(K))


# ──────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────


def train_step(
    model: eqx.Module,
    loss_fn: Callable[[eqx.Module], jax.Array],
    opt_state: Any,
    optimizer: optax.GradientTransformation,
) -> tuple[eqx.Module, Any, jax.Array]:
    """Single optimisation step using optax.

    Args:
        model: equinox module to optimise
        loss_fn: callable(model) → scalar loss
        opt_state: optax optimiser state
        optimizer: optax optimiser (e.g. optax.adam(1e-3))

    Returns:
        (updated_model, updated_opt_state, loss_value)
    """
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train(
    model: eqx.Module,
    loss_fn: Callable[[eqx.Module], jax.Array],
    optimizer: optax.GradientTransformation,
    n_epochs: int,
    *,
    callback: Callable[[int, eqx.Module, jax.Array], None] | None = None,
) -> tuple[eqx.Module, list[float]]:
    """Full training loop.

    Args:
        model: equinox module to train
        loss_fn: callable(model) → scalar loss
        optimizer: optax optimiser
        n_epochs: number of training steps
        callback: optional fn(epoch, model, loss) called each step

    Returns:
        (trained_model, loss_history)
    """
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    losses: list[float] = []

    step_fn = eqx.filter_jit(
        lambda m, s: train_step(m, loss_fn, s, optimizer),
    )

    for epoch in range(n_epochs):
        model, opt_state, loss = step_fn(model, opt_state)
        loss_val = float(loss)
        losses.append(loss_val)
        if callback is not None:
            callback(epoch, model, loss)

    return model, losses


# ──────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────


def generate_kuramoto_data(
    N: int,
    T: int,
    dt: float = 0.01,
    K_scale: float = 0.3,
    *,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Generate synthetic Kuramoto trajectory with known ground truth.

    Args:
        N: number of oscillators
        T: number of timesteps
        dt: integration timestep
        K_scale: coupling matrix scale
        key: PRNG key

    Returns:
        (K_true, omegas_true, phases0, trajectory)
        where trajectory is (T, N)
    """
    k1, k2, k3 = jax.random.split(key, 3)
    raw = jax.random.normal(k1, (N, N)) * K_scale
    K_true = (raw + raw.T) / 2.0
    K_true = K_true.at[jnp.diag_indices(N)].set(0.0)
    omegas_true = jax.random.normal(k2, (N,)) * 0.3
    phases0 = jax.random.uniform(k3, (N,), maxval=2.0 * jnp.pi)
    _, trajectory = kuramoto_forward(phases0, omegas_true, K_true, dt, T)
    return K_true, omegas_true, phases0, trajectory


def generate_chimera_data(
    N: int,
    T: int,
    dt: float = 0.01,
    coupling_strength: float = 0.5,
    coupling_range: int = 4,
    *,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Generate chimera-producing Kuramoto dynamics on a ring.

    Uses non-local coupling on a 1D ring (Kuramoto & Battogtokh 2002)
    that produces coexistence of synchronised and incoherent domains.

    Args:
        N: number of oscillators on the ring
        T: number of timesteps
        dt: timestep
        coupling_strength: overall coupling scale
        coupling_range: number of neighbours on each side
        key: PRNG key

    Returns:
        (K, phases0, trajectory) where K is (N, N), trajectory is (T, N)
    """
    k1 = key
    # Non-local ring coupling: each oscillator couples to ±coupling_range neighbours
    K = jnp.zeros((N, N))
    for offset in range(1, coupling_range + 1):
        idx_fwd = jnp.arange(N)
        idx_back = jnp.arange(N)
        K = K.at[idx_fwd, (idx_fwd + offset) % N].set(coupling_strength / N)
        K = K.at[idx_back, (idx_back - offset) % N].set(coupling_strength / N)

    omegas = jnp.zeros(N)
    # Start with partially coherent state (chimera seed)
    phases0 = jnp.where(
        jnp.arange(N) < N // 2,
        0.1 * jax.random.normal(k1, (N,)),
        jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi),
    )
    _, trajectory = kuramoto_forward(phases0, omegas, K, dt, T)
    return K, phases0, trajectory
