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
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .functional import kuramoto_forward, order_parameter

if TYPE_CHECKING:
    import optax

# ──────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────


def sync_loss(
    model: eqx.Module,
    phases: jax.Array,
    target_R: float = 1.0,
) -> jax.Array:
    """Drive oscillator layer toward a target synchronisation level.

    Parameters
    ----------
    model : eqx.Module
        equinox layer with __call__(phases) → final_phases.
    phases : jax.Array
        (N,) initial phases.
    target_R : float
        target order parameter R in [0, 1].

    Returns
    -------
    jax.Array
        Scalar loss (R - target_R)^2.
    """
    # type ignore: Equinox modules expose __call__ dynamically by subclass.
    final = model(phases)  # type: ignore[operator]
    R = order_parameter(final)
    return (R - target_R) ** 2


def trajectory_loss(
    model: eqx.Module,
    phases: jax.Array,
    observed: jax.Array,
    *,
    backend: str = "euler",
) -> jax.Array:
    """Fit model trajectory to observed phase data.

    Uses circular distance (via cos) to handle 2pi wrapping.

    Parameters
    ----------
    model : eqx.Module
        equinox layer with forward_with_trajectory(phases) → (final, traj).
    phases : jax.Array
        (N,) initial phases.
    observed : jax.Array
        (T, N) observed phase trajectory.
    backend : str
        Integration backend forwarded to the layer. ``"euler"`` (default)
        calls ``forward_with_trajectory(phases)`` unchanged, so any
        trajectory-capable layer works and existing hashes are preserved.
        ``"diffrax"`` requests the checkpointed continuous-adjoint path and
        therefore requires a backend-aware layer such as
        :class:`~scpn_phase_orchestrator.nn.ude.UDEKuramotoLayer`.

    Returns
    -------
    jax.Array
        Scalar mean circular distance.
    """
    # type ignore: training accepts the trajectory-capable Equinox protocol.
    if backend == "euler":
        _, predicted = model.forward_with_trajectory(phases)  # type: ignore[attr-defined]
    else:
        # type ignore: the backend-aware layer forwards the integration backend.
        _, predicted = model.forward_with_trajectory(  # type: ignore[attr-defined]
            phases, backend=backend
        )
    T = min(predicted.shape[0], observed.shape[0])
    pred = predicted[:T]
    obs = observed[:T]
    return jnp.mean(1.0 - jnp.cos(pred - obs))


def coupling_sparsity_loss(
    K: jax.Array,
    target_density: float = 0.1,
) -> jax.Array:
    """L1 penalty driving K toward target density.

    Parameters
    ----------
    K : jax.Array
        (N, N) coupling matrix.
    target_density : float
        fraction of nonzero entries desired.

    Returns
    -------
    jax.Array
        Scalar penalty: |mean(|K|) - target_density * mean(|K|_initial)|.
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

    Parameters
    ----------
    model : eqx.Module
        equinox module to optimise.
    loss_fn : Callable[[eqx.Module], jax.Array]
        callable(model) → scalar loss.
    opt_state : Any
        optax optimiser state.
    optimizer : optax.GradientTransformation
        optax optimiser (e.g. optax.adam(1e-3)).

    Returns
    -------
    tuple[eqx.Module, Any, jax.Array]
        (updated_model, updated_opt_state, loss_value).
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

    Parameters
    ----------
    model : eqx.Module
        equinox module to train.
    loss_fn : Callable[[eqx.Module], jax.Array]
        callable(model) → scalar loss.
    optimizer : optax.GradientTransformation
        optax optimiser.
    n_epochs : int
        number of training steps.
    callback : Callable[[int, eqx.Module, jax.Array], None] | None
        optional fn(epoch, model, loss) called each step.

    Returns
    -------
    tuple[eqx.Module, list[float]]
        (trained_model, loss_history).
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

    Parameters
    ----------
    N : int
        number of oscillators.
    T : int
        number of timesteps.
    dt : float
        integration timestep.
    K_scale : float
        coupling matrix scale.
    key : jax.Array
        PRNG key.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        (K_true, omegas_true, phases0, trajectory) where trajectory is (T, N).
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

    Parameters
    ----------
    N : int
        number of oscillators on the ring.
    T : int
        number of timesteps.
    dt : float
        timestep.
    coupling_strength : float
        overall coupling scale.
    coupling_range : int
        number of neighbours on each side.
    key : jax.Array
        PRNG key.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        (K, phases0, trajectory) where K is (N, N), trajectory is (T, N).
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
