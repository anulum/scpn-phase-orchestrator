# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Differentiable supervisor policy for nn/

"""Differentiable supervisor policies for closed-loop Kuramoto control.

This module is the JAX/equinox counterpart to
``supervisor.policy.SupervisorPolicy``. It keeps the learning surface fully
differentiable and array-native, then exposes a small adapter for the existing
``ControlAction`` actuation path. Safety projection, rate limits, and live
interlocks remain outside the gradient path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from scpn_phase_orchestrator.actuation.mapper import ControlAction

from .functional import kuramoto_forward, order_parameter

if TYPE_CHECKING:
    import optax

__all__ = [
    "DifferentiableSupervisorConfig",
    "DifferentiableSupervisorPolicy",
    "SupervisorAction",
    "KuramotoSupervisorScenario",
    "SupervisorLossAux",
    "SupervisorPPOBatch",
    "SupervisorPPORollout",
    "SupervisorPPOAux",
    "SupervisorPPOCheckpoint",
    "SupervisorPPOTrainResult",
    "masked_order_parameter",
    "apply_supervisor_action",
    "closed_loop_supervisor_loss",
    "supervisor_train_step",
    "pack_supervisor_action",
    "unpack_supervisor_action",
    "sample_supervisor_action",
    "supervisor_action_log_prob",
    "ppo_supervisor_loss",
    "ppo_supervisor_train_step",
    "ppo_supervisor_train_epochs",
    "ppo_supervisor_train_with_checkpoint",
    "collect_supervisor_rollouts",
    "save_supervisor_ppo_checkpoint",
    "load_supervisor_ppo_checkpoint",
    "control_actions_from_supervisor",
]

_SUPERVISOR_PPO_CHECKPOINT_FORMAT = (
    "scpn_phase_orchestrator.nn.supervisor.ppo_checkpoint"
)
_SUPERVISOR_PPO_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DifferentiableSupervisorConfig:
    """Static configuration for ``DifferentiableSupervisorPolicy``.

    Args:
        n_oscillators: Number of oscillators in the controlled Kuramoto system.
        hidden_width: Width of each MLP hidden layer.
        hidden_depth: Number of hidden layers in the MLP.
        n_layer_controls: Number of mask-scoped ``K`` controls. The default
            maps to good and bad partitions in ``KuramotoSupervisorScenario``.
        max_global_delta_K: Absolute bound for global coupling increments.
        max_global_delta_zeta: Absolute bound for global damping/drive command.
        max_layer_delta_K: Absolute bound for partition-local coupling deltas.
        control_energy_weight: Quadratic penalty on control action magnitude.
        bad_sync_weight: Penalty for synchronising the bad partition.
        smoothness_weight: Quadratic penalty on action changes over rollout.
    """

    n_oscillators: int
    hidden_width: int = 32
    hidden_depth: int = 2
    n_layer_controls: int = 2
    max_global_delta_K: float = 0.05
    max_global_delta_zeta: float = 0.1
    max_layer_delta_K: float = 0.03
    control_energy_weight: float = 1.0e-2
    bad_sync_weight: float = 0.25
    smoothness_weight: float = 1.0e-3


class SupervisorAction(NamedTuple):
    """Continuous differentiable control emitted by the neural supervisor."""

    delta_K_global: jax.Array
    delta_zeta_global: jax.Array
    delta_K_layers: jax.Array
    value_estimate: jax.Array


class KuramotoSupervisorScenario(NamedTuple):
    """Closed-loop differentiable Kuramoto control problem.

    ``good_mask`` and ``bad_mask`` are non-negative oscillator membership
    weights. Binary masks are typical, but soft memberships are supported for
    differentiable curriculum construction.
    """

    phases: jax.Array
    omegas: jax.Array
    base_K: jax.Array
    good_mask: jax.Array
    bad_mask: jax.Array
    dt: float
    inner_steps: int
    horizon: int


class SupervisorLossAux(NamedTuple):
    """Diagnostics returned by ``closed_loop_supervisor_loss``."""

    final_R_good: jax.Array
    final_R_bad: jax.Array
    control_energy: jax.Array
    smoothness: jax.Array


class SupervisorPPOBatch(NamedTuple):
    """On-policy PPO batch for the differentiable supervisor.

    Arrays carry a leading batch dimension. ``actions`` must contain bounded
    continuous actions produced by ``pack_supervisor_action``.
    """

    phases: jax.Array
    omegas: jax.Array
    base_K: jax.Array
    good_mask: jax.Array
    bad_mask: jax.Array
    actions: jax.Array
    old_log_probs: jax.Array
    advantages: jax.Array
    returns: jax.Array
    values: jax.Array
    dt: float
    inner_steps: int
    horizon: int


class SupervisorPPORollout(NamedTuple):
    """Replay-only rollout outputs for PPO-style supervisor training."""

    batch: SupervisorPPOBatch
    episode_returns: jax.Array
    episode_return_mean: jax.Array
    episode_return_std: jax.Array


class SupervisorPPOAux(NamedTuple):
    """Diagnostics returned by ``ppo_supervisor_loss``."""

    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array


class SupervisorPPOCheckpoint(NamedTuple):
    """Loaded PPO checkpoint state for deterministic supervisor training resume."""

    policy: DifferentiableSupervisorPolicy
    opt_state: Any
    key: jax.Array
    loss_history: jax.Array
    n_updates: int
    metadata: dict[str, Any]


class SupervisorPPOTrainResult(NamedTuple):
    """PPO training result with checkpoint-resume bookkeeping."""

    policy: DifferentiableSupervisorPolicy
    opt_state: Any
    key: jax.Array
    loss_history: jax.Array
    n_updates: int
    checkpoint_path: Path | None


class _SupervisorPPOCheckpointPayload(NamedTuple):
    policy: DifferentiableSupervisorPolicy
    opt_state: Any
    key: jax.Array
    loss_history: jax.Array


class DifferentiableSupervisorPolicy(eqx.Module):
    """Equinox neural supervisor for differentiable Kuramoto control.

    The policy consumes a compact feature vector derived from phases, masks,
    and coupling statistics. Its output is a bounded continuous control action
    that can be differentiated through a full ``jax.lax.scan`` rollout.
    """

    network: eqx.nn.MLP
    log_std: jax.Array
    config: DifferentiableSupervisorConfig = eqx.field(static=True)

    def __init__(
        self,
        config: DifferentiableSupervisorConfig,
        *,
        key: jax.Array,
    ) -> None:
        self.config = config
        out_size = 3 + config.n_layer_controls
        self.log_std = jnp.full((2 + config.n_layer_controls,), -0.5)
        self.network = eqx.nn.MLP(
            in_size=8,
            out_size=out_size,
            width_size=config.hidden_width,
            depth=config.hidden_depth,
            activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, scenario: KuramotoSupervisorScenario) -> SupervisorAction:
        """Return a bounded continuous control action for ``scenario``."""
        action_mean, value = _policy_mean_and_value(self, scenario)
        return unpack_supervisor_action(
            jnp.tanh(action_mean) * _action_bounds(self.config),
            value_estimate=value,
            config=self.config,
        )


def masked_order_parameter(phases: jax.Array, weights: jax.Array) -> jax.Array:
    """Weighted Kuramoto order parameter for a partition of oscillators."""
    safe_weights = jnp.clip(weights, min=0.0)
    total = jnp.maximum(jnp.sum(safe_weights), 1.0e-12)
    z = jnp.sum(safe_weights * jnp.exp(1j * phases)) / total
    return jnp.abs(z)


def apply_supervisor_action(
    base_K: jax.Array,
    action: SupervisorAction,
    scenario: KuramotoSupervisorScenario,
) -> jax.Array:
    """Apply continuous supervisor output to a symmetric coupling matrix."""
    n = base_K.shape[0]
    offdiag = 1.0 - jnp.eye(n, dtype=base_K.dtype)
    K = base_K + action.delta_K_global * offdiag

    layer_masks = (scenario.good_mask, scenario.bad_mask)
    for idx, mask in enumerate(layer_masks[: action.delta_K_layers.shape[0]]):
        membership = jnp.outer(mask, mask) * offdiag
        K = K + action.delta_K_layers[idx] * membership

    K = 0.5 * (K + K.T)
    return K * offdiag


def closed_loop_supervisor_loss(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
) -> tuple[jax.Array, SupervisorLossAux]:
    """Differentiable closed-loop objective for Kuramoto supervisor training.

    The reward maximises good-partition synchrony while penalising bad-partition
    synchrony, control energy, and abrupt action changes. The returned value is
    a minimisation loss suitable for ``jax.grad`` or optax.
    """

    zero_action = SupervisorAction(
        delta_K_global=jnp.array(0.0),
        delta_zeta_global=jnp.array(0.0),
        delta_K_layers=jnp.zeros(policy.config.n_layer_controls),
        value_estimate=jnp.array(0.0),
    )

    def body(
        carry: tuple[jax.Array, SupervisorAction],
        _: None,
    ) -> tuple[tuple[jax.Array, SupervisorAction], tuple[jax.Array, jax.Array]]:
        phases, previous_action = carry
        step_scenario = scenario._replace(phases=phases)
        action = policy(step_scenario)
        controlled_K = apply_supervisor_action(scenario.base_K, action, scenario)
        final, _ = kuramoto_forward(
            phases,
            scenario.omegas,
            controlled_K,
            scenario.dt,
            scenario.inner_steps,
        )
        energy = _control_energy(action)
        smoothness = _action_distance(action, previous_action)
        return (final, action), (energy, smoothness)

    (final_phases, _), (energies, smoothnesses) = jax.lax.scan(
        body,
        (scenario.phases, zero_action),
        None,
        length=scenario.horizon,
    )
    final_R_good = masked_order_parameter(final_phases, scenario.good_mask)
    final_R_bad = masked_order_parameter(final_phases, scenario.bad_mask)
    control_energy = jnp.mean(energies)
    smoothness = jnp.mean(smoothnesses)
    reward = final_R_good - policy.config.bad_sync_weight * final_R_bad
    loss = (
        -reward
        + policy.config.control_energy_weight * control_energy
        + policy.config.smoothness_weight * smoothness
    )
    aux = SupervisorLossAux(
        final_R_good=final_R_good,
        final_R_bad=final_R_bad,
        control_energy=control_energy,
        smoothness=smoothness,
    )
    return loss, aux


def supervisor_train_step(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
) -> tuple[DifferentiableSupervisorPolicy, Any, jax.Array]:
    """Run one optax update for the differentiable supervisor objective."""

    def loss_fn(model: DifferentiableSupervisorPolicy) -> jax.Array:
        loss, _ = closed_loop_supervisor_loss(model, scenario)
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(policy)
    params = eqx.filter(policy, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated = eqx.apply_updates(policy, updates)
    return updated, opt_state, loss


def pack_supervisor_action(action: SupervisorAction) -> jax.Array:
    """Pack ``SupervisorAction`` controls into a flat continuous action vector."""
    return jnp.concatenate(
        [
            jnp.atleast_1d(action.delta_K_global),
            jnp.atleast_1d(action.delta_zeta_global),
            action.delta_K_layers,
        ]
    )


def unpack_supervisor_action(
    values: jax.Array,
    *,
    value_estimate: jax.Array,
    config: DifferentiableSupervisorConfig,
) -> SupervisorAction:
    """Unpack a flat action vector using ``config.n_layer_controls``."""
    return SupervisorAction(
        delta_K_global=values[0],
        delta_zeta_global=values[1],
        delta_K_layers=values[2 : 2 + config.n_layer_controls],
        value_estimate=value_estimate,
    )


def sample_supervisor_action(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    key: jax.Array,
) -> tuple[SupervisorAction, jax.Array]:
    """Sample a bounded squashed-Gaussian action and its log probability."""
    mean, value = _policy_mean_and_value(policy, scenario)
    std = jnp.exp(policy.log_std)
    pre_squash = mean + std * jax.random.normal(key, mean.shape)
    bounded = jnp.tanh(pre_squash) * _action_bounds(policy.config)
    action = unpack_supervisor_action(
        bounded,
        value_estimate=value,
        config=policy.config,
    )
    return action, _squashed_gaussian_log_prob(mean, policy.log_std, pre_squash)


def supervisor_action_log_prob(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    action: SupervisorAction,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return squashed-Gaussian log probability, entropy proxy, and value."""
    mean, value = _policy_mean_and_value(policy, scenario)
    bounds = _action_bounds(policy.config)
    scaled = jnp.clip(pack_supervisor_action(action) / bounds, -0.999999, 0.999999)
    pre_squash = jnp.arctanh(scaled)
    log_prob = _squashed_gaussian_log_prob(mean, policy.log_std, pre_squash)
    entropy = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * policy.log_std)
    return log_prob, entropy, value


def ppo_supervisor_loss(
    policy: DifferentiableSupervisorPolicy,
    batch: SupervisorPPOBatch,
    *,
    clip_epsilon: float = 0.2,
    value_clip: float | None = None,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> tuple[jax.Array, SupervisorPPOAux]:
    """Clipped PPO objective for bounded differentiable supervisor actions."""
    if value_clip is not None:
        value_clip = _non_negative_float(value_clip, "value_clip")

    def item_loss(
        phases: jax.Array,
        omegas: jax.Array,
        base_K: jax.Array,
        good_mask: jax.Array,
        bad_mask: jax.Array,
        action_values: jax.Array,
        old_log_prob: jax.Array,
        advantage: jax.Array,
        ret: jax.Array,
        old_value: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        scenario = KuramotoSupervisorScenario(
            phases=phases,
            omegas=omegas,
            base_K=base_K,
            good_mask=good_mask,
            bad_mask=bad_mask,
            dt=batch.dt,
            inner_steps=batch.inner_steps,
            horizon=batch.horizon,
        )
        action = unpack_supervisor_action(
            action_values,
            value_estimate=jnp.array(0.0),
            config=policy.config,
        )
        log_prob, entropy, value = supervisor_action_log_prob(policy, scenario, action)
        ratio = jnp.exp(log_prob - old_log_prob)
        unclipped = ratio * advantage
        clipped = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
        policy_loss = -jnp.minimum(unclipped, clipped)
        unclipped_value_loss = (value - ret) ** 2
        if value_clip is not None:
            clipped_value = old_value + jnp.clip(
                value - old_value, -value_clip, value_clip
            )
            clipped_value_loss = (clipped_value - ret) ** 2
            value_loss = jnp.maximum(unclipped_value_loss, clipped_value_loss)
        else:
            value_loss = unclipped_value_loss
        approx_kl = old_log_prob - log_prob
        clipped_flag = jnp.abs(ratio - 1.0) > clip_epsilon
        return (
            policy_loss,
            value_loss,
            entropy,
            approx_kl,
            clipped_flag.astype(jnp.float32),
        )

    policy_losses, value_losses, entropies, approx_kls, clip_flags = jax.vmap(
        item_loss
    )(
        batch.phases,
        batch.omegas,
        batch.base_K,
        batch.good_mask,
        batch.bad_mask,
        batch.actions,
        batch.old_log_probs,
        batch.advantages,
        batch.returns,
        batch.values,
    )
    policy_loss = jnp.mean(policy_losses)
    value_loss = jnp.mean(value_losses)
    entropy = jnp.mean(entropies)
    total = policy_loss + value_weight * value_loss - entropy_weight * entropy
    aux = SupervisorPPOAux(
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approx_kl=jnp.mean(approx_kls),
        clip_fraction=jnp.mean(clip_flags),
    )
    return total, aux


def ppo_supervisor_train_step(
    policy: DifferentiableSupervisorPolicy,
    batch: SupervisorPPOBatch,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    *,
    clip_epsilon: float = 0.2,
    value_clip: float | None = None,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    max_grad_norm: float | None = None,
) -> tuple[DifferentiableSupervisorPolicy, Any, jax.Array]:
    """Run one optax update using the clipped PPO supervisor objective."""

    def loss_fn(model: DifferentiableSupervisorPolicy) -> jax.Array:
        loss, _ = ppo_supervisor_loss(
            model,
            batch,
            clip_epsilon=clip_epsilon,
            value_clip=value_clip,
            value_weight=value_weight,
            entropy_weight=entropy_weight,
        )
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(policy)
    params = eqx.filter(policy, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    if max_grad_norm is not None:
        max_grad_norm = _positive_float(max_grad_norm, "max_grad_norm")
        updates = _clip_updates_by_global_norm(updates, max_grad_norm)
    updated = eqx.apply_updates(policy, updates)
    return updated, opt_state, loss


def ppo_supervisor_train_epochs(
    policy: DifferentiableSupervisorPolicy,
    batch: SupervisorPPOBatch,
    key: jax.Array,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    *,
    n_epochs: int,
    minibatch_size: int = 32,
    clip_epsilon: float = 0.2,
    value_clip: float | None = None,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    max_grad_norm: float | None = None,
    kl_early_stop: float | None = None,
) -> tuple[DifferentiableSupervisorPolicy, Any, jax.Array, int]:
    """Run PPO training for multiple epochs with deterministic minibatching.

    Args:
        policy: Optimisable differentiable supervisor policy.
        batch: Flattened PPO batch from rollout collection.
        key: JAX PRNG key used only for shuffle ordering.
        opt_state: Optimiser state.
        optimizer: Optax optimiser transformation.
        n_epochs: Number of passes over the dataset.
        minibatch_size: Per-update minibatch size.
        clip_epsilon: PPO clipping radius.
        value_weight: Value-function loss coefficient.
        entropy_weight: Entropy bonus coefficient.
        max_grad_norm: Optional global gradient-norm clip radius.
        kl_early_stop: Optional KL threshold for early stopping.

    Returns:
        (policy, opt_state, loss_history, n_updates)
    """
    policy, opt_state, loss_history, n_updates, _ = _ppo_supervisor_train_epochs_impl(
        policy,
        batch,
        key,
        opt_state,
        optimizer,
        n_epochs=n_epochs,
        minibatch_size=minibatch_size,
        clip_epsilon=clip_epsilon,
        value_clip=value_clip,
        value_weight=value_weight,
        entropy_weight=entropy_weight,
        max_grad_norm=max_grad_norm,
        kl_early_stop=kl_early_stop,
    )
    return policy, opt_state, loss_history, n_updates


def ppo_supervisor_train_with_checkpoint(
    policy: DifferentiableSupervisorPolicy,
    batch: SupervisorPPOBatch,
    key: jax.Array,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    *,
    n_epochs: int,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    minibatch_size: int = 32,
    clip_epsilon: float = 0.2,
    value_clip: float | None = None,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    max_grad_norm: float | None = None,
    kl_early_stop: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> SupervisorPPOTrainResult:
    """Run PPO epochs and optionally checkpoint a deterministic resume state."""
    prior_losses = jnp.asarray([])
    prior_updates = 0
    checkpoint_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_path = Path(checkpoint_dir)

    if resume:
        if checkpoint_path is None:
            raise ValueError("checkpoint_dir is required when resume=True")
        checkpoint = load_supervisor_ppo_checkpoint(
            checkpoint_path,
            template_policy=policy,
            template_opt_state=opt_state,
        )
        policy = checkpoint.policy
        opt_state = checkpoint.opt_state
        key = checkpoint.key
        prior_losses = checkpoint.loss_history
        prior_updates = checkpoint.n_updates

    policy, opt_state, new_losses, n_updates, next_key = (
        _ppo_supervisor_train_epochs_impl(
            policy,
            batch,
            key,
            opt_state,
            optimizer,
            n_epochs=n_epochs,
            minibatch_size=minibatch_size,
            clip_epsilon=clip_epsilon,
            value_clip=value_clip,
            value_weight=value_weight,
            entropy_weight=entropy_weight,
            max_grad_norm=max_grad_norm,
            kl_early_stop=kl_early_stop,
        )
    )
    loss_history = (
        new_losses
        if prior_losses.size == 0
        else jnp.concatenate([prior_losses, new_losses])
    )
    total_updates = prior_updates + n_updates

    if checkpoint_path is not None:
        save_supervisor_ppo_checkpoint(
            checkpoint_path,
            policy=policy,
            opt_state=opt_state,
            key=next_key,
            n_updates=total_updates,
            loss_history=loss_history,
            metadata=metadata,
            overwrite=True,
        )

    return SupervisorPPOTrainResult(
        policy=policy,
        opt_state=opt_state,
        key=next_key,
        loss_history=loss_history,
        n_updates=total_updates,
        checkpoint_path=checkpoint_path,
    )


def _ppo_supervisor_train_epochs_impl(
    policy: DifferentiableSupervisorPolicy,
    batch: SupervisorPPOBatch,
    key: jax.Array,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
    *,
    n_epochs: int,
    minibatch_size: int = 32,
    clip_epsilon: float = 0.2,
    value_clip: float | None = None,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
    max_grad_norm: float | None = None,
    kl_early_stop: float | None = None,
) -> tuple[DifferentiableSupervisorPolicy, Any, jax.Array, int, jax.Array]:
    n_epochs = _positive_int(n_epochs, "n_epochs")
    batch_size = int(batch.phases.shape[0])
    minibatch_size = _positive_int(minibatch_size, "minibatch_size")
    if batch_size <= 0:
        raise ValueError("batch.phases must contain at least one step")
    if minibatch_size > batch_size:
        raise ValueError("minibatch_size cannot exceed batch size")
    if kl_early_stop is not None:
        kl_early_stop = _positive_float(kl_early_stop, "kl_early_stop")
    _validate_supervisor_ppo_batch_size(batch, batch_size)

    updates: list[jax.Array] = []
    n_updates = 0
    current_key = key
    for _epoch in range(n_epochs):
        epoch_key, current_key = jax.random.split(current_key)
        indices = jax.random.permutation(epoch_key, batch_size)

        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            batch_idx = indices[start:end]
            batch_slice = _take_batch_rows(batch, batch_idx)

            policy, opt_state, loss = ppo_supervisor_train_step(
                policy,
                batch_slice,
                opt_state,
                optimizer,
                clip_epsilon=clip_epsilon,
                value_clip=value_clip,
                value_weight=value_weight,
                entropy_weight=entropy_weight,
                max_grad_norm=max_grad_norm,
            )
            updates.append(loss)
            n_updates += 1

            if kl_early_stop is not None:
                _, aux = ppo_supervisor_loss(
                    policy,
                    batch_slice,
                    clip_epsilon=clip_epsilon,
                    value_clip=value_clip,
                    value_weight=value_weight,
                    entropy_weight=entropy_weight,
                )
                if float(jnp.abs(aux.approx_kl)) > kl_early_stop:
                    return (
                        policy,
                        opt_state,
                        jnp.asarray(updates),
                        n_updates,
                        current_key,
                    )

    return (
        policy,
        opt_state,
        jnp.asarray(updates),
        n_updates,
        current_key,
    )


def collect_supervisor_rollouts(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
    *,
    key: jax.Array,
    n_episodes: int,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    trajectory_jitter: float = 0.0,
) -> SupervisorPPORollout:
    """Collect deterministic, replay-only PPO rollouts from a starting scenario."""
    n_episodes = _positive_int(n_episodes, "n_episodes")
    horizon = _positive_int(scenario.horizon, "scenario.horizon")
    inner_steps = _positive_int(scenario.inner_steps, "scenario.inner_steps")
    gamma = _bounded_unit_scalar(gamma, "gamma")
    gae_lambda = _bounded_unit_scalar(gae_lambda, "gae_lambda")
    if isinstance(trajectory_jitter, bool) or not isinstance(
        trajectory_jitter, int | float
    ):
        raise ValueError("trajectory_jitter must be a finite float")
    trajectory_jitter = float(trajectory_jitter)
    if not isfinite(trajectory_jitter) or trajectory_jitter < 0.0:
        raise ValueError("trajectory_jitter must be non-negative")

    rollout_keys = jax.random.split(key, n_episodes + 1)[1:]
    all_phases: list[jax.Array] = []
    all_omegas: list[jax.Array] = []
    all_base_k: list[jax.Array] = []
    all_good_masks: list[jax.Array] = []
    all_bad_masks: list[jax.Array] = []
    all_actions: list[jax.Array] = []
    all_old_log_probs: list[jax.Array] = []
    all_advantages: list[jax.Array] = []
    all_returns: list[jax.Array] = []
    all_values: list[jax.Array] = []
    episode_returns: list[jax.Array] = []

    for episode_key in rollout_keys:
        state = scenario
        if trajectory_jitter > 0.0:
            jitter_key, step_key = jax.random.split(episode_key)
            state = state._replace(
                phases=state.phases
                + trajectory_jitter
                * jax.random.normal(jitter_key, shape=state.phases.shape),
            )
            step_key = jax.random.fold_in(step_key, 1)
        else:
            step_key = episode_key

        step_rewards: list[jax.Array] = []
        step_values: list[jax.Array] = []
        step_actions: list[jax.Array] = []
        step_log_probs: list[jax.Array] = []
        step_phases: list[jax.Array] = []

        for _ in range(horizon):
            current_key, step_key = jax.random.split(step_key)
            action, action_log_prob = sample_supervisor_action(
                policy, state, key=current_key
            )
            controlled_K = apply_supervisor_action(scenario.base_K, action, state)
            next_phases, _ = kuramoto_forward(
                state.phases,
                state.omegas,
                controlled_K,
                state.dt,
                inner_steps,
            )
            reward = masked_order_parameter(next_phases, state.good_mask) - (
                policy.config.bad_sync_weight
                * masked_order_parameter(next_phases, state.bad_mask)
            )

            step_phases.append(state.phases)
            step_rewards.append(reward)
            step_values.append(action.value_estimate)
            step_actions.append(pack_supervisor_action(action))
            step_log_probs.append(action_log_prob)
            state = state._replace(phases=next_phases)

        terminal_value = _policy_mean_and_value(policy, state)[1]
        trajectory_values = jnp.stack(step_values + [terminal_value])
        rewards = jnp.stack(step_rewards)
        deltas = rewards + gamma * trajectory_values[1:] - trajectory_values[:-1]

        episode_advantages = [jnp.array(0.0)] * horizon
        running_advantage = jnp.array(0.0)
        for index in reversed(range(horizon)):
            running_advantage = deltas[index] + gamma * gae_lambda * running_advantage
            episode_advantages[index] = running_advantage
        advantages = jnp.stack(episode_advantages)
        returns = advantages + trajectory_values[:-1]

        episode_returns.append(jnp.sum(rewards))
        for index in range(horizon):
            all_phases.append(step_phases[index])
            all_omegas.append(scenario.omegas)
            all_base_k.append(scenario.base_K)
            all_good_masks.append(scenario.good_mask)
            all_bad_masks.append(scenario.bad_mask)
            all_actions.append(step_actions[index])
            all_old_log_probs.append(step_log_probs[index])
            all_advantages.append(advantages[index])
            all_returns.append(returns[index])
            all_values.append(step_values[index])

    episode_returns_array = jnp.stack(episode_returns)
    batch = SupervisorPPOBatch(
        phases=jnp.stack(all_phases),
        omegas=jnp.stack(all_omegas),
        base_K=jnp.stack(all_base_k),
        good_mask=jnp.stack(all_good_masks),
        bad_mask=jnp.stack(all_bad_masks),
        actions=jnp.stack(all_actions),
        old_log_probs=jnp.stack(all_old_log_probs),
        advantages=jnp.stack(all_advantages),
        returns=jnp.stack(all_returns),
        values=jnp.stack(all_values),
        dt=scenario.dt,
        inner_steps=inner_steps,
        horizon=horizon,
    )
    return SupervisorPPORollout(
        batch=batch,
        episode_returns=episode_returns_array,
        episode_return_mean=jnp.mean(episode_returns_array),
        episode_return_std=jnp.std(episode_returns_array),
    )


def save_supervisor_ppo_checkpoint(
    checkpoint_dir: str | Path,
    *,
    policy: DifferentiableSupervisorPolicy,
    opt_state: Any,
    key: jax.Array,
    n_updates: int,
    loss_history: jax.Array,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Persist PPO supervisor trainer state for deterministic resume."""
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.exists() and not checkpoint_path.is_dir():
        raise NotADirectoryError(f"{checkpoint_path} is not a checkpoint directory")
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    state_path = checkpoint_path / "state.eqx"
    metadata_path = checkpoint_path / "metadata.json"
    if not overwrite and (state_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"{checkpoint_path} already contains a supervisor PPO checkpoint"
        )

    n_updates = _non_negative_int(n_updates, "n_updates")
    key = jnp.asarray(key)
    loss_history = jnp.asarray(loss_history)
    user_metadata = _json_object(metadata, "metadata")
    payload = _SupervisorPPOCheckpointPayload(
        policy=policy,
        opt_state=opt_state,
        key=key,
        loss_history=loss_history,
    )
    checkpoint_metadata = {
        "format": _SUPERVISOR_PPO_CHECKPOINT_FORMAT,
        "schema_version": _SUPERVISOR_PPO_CHECKPOINT_SCHEMA_VERSION,
        "n_updates": n_updates,
        "key_shape": list(key.shape),
        "key_dtype": str(key.dtype),
        "loss_history_shape": list(loss_history.shape),
        "loss_history_dtype": str(loss_history.dtype),
        "metadata": user_metadata,
    }

    state_tmp = checkpoint_path / "state.eqx.tmp"
    metadata_tmp = checkpoint_path / "metadata.json.tmp"
    eqx.tree_serialise_leaves(state_tmp, payload)
    metadata_tmp.write_text(
        json.dumps(checkpoint_metadata, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    state_tmp.replace(state_path)
    metadata_tmp.replace(metadata_path)
    return checkpoint_path


def load_supervisor_ppo_checkpoint(
    checkpoint_dir: str | Path,
    *,
    template_policy: DifferentiableSupervisorPolicy,
    template_opt_state: Any,
) -> SupervisorPPOCheckpoint:
    """Load a PPO supervisor checkpoint against explicit policy/state templates."""
    checkpoint_path = Path(checkpoint_dir)
    metadata_path = checkpoint_path / "metadata.json"
    state_path = checkpoint_path / "state.eqx"
    metadata = _load_checkpoint_metadata(metadata_path)
    if not state_path.exists():
        raise FileNotFoundError(f"missing checkpoint payload: {state_path}")

    key_shape = _metadata_shape(metadata, "key_shape")
    loss_history_shape = _metadata_shape(metadata, "loss_history_shape")
    key_dtype = _metadata_dtype(metadata, "key_dtype")
    loss_history_dtype = _metadata_dtype(metadata, "loss_history_dtype")
    template_payload = _SupervisorPPOCheckpointPayload(
        policy=template_policy,
        opt_state=template_opt_state,
        key=jnp.zeros(key_shape, dtype=key_dtype),
        loss_history=jnp.zeros(loss_history_shape, dtype=loss_history_dtype),
    )
    loaded = eqx.tree_deserialise_leaves(state_path, template_payload)
    return SupervisorPPOCheckpoint(
        policy=loaded.policy,
        opt_state=loaded.opt_state,
        key=loaded.key,
        loss_history=loaded.loss_history,
        n_updates=_non_negative_int(metadata["n_updates"], "n_updates"),
        metadata=_json_object(metadata.get("metadata"), "checkpoint metadata"),
    )


def control_actions_from_supervisor(
    action: SupervisorAction,
    *,
    ttl_s: float = 5.0,
    include_layer_actions: bool = True,
) -> list[ControlAction]:
    """Convert a detached neural supervisor output into actuation commands."""
    actions = [
        ControlAction(
            knob="K",
            scope="global",
            value=float(action.delta_K_global),
            ttl_s=ttl_s,
            justification="differentiable supervisor: global coupling proposal",
        ),
        ControlAction(
            knob="zeta",
            scope="global",
            value=float(action.delta_zeta_global),
            ttl_s=ttl_s,
            justification="differentiable supervisor: damping proposal",
        ),
    ]
    if include_layer_actions:
        for idx, value in enumerate(action.delta_K_layers):
            actions.append(
                ControlAction(
                    knob="K",
                    scope=f"layer_{idx}",
                    value=float(value),
                    ttl_s=ttl_s,
                    justification=(
                        "differentiable supervisor: partition coupling proposal"
                    ),
                )
            )
    return actions


def _supervisor_features(scenario: KuramotoSupervisorScenario) -> jax.Array:
    R_global = order_parameter(scenario.phases)
    R_good = masked_order_parameter(scenario.phases, scenario.good_mask)
    R_bad = masked_order_parameter(scenario.phases, scenario.bad_mask)
    mean_omega = jnp.mean(scenario.omegas)
    std_omega = jnp.std(scenario.omegas)
    mean_K = jnp.mean(scenario.base_K)
    abs_K = jnp.mean(jnp.abs(scenario.base_K))
    phase_spread = 1.0 - R_global
    return jnp.array(
        [R_global, R_good, R_bad, phase_spread, mean_omega, std_omega, mean_K, abs_K]
    )


def _policy_mean_and_value(
    policy: DifferentiableSupervisorPolicy,
    scenario: KuramotoSupervisorScenario,
) -> tuple[jax.Array, jax.Array]:
    raw = policy.network(_supervisor_features(scenario))
    return raw[:-1], raw[-1]


def _action_bounds(config: DifferentiableSupervisorConfig) -> jax.Array:
    return jnp.concatenate(
        [
            jnp.array([config.max_global_delta_K, config.max_global_delta_zeta]),
            jnp.full((config.n_layer_controls,), config.max_layer_delta_K),
        ]
    )


def _squashed_gaussian_log_prob(
    mean: jax.Array,
    log_std: jax.Array,
    pre_squash: jax.Array,
) -> jax.Array:
    std = jnp.exp(log_std)
    normalised = (pre_squash - mean) / std
    gaussian = -0.5 * (normalised**2 + 2.0 * log_std + jnp.log(2.0 * jnp.pi))
    squash_correction = jnp.log(1.0 - jnp.tanh(pre_squash) ** 2 + 1.0e-6)
    return jnp.sum(gaussian - squash_correction)


def _control_energy(action: SupervisorAction) -> jax.Array:
    return (
        action.delta_K_global**2
        + action.delta_zeta_global**2
        + jnp.mean(action.delta_K_layers**2)
    )


def _action_distance(left: SupervisorAction, right: SupervisorAction) -> jax.Array:
    return (
        (left.delta_K_global - right.delta_K_global) ** 2
        + (left.delta_zeta_global - right.delta_zeta_global) ** 2
        + jnp.mean((left.delta_K_layers - right.delta_K_layers) ** 2)
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_supervisor_ppo_batch_size(
    batch: SupervisorPPOBatch,
    batch_size: int,
) -> None:
    fields = (
        "phases",
        "omegas",
        "base_K",
        "good_mask",
        "bad_mask",
        "actions",
        "old_log_probs",
        "advantages",
        "returns",
        "values",
    )
    for field_name in fields:
        value = getattr(batch, field_name)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"{field_name} must have first dimension {batch_size}, got"
                f" {value.shape[0]}"
            )


def _take_batch_rows(
    batch: SupervisorPPOBatch, indices: jax.Array
) -> SupervisorPPOBatch:
    return SupervisorPPOBatch(
        phases=batch.phases[indices],
        omegas=batch.omegas[indices],
        base_K=batch.base_K[indices],
        good_mask=batch.good_mask[indices],
        bad_mask=batch.bad_mask[indices],
        actions=batch.actions[indices],
        old_log_probs=batch.old_log_probs[indices],
        advantages=batch.advantages[indices],
        returns=batch.returns[indices],
        values=batch.values[indices],
        dt=batch.dt,
        inner_steps=batch.inner_steps,
        horizon=batch.horizon,
    )


def _global_l2_norm(values: Any) -> jax.Array:
    leaves = jax.tree.leaves(values)
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(
        jnp.sum(
            jnp.array([jnp.sum(jnp.square(leaf)) for leaf in leaves], dtype=jnp.float32)
        )
    )


def _clip_updates_by_global_norm(
    updates: Any,
    max_grad_norm: float,
) -> Any:
    norm = _global_l2_norm(updates)
    clip_scale = jnp.minimum(1.0, max_grad_norm / (norm + 1.0e-12))
    return jax.tree.map(lambda x: x * clip_scale, updates)


def _bounded_unit_scalar(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be in [0, 1]")
    value_float = float(value)
    if not isfinite(value_float) or value_float < 0.0 or value_float > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return value_float


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite positive scalar")
    value_float = float(value)
    if not isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{field} must be a finite positive scalar")
    return value_float


def _non_negative_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a finite non-negative scalar")
    value_float = float(value)
    if not isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{field} must be a finite non-negative scalar")
    return value_float


def _json_object(value: object, field: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    json.dumps(value, sort_keys=True, allow_nan=False)
    return dict(value)


def _load_checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing checkpoint metadata: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("checkpoint metadata is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    if (
        raw.get("format") != _SUPERVISOR_PPO_CHECKPOINT_FORMAT
        or raw.get("schema_version") != _SUPERVISOR_PPO_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError("checkpoint schema is not supported")
    return raw


def _metadata_shape(metadata: dict[str, Any], field: str) -> tuple[int, ...]:
    raw = metadata.get(field)
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a shape list")
    shape: list[int] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field}[{index}] must be a non-negative integer")
        shape.append(value)
    return tuple(shape)


def _metadata_dtype(metadata: dict[str, Any], field: str) -> jnp.dtype:
    raw = metadata.get(field)
    if not isinstance(raw, str):
        raise ValueError(f"{field} must be a dtype string")
    try:
        return jnp.dtype(raw)
    except TypeError as exc:
        raise ValueError(f"{field} is not a supported dtype") from exc
