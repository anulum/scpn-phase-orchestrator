# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor PPO training

"""PPO loss, training step, and epoch loops for the supervisor policy."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._shared import (
    _non_negative_float,
    _non_negative_int,
    _positive_float,
    _positive_int,
)
from ._types import (
    KuramotoSupervisorScenario,
    SupervisorPPOAux,
    SupervisorPPOBatch,
    SupervisorPPOTrainResult,
)
from .checkpoint import load_supervisor_ppo_checkpoint, save_supervisor_ppo_checkpoint
from .policy import (
    DifferentiableSupervisorPolicy,
    supervisor_action_log_prob,
    unpack_supervisor_action,
)

if TYPE_CHECKING:
    import optax


def ppo_supervisor_loss(
    policy: DifferentiableSupervisorPolicy,
    batch: SupervisorPPOBatch,
    *,
    clip_epsilon: float = 0.2,
    value_clip: float | None = None,
    value_weight: float = 0.5,
    entropy_weight: float = 0.01,
) -> tuple[jax.Array, SupervisorPPOAux]:
    """Clipped PPO objective for bounded differentiable supervisor actions.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    batch : SupervisorPPOBatch
        The PPO training batch.
    clip_epsilon : float
        PPO clipping epsilon.
    value_clip : float | None
        Value-function clip range, or ``None``.
    value_weight : float
        Weight of the value loss.
    entropy_weight : float
        Weight of the entropy bonus.

    Returns
    -------
    tuple[jax.Array, SupervisorPPOAux]
        The clipped PPO loss and its auxiliary metrics.
    """
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
    """Run one optax update using the clipped PPO supervisor objective.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    batch : SupervisorPPOBatch
        The PPO training batch.
    opt_state : Any
        The optax optimiser state.
    optimizer : optax.GradientTransformation
        The optax optimiser.
    clip_epsilon : float
        PPO clipping epsilon.
    value_clip : float | None
        Value-function clip range, or ``None``.
    value_weight : float
        Weight of the value loss.
    entropy_weight : float
        Weight of the entropy bonus.
    max_grad_norm : float | None
        Gradient-norm clip threshold, or ``None``.

    Returns
    -------
    tuple[DifferentiableSupervisorPolicy, Any, jax.Array]
        The updated policy, optimiser state, and loss.
    """

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
    entropy_schedule: tuple[float, ...] | None = None,
    max_grad_norm: float | None = None,
    kl_early_stop: float | None = None,
) -> tuple[DifferentiableSupervisorPolicy, Any, jax.Array, int]:
    """Run PPO training for multiple epochs with deterministic minibatching.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        Optimisable differentiable supervisor policy.
    batch : SupervisorPPOBatch
        Flattened PPO batch from rollout collection.
    key : jax.Array
        JAX PRNG key used only for shuffle ordering.
    opt_state : Any
        Optimiser state.
    optimizer : optax.GradientTransformation
        Optax optimiser transformation.
    n_epochs : int
        Number of passes over the dataset.
    minibatch_size : int
        Per-update minibatch size.
    clip_epsilon : float
        PPO clipping radius.
    value_weight : float
        Value-function loss coefficient.
    entropy_weight : float
        Entropy bonus coefficient.
    entropy_schedule : tuple[float, ...] | None
        Optional non-negative per-update entropy weights. When shorter than the number
        of updates, the final value is held.
    max_grad_norm : float | None
        Optional global gradient-norm clip radius.
    kl_early_stop : float | None
        Optional KL threshold for early stopping.

    Returns
    -------
    tuple[DifferentiableSupervisorPolicy, Any, jax.Array, int]
        (policy, opt_state, loss_history, n_updates).
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
        entropy_schedule=entropy_schedule,
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
    entropy_schedule: tuple[float, ...] | None = None,
    max_grad_norm: float | None = None,
    kl_early_stop: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> SupervisorPPOTrainResult:
    """Run PPO epochs and optionally checkpoint a deterministic resume state.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    batch : SupervisorPPOBatch
        The PPO training batch.
    key : jax.Array
        JAX PRNG key.
    opt_state : Any
        The optax optimiser state.
    optimizer : optax.GradientTransformation
        The optax optimiser.
    n_epochs : int
        Number of training epochs.
    checkpoint_dir : str | Path | None
        Directory for training checkpoints, or ``None``.
    resume : bool
        Whether to resume from a checkpoint.
    minibatch_size : int
        Minibatch size.
    clip_epsilon : float
        PPO clipping epsilon.
    value_clip : float | None
        Value-function clip range, or ``None``.
    value_weight : float
        Weight of the value loss.
    entropy_weight : float
        Weight of the entropy bonus.
    entropy_schedule : tuple[float, ...] | None
        Per-epoch entropy-weight schedule, or ``None``.
    max_grad_norm : float | None
        Gradient-norm clip threshold, or ``None``.
    kl_early_stop : float | None
        KL early-stopping threshold, or ``None``.
    metadata : dict[str, Any] | None
        Associated metadata mapping, or ``None``.

    Returns
    -------
    SupervisorPPOTrainResult
        The PPO training result.

    Raises
    ------
    ValueError
        If the inputs are invalid or inconsistent.
    """
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
            entropy_schedule=entropy_schedule,
            max_grad_norm=max_grad_norm,
            kl_early_stop=kl_early_stop,
            initial_update_index=prior_updates,
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
    entropy_schedule: tuple[float, ...] | None = None,
    max_grad_norm: float | None = None,
    kl_early_stop: float | None = None,
    initial_update_index: int = 0,
) -> tuple[DifferentiableSupervisorPolicy, Any, jax.Array, int, jax.Array]:
    n_epochs = _positive_int(n_epochs, "n_epochs")
    batch_size = int(batch.phases.shape[0])
    minibatch_size = _positive_int(minibatch_size, "minibatch_size")
    initial_update_index = _non_negative_int(
        initial_update_index,
        "initial_update_index",
    )
    entropy_schedule_values = _non_negative_float_sequence(
        entropy_schedule,
        "entropy_schedule",
    )
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
            update_index = initial_update_index + n_updates
            current_entropy_weight = _scheduled_scalar(
                entropy_weight,
                entropy_schedule_values,
                update_index,
            )

            policy, opt_state, loss = ppo_supervisor_train_step(
                policy,
                batch_slice,
                opt_state,
                optimizer,
                clip_epsilon=clip_epsilon,
                value_clip=value_clip,
                value_weight=value_weight,
                entropy_weight=current_entropy_weight,
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
                    entropy_weight=current_entropy_weight,
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


def _non_negative_float_sequence(
    values: tuple[float, ...] | None,
    field: str,
) -> tuple[float, ...] | None:
    if values is None:
        return None
    if not values:
        raise ValueError(f"{field} must contain at least one value")
    return tuple(
        _non_negative_float(value, f"{field}[{index}]")
        for index, value in enumerate(values)
    )


def _scheduled_scalar(
    default: float,
    schedule: tuple[float, ...] | None,
    update_index: int,
) -> float:
    if schedule is None:
        return default
    return schedule[min(update_index, len(schedule) - 1)]
