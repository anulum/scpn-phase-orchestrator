# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — differentiable supervisor policy

"""Differentiable supervisor policy network and core action/loss operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from scpn_phase_orchestrator.actuation.mapper import ControlAction

from ..functional import kuramoto_forward
from ._shared import (
    _action_bounds,
    _control_energy,
    _policy_mean_and_value,
    _squashed_gaussian_log_prob,
    masked_order_parameter,
)
from ._types import (
    DifferentiableSupervisorConfig,
    KuramotoSupervisorScenario,
    SupervisorAction,
    SupervisorLossAux,
)

if TYPE_CHECKING:
    import optax


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


def apply_supervisor_action(
    base_K: jax.Array,
    action: SupervisorAction,
    scenario: KuramotoSupervisorScenario,
) -> jax.Array:
    """Apply continuous supervisor output to a symmetric coupling matrix.

    Parameters
    ----------
    base_K : jax.Array
        Base symmetric coupling matrix.
    action : SupervisorAction
        The supervisor control action.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.

    Returns
    -------
    jax.Array
        The modified symmetric coupling matrix.
    """
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

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.

    Returns
    -------
    tuple[jax.Array, SupervisorLossAux]
        The loss and its auxiliary metrics.
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
        final, _traj = kuramoto_forward(
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
    """Run one optax update for the differentiable supervisor objective.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    opt_state : Any
        The optax optimiser state.
    optimizer : optax.GradientTransformation
        The optax optimiser.

    Returns
    -------
    tuple[DifferentiableSupervisorPolicy, Any, jax.Array]
        The updated policy, optimiser state, and loss.
    """

    def loss_fn(model: DifferentiableSupervisorPolicy) -> jax.Array:
        loss, _ = closed_loop_supervisor_loss(model, scenario)
        return loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(policy)
    params = eqx.filter(policy, eqx.is_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    updated = eqx.apply_updates(policy, updates)
    return updated, opt_state, loss


def pack_supervisor_action(action: SupervisorAction) -> jax.Array:
    """Pack ``SupervisorAction`` controls into a flat continuous action vector.

    Parameters
    ----------
    action : SupervisorAction
        The supervisor control action.

    Returns
    -------
    jax.Array
        The flat continuous action vector.
    """
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
    """Unpack a flat action vector using ``config.n_layer_controls``.

    Parameters
    ----------
    values : jax.Array
        Flat packed action values.
    value_estimate : jax.Array
        The critic value estimate.
    config : DifferentiableSupervisorConfig
        The supervisor configuration.

    Returns
    -------
    SupervisorAction
        The reconstructed ``SupervisorAction``.
    """
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
    """Sample a bounded squashed-Gaussian action and its log probability.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    key : jax.Array
        JAX PRNG key.

    Returns
    -------
    tuple[SupervisorAction, jax.Array]
        The sampled action and its log probability.
    """
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
    """Return squashed-Gaussian log probability, entropy proxy, and value.

    Parameters
    ----------
    policy : DifferentiableSupervisorPolicy
        The differentiable supervisor policy.
    scenario : KuramotoSupervisorScenario
        The Kuramoto supervisor scenario.
    action : SupervisorAction
        The supervisor control action.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        The log probability, entropy proxy, and value.
    """
    mean, value = _policy_mean_and_value(policy, scenario)
    bounds = _action_bounds(policy.config)
    scaled = jnp.clip(pack_supervisor_action(action) / bounds, -0.999999, 0.999999)
    pre_squash = jnp.arctanh(scaled)
    log_prob = _squashed_gaussian_log_prob(mean, policy.log_std, pre_squash)
    entropy = 0.5 * jnp.sum(1.0 + jnp.log(2.0 * jnp.pi) + 2.0 * policy.log_std)
    return log_prob, entropy, value


def supervisor_action_bound_penalty(
    action: SupervisorAction,
    config: DifferentiableSupervisorConfig,
) -> jax.Array:
    """Differentiable quadratic penalty for proposals outside action bounds.

    Parameters
    ----------
    action : SupervisorAction
        The supervisor control action.
    config : DifferentiableSupervisorConfig
        The supervisor configuration.

    Returns
    -------
    jax.Array
        The quadratic out-of-bounds penalty.
    """
    values = pack_supervisor_action(action)
    bounds = _action_bounds(config)
    excess = jnp.maximum(jnp.abs(values) - bounds, 0.0)
    return jnp.sum(excess**2)


def control_actions_from_supervisor(
    action: SupervisorAction,
    *,
    ttl_s: float = 5.0,
    include_layer_actions: bool = True,
) -> list[ControlAction]:
    """Convert a detached neural supervisor output into actuation commands.

    Parameters
    ----------
    action : SupervisorAction
        The supervisor control action.
    ttl_s : float
        Action time-to-live in seconds.
    include_layer_actions : bool
        Whether to include per-layer actions.

    Returns
    -------
    list[ControlAction]
        The actuation control actions.
    """
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


def _action_distance(left: SupervisorAction, right: SupervisorAction) -> jax.Array:
    return (
        (left.delta_K_global - right.delta_K_global) ** 2
        + (left.delta_zeta_global - right.delta_zeta_global) ** 2
        + jnp.mean((left.delta_K_layers - right.delta_K_layers) ** 2)
    )
