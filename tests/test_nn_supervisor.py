# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for differentiable nn supervisor policy

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
optax = pytest.importorskip("optax")

from scpn_phase_orchestrator.actuation.mapper import ActuationMapper
from scpn_phase_orchestrator.binding.types import ActuatorMapping
from scpn_phase_orchestrator.nn.supervisor import (
    DifferentiableSupervisorConfig,
    DifferentiableSupervisorPolicy,
    KuramotoSupervisorScenario,
    SupervisorPPOBatch,
    apply_supervisor_action,
    closed_loop_supervisor_loss,
    control_actions_from_supervisor,
    pack_supervisor_action,
    ppo_supervisor_loss,
    ppo_supervisor_train_step,
    sample_supervisor_action,
    supervisor_train_step,
)


def _scenario() -> KuramotoSupervisorScenario:
    phases = jnp.array([0.0, 0.1, 2.7, 3.1])
    omegas = jnp.array([0.04, 0.03, -0.03, -0.04])
    base_K = jnp.full((4, 4), 0.03) - jnp.eye(4) * 0.03
    good_mask = jnp.array([1.0, 1.0, 0.0, 0.0])
    bad_mask = jnp.array([0.0, 0.0, 1.0, 1.0])
    return KuramotoSupervisorScenario(
        phases=phases,
        omegas=omegas,
        base_K=base_K,
        good_mask=good_mask,
        bad_mask=bad_mask,
        dt=0.02,
        inner_steps=4,
        horizon=3,
    )


def test_policy_outputs_bounded_continuous_controls() -> None:
    config = DifferentiableSupervisorConfig(
        n_oscillators=4,
        hidden_width=8,
        hidden_depth=2,
        max_global_delta_K=0.08,
        max_global_delta_zeta=0.05,
        max_layer_delta_K=0.04,
    )
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(1))
    action = policy(_scenario())

    assert action.delta_K_global.shape == ()
    assert action.delta_zeta_global.shape == ()
    assert action.delta_K_layers.shape == (2,)
    assert float(jnp.abs(action.delta_K_global)) <= config.max_global_delta_K
    assert float(jnp.abs(action.delta_zeta_global)) <= config.max_global_delta_zeta
    assert jnp.all(jnp.abs(action.delta_K_layers) <= config.max_layer_delta_K)
    assert jnp.isfinite(action.value_estimate)


def test_squashed_gaussian_sampler_returns_log_prob() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(11))
    scenario = _scenario()

    action, log_prob = sample_supervisor_action(
        policy, scenario, key=jax.random.PRNGKey(12)
    )

    assert pack_supervisor_action(action).shape == (2 + config.n_layer_controls,)
    assert jnp.isfinite(log_prob)
    assert float(jnp.abs(action.delta_K_global)) <= config.max_global_delta_K


def test_apply_supervisor_action_preserves_shape_symmetry_and_zero_diagonal() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(2))
    scenario = _scenario()
    action = policy(scenario)

    adjusted = apply_supervisor_action(scenario.base_K, action, scenario)

    assert adjusted.shape == scenario.base_K.shape
    np.testing.assert_allclose(adjusted, adjusted.T, atol=1e-7)
    np.testing.assert_allclose(jnp.diag(adjusted), jnp.zeros(4), atol=1e-7)
    assert jnp.all(jnp.isfinite(adjusted))


def test_closed_loop_loss_is_differentiable_through_policy() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(3))
    scenario = _scenario()

    loss, aux = closed_loop_supervisor_loss(policy, scenario)
    loss_value, grads = eqx.filter_value_and_grad(
        lambda p: closed_loop_supervisor_loss(p, scenario)[0]
    )(policy)

    assert jnp.isfinite(loss)
    assert jnp.isfinite(loss_value)
    assert 0.0 <= float(aux.final_R_good) <= 1.0
    assert 0.0 <= float(aux.final_R_bad) <= 1.0
    leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_array))
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_supervisor_train_step_updates_policy_with_optax() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(4))
    scenario = _scenario()
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    updated, updated_state, loss = supervisor_train_step(
        policy, scenario, opt_state, optimizer
    )

    assert jnp.isfinite(loss)
    assert updated_state is not opt_state
    before = jax.tree.leaves(eqx.filter(policy, eqx.is_array))
    after = jax.tree.leaves(eqx.filter(updated, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(before, after, strict=True))


def test_ppo_supervisor_loss_and_train_step_are_finite() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(13))
    scenario = _scenario()
    action, old_log_prob = sample_supervisor_action(
        policy, scenario, key=jax.random.PRNGKey(14)
    )
    batch = SupervisorPPOBatch(
        phases=jnp.stack([scenario.phases, scenario.phases + 0.05]),
        omegas=jnp.stack([scenario.omegas, scenario.omegas]),
        base_K=jnp.stack([scenario.base_K, scenario.base_K]),
        good_mask=jnp.stack([scenario.good_mask, scenario.good_mask]),
        bad_mask=jnp.stack([scenario.bad_mask, scenario.bad_mask]),
        actions=jnp.stack(
            [pack_supervisor_action(action), pack_supervisor_action(action)]
        ),
        old_log_probs=jnp.stack([old_log_prob, old_log_prob]),
        advantages=jnp.array([1.0, 0.5]),
        returns=jnp.array([0.2, 0.1]),
        dt=scenario.dt,
        inner_steps=scenario.inner_steps,
        horizon=scenario.horizon,
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    loss, aux = ppo_supervisor_loss(policy, batch)
    updated, updated_state, train_loss = ppo_supervisor_train_step(
        policy, batch, opt_state, optimizer
    )

    assert jnp.isfinite(loss)
    assert jnp.isfinite(aux.policy_loss)
    assert jnp.isfinite(aux.value_loss)
    assert jnp.isfinite(aux.entropy)
    assert jnp.isfinite(train_loss)
    assert updated_state is not opt_state
    before = jax.tree.leaves(eqx.filter(policy, eqx.is_array))
    after = jax.tree.leaves(eqx.filter(updated, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(before, after, strict=True))


def test_control_action_adapter_feeds_existing_mapper() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(5))
    action = policy(_scenario())

    actions = control_actions_from_supervisor(action, ttl_s=7.0)
    mapper = ActuationMapper(
        [
            ActuatorMapping(
                name="k_driver", knob="K", scope="global", limits=(-0.2, 0.2)
            ),
            ActuatorMapping(
                name="zeta_driver", knob="zeta", scope="global", limits=(-0.2, 0.2)
            ),
        ]
    )
    commands = mapper.map_actions(actions)

    assert {command["knob"] for command in commands} == {"K", "zeta"}
    assert all(command["ttl_s"] == 7.0 for command in commands)
