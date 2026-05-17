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
    SupervisorPPOCorpusRollout,
    SupervisorPPORollout,
    SupervisorScenarioCorpus,
    apply_supervisor_action,
    build_supervisor_scenario_corpus,
    closed_loop_supervisor_loss,
    collect_supervisor_corpus_rollouts,
    collect_supervisor_rollouts,
    control_actions_from_supervisor,
    load_supervisor_ppo_checkpoint,
    pack_supervisor_action,
    ppo_supervisor_loss,
    ppo_supervisor_train_epochs,
    ppo_supervisor_train_step,
    ppo_supervisor_train_with_checkpoint,
    sample_supervisor_action,
    save_supervisor_ppo_checkpoint,
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
        values=jnp.stack([action.value_estimate, action.value_estimate]),
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


def test_ppo_supervisor_train_epochs_minibatches_and_is_deterministic() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(30))
    rollout = collect_supervisor_rollouts(
        policy,
        _scenario(),
        key=jax.random.PRNGKey(31),
        n_episodes=2,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.0,
    )
    batch = rollout.batch
    optimizer = optax.adam(1e-3)
    minibatch_size = 4
    expected_updates = (
        (batch.phases.shape[0] + minibatch_size - 1) // minibatch_size
    ) * 2

    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))
    first_policy, _, first_losses, first_updates = ppo_supervisor_train_epochs(
        policy,
        batch,
        key=jax.random.PRNGKey(32),
        opt_state=opt_state,
        optimizer=optimizer,
        n_epochs=2,
        minibatch_size=minibatch_size,
        max_grad_norm=1.0,
    )

    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))
    second_policy, _, second_losses, second_updates = ppo_supervisor_train_epochs(
        policy,
        batch,
        key=jax.random.PRNGKey(32),
        opt_state=opt_state,
        optimizer=optimizer,
        n_epochs=2,
        minibatch_size=minibatch_size,
        max_grad_norm=1.0,
    )
    assert first_updates == expected_updates
    assert first_updates == second_updates
    assert first_policy is not policy
    assert second_policy is not policy
    assert jnp.allclose(first_losses, second_losses)
    first_before = eqx.filter(policy, eqx.is_array)
    first_after = eqx.filter(first_policy, eqx.is_array)
    second_after = eqx.filter(second_policy, eqx.is_array)
    assert any(
        not jnp.allclose(a, b)
        for a, b in zip(
            jax.tree.leaves(first_before),
            jax.tree.leaves(first_after),
            strict=True,
        )
    )
    assert all(
        jnp.allclose(a, b)
        for a, b in zip(
            jax.tree.leaves(first_after),
            jax.tree.leaves(second_after),
            strict=True,
        )
    )


def test_ppo_supervisor_train_step_supports_grad_clip() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(33))
    batch = SupervisorPPOBatch(
        phases=jnp.tile(_scenario().phases, (2, 1)),
        omegas=jnp.tile(_scenario().omegas, (2, 1)),
        base_K=jnp.tile(_scenario().base_K, (2, 1, 1)),
        good_mask=jnp.tile(_scenario().good_mask, (2, 1)),
        bad_mask=jnp.tile(_scenario().bad_mask, (2, 1)),
        values=jnp.zeros(2),
        actions=jnp.stack(
            [jnp.zeros(4)] * 2,
        ),
        old_log_probs=jnp.array([0.0, 0.0]),
        advantages=jnp.array([1.0, -1.0]),
        returns=jnp.array([0.1, 0.2]),
        dt=_scenario().dt,
        inner_steps=4,
        horizon=3,
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    updated, _, loss = ppo_supervisor_train_step(
        policy,
        batch,
        opt_state=opt_state,
        optimizer=optimizer,
        clip_epsilon=0.2,
        max_grad_norm=0.1,
    )

    assert jnp.isfinite(loss)
    updated_params = eqx.filter(updated, eqx.is_array)
    baseline_params = eqx.filter(policy, eqx.is_array)
    assert any(
        not jnp.allclose(a, b)
        for a, b in zip(
            jax.tree.leaves(baseline_params),
            jax.tree.leaves(updated_params),
            strict=True,
        )
    )


def test_ppo_supervisor_train_epochs_rejects_invalid_minibatch_size() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy_model = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(34))
    rollout = collect_supervisor_rollouts(
        policy_model,
        _scenario(),
        key=jax.random.PRNGKey(35),
        n_episodes=1,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.0,
    )
    batch = rollout.batch
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(policy_model, eqx.is_array))

    with pytest.raises(ValueError, match="minibatch_size"):
        ppo_supervisor_train_epochs(
            policy_model,
            batch,
            key=jax.random.PRNGKey(36),
            opt_state=opt_state,
            optimizer=optimizer,
            n_epochs=2,
            minibatch_size=0,
        )

    with pytest.raises(ValueError, match="minibatch_size cannot exceed batch size"):
        ppo_supervisor_train_epochs(
            policy_model,
            batch,
            key=jax.random.PRNGKey(36),
            opt_state=opt_state,
            optimizer=optimizer,
            n_epochs=1,
            minibatch_size=batch.phases.shape[0] + 1,
        )


def test_supervisor_ppo_checkpoint_round_trip_preserves_next_update(tmp_path) -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(42))
    rollout = collect_supervisor_rollouts(
        policy,
        _scenario(),
        key=jax.random.PRNGKey(43),
        n_episodes=2,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.0,
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    trained, trained_opt_state, losses, n_updates = ppo_supervisor_train_epochs(
        policy,
        rollout.batch,
        key=jax.random.PRNGKey(44),
        opt_state=opt_state,
        optimizer=optimizer,
        n_epochs=1,
        minibatch_size=4,
        max_grad_norm=1.0,
    )
    save_supervisor_ppo_checkpoint(
        tmp_path / "checkpoint",
        policy=trained,
        opt_state=trained_opt_state,
        key=jax.random.PRNGKey(45),
        n_updates=n_updates,
        loss_history=losses,
        metadata={"experiment": "unit-round-trip"},
    )

    template_policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(46))
    template_opt_state = optimizer.init(eqx.filter(template_policy, eqx.is_array))
    restored = load_supervisor_ppo_checkpoint(
        tmp_path / "checkpoint",
        template_policy=template_policy,
        template_opt_state=template_opt_state,
    )

    assert restored.n_updates == n_updates
    assert restored.metadata == {"experiment": "unit-round-trip"}
    assert jnp.array_equal(restored.key, jax.random.PRNGKey(45))
    assert jnp.array_equal(restored.loss_history, losses)
    for before, after in zip(
        jax.tree.leaves(eqx.filter(trained, eqx.is_array)),
        jax.tree.leaves(eqx.filter(restored.policy, eqx.is_array)),
        strict=True,
    ):
        assert jnp.array_equal(before, after)
    for before, after in zip(
        jax.tree.leaves(trained_opt_state),
        jax.tree.leaves(restored.opt_state),
        strict=True,
    ):
        assert jnp.array_equal(before, after)

    continued, continued_state, continued_loss = ppo_supervisor_train_step(
        trained,
        rollout.batch,
        opt_state=trained_opt_state,
        optimizer=optimizer,
        max_grad_norm=1.0,
    )
    restored_continued, restored_state, restored_loss = ppo_supervisor_train_step(
        restored.policy,
        rollout.batch,
        opt_state=restored.opt_state,
        optimizer=optimizer,
        max_grad_norm=1.0,
    )

    assert jnp.array_equal(continued_loss, restored_loss)
    for before, after in zip(
        jax.tree.leaves(eqx.filter(continued, eqx.is_array)),
        jax.tree.leaves(eqx.filter(restored_continued, eqx.is_array)),
        strict=True,
    ):
        assert jnp.array_equal(before, after)
    for before, after in zip(
        jax.tree.leaves(continued_state),
        jax.tree.leaves(restored_state),
        strict=True,
    ):
        assert jnp.array_equal(before, after)


def test_supervisor_ppo_checkpoint_rejects_malformed_metadata(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "metadata.json").write_text(
        '{"schema_version": 999, "format": "wrong"}',
        encoding="utf-8",
    )

    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    template_policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(47))
    optimizer = optax.adam(1e-3)
    template_opt_state = optimizer.init(eqx.filter(template_policy, eqx.is_array))

    with pytest.raises(ValueError, match="checkpoint schema"):
        load_supervisor_ppo_checkpoint(
            checkpoint_dir,
            template_policy=template_policy,
            template_opt_state=template_opt_state,
        )


def test_supervisor_ppo_checkpoint_trainer_resume_matches_direct_epochs(
    tmp_path,
) -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(48))
    rollout = collect_supervisor_rollouts(
        policy,
        _scenario(),
        key=jax.random.PRNGKey(49),
        n_episodes=2,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.0,
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    direct_policy, direct_state, direct_losses, direct_updates = (
        ppo_supervisor_train_epochs(
            policy,
            rollout.batch,
            key=jax.random.PRNGKey(50),
            opt_state=opt_state,
            optimizer=optimizer,
            n_epochs=2,
            minibatch_size=4,
            max_grad_norm=1.0,
        )
    )
    checkpoint_dir = tmp_path / "resume-checkpoint"
    first = ppo_supervisor_train_with_checkpoint(
        policy,
        rollout.batch,
        key=jax.random.PRNGKey(50),
        opt_state=opt_state,
        optimizer=optimizer,
        n_epochs=1,
        minibatch_size=4,
        max_grad_norm=1.0,
        checkpoint_dir=checkpoint_dir,
        metadata={"experiment": "resume-wrapper"},
    )
    second = ppo_supervisor_train_with_checkpoint(
        policy,
        rollout.batch,
        key=jax.random.PRNGKey(999),
        opt_state=opt_state,
        optimizer=optimizer,
        n_epochs=1,
        minibatch_size=4,
        max_grad_norm=1.0,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        metadata={"experiment": "resume-wrapper"},
    )

    assert first.checkpoint_path == checkpoint_dir
    assert second.checkpoint_path == checkpoint_dir
    assert second.n_updates == direct_updates
    assert jnp.array_equal(second.loss_history, direct_losses)
    for before, after in zip(
        jax.tree.leaves(eqx.filter(direct_policy, eqx.is_array)),
        jax.tree.leaves(eqx.filter(second.policy, eqx.is_array)),
        strict=True,
    ):
        assert jnp.array_equal(before, after)
    for before, after in zip(
        jax.tree.leaves(direct_state),
        jax.tree.leaves(second.opt_state),
        strict=True,
    ):
        assert jnp.array_equal(before, after)


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


def test_collect_supervisor_rollouts_is_deterministic_and_returns_valid_batch() -> None:
    config = DifferentiableSupervisorConfig(
        n_oscillators=4,
        hidden_width=8,
        hidden_depth=2,
    )
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(21))
    scenario = _scenario()

    rollout_a = collect_supervisor_rollouts(
        policy,
        scenario,
        key=jax.random.PRNGKey(31),
        n_episodes=3,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.01,
    )
    rollout_b = collect_supervisor_rollouts(
        policy,
        scenario,
        key=jax.random.PRNGKey(31),
        n_episodes=3,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.01,
    )

    assert isinstance(rollout_a, SupervisorPPORollout)
    assert rollout_a.episode_returns.shape == (3,)
    assert rollout_a.batch.phases.shape == (9, 4)
    assert rollout_a.batch.omegas.shape == (9, 4)
    assert rollout_a.batch.base_K.shape == (9, 4, 4)
    assert rollout_a.batch.good_mask.shape == (9, 4)
    assert rollout_a.batch.bad_mask.shape == (9, 4)
    assert rollout_a.batch.actions.shape == (9, 4)
    assert rollout_a.batch.old_log_probs.shape == (9,)
    assert rollout_a.batch.advantages.shape == (9,)
    assert rollout_a.batch.returns.shape == (9,)
    assert rollout_a.batch.values.shape == (9,)
    assert jnp.isfinite(rollout_a.batch.phases).all()
    assert jnp.isfinite(rollout_a.batch.returns).all()
    assert jnp.isfinite(rollout_a.batch.values).all()
    assert float(rollout_a.episode_return_std) >= 0.0
    assert jnp.array_equal(rollout_a.batch.phases, rollout_b.batch.phases)
    assert jnp.array_equal(rollout_a.episode_returns, rollout_b.episode_returns)


def test_collect_supervisor_rollouts_rejects_invalid_hyperparameters() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(22))
    scenario = _scenario()

    with pytest.raises(ValueError, match="n_episodes"):
        collect_supervisor_rollouts(
            policy,
            scenario,
            key=jax.random.PRNGKey(33),
            n_episodes=0,
        )

    with pytest.raises(ValueError, match="gamma"):
        collect_supervisor_rollouts(
            policy,
            scenario,
            key=jax.random.PRNGKey(33),
            n_episodes=1,
            gamma=-0.01,
        )

    with pytest.raises(ValueError, match="gae_lambda"):
        collect_supervisor_rollouts(
            policy,
            scenario,
            key=jax.random.PRNGKey(33),
            n_episodes=1,
            gae_lambda=1.2,
        )


def test_ppo_value_loss_uses_optional_value_clipping() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(40))
    scenario = _scenario()
    action, old_log_prob = sample_supervisor_action(
        policy, scenario, key=jax.random.PRNGKey(41)
    )

    batch = SupervisorPPOBatch(
        phases=jnp.tile(scenario.phases, (2, 1)),
        omegas=jnp.tile(scenario.omegas, (2, 1)),
        base_K=jnp.tile(scenario.base_K, (2, 1, 1)),
        good_mask=jnp.tile(scenario.good_mask, (2, 1)),
        bad_mask=jnp.tile(scenario.bad_mask, (2, 1)),
        actions=jnp.tile(pack_supervisor_action(action), (2, 1)),
        old_log_probs=jnp.array([old_log_prob, old_log_prob]),
        advantages=jnp.zeros(2),
        returns=jnp.array([action.value_estimate + 0.25, action.value_estimate - 0.25]),
        values=jnp.array([action.value_estimate + 2.5, action.value_estimate - 2.5]),
        dt=scenario.dt,
        inner_steps=scenario.inner_steps,
        horizon=scenario.horizon,
    )

    _, unclipped_aux = ppo_supervisor_loss(
        policy,
        batch,
        clip_epsilon=0.2,
        value_clip=None,
    )
    _, clipped_aux = ppo_supervisor_loss(
        policy,
        batch,
        clip_epsilon=0.2,
        value_clip=0.0,
    )

    assert jnp.isfinite(unclipped_aux.value_loss)
    assert jnp.isfinite(clipped_aux.value_loss)
    assert float(clipped_aux.value_loss) >= float(unclipped_aux.value_loss)


def test_supervisor_scenario_corpus_converts_replay_records() -> None:
    records = [
        {
            "phases": [0.0, 0.1, 2.7, 3.1],
            "omegas": [0.04, 0.03, -0.03, -0.04],
            "base_K": [
                [0.0, 0.03, 0.03, 0.03],
                [0.03, 0.0, 0.03, 0.03],
                [0.03, 0.03, 0.0, 0.03],
                [0.03, 0.03, 0.03, 0.0],
            ],
            "good_mask": [1.0, 1.0, 0.0, 0.0],
            "bad_mask": [0.0, 0.0, 1.0, 1.0],
            "dt": 0.02,
            "inner_steps": 4,
            "horizon": 3,
            "metadata": {"source": "unit-replay", "stream_id": "fixture-1"},
        },
        {
            "phases": np.array([0.2, 0.0, 2.9, 3.0]),
            "omegas": np.array([0.05, 0.02, -0.02, -0.05]),
            "base_K": np.full((4, 4), 0.02) - np.eye(4) * 0.02,
            "good_mask": np.array([1.0, 1.0, 0.0, 0.0]),
            "bad_mask": np.array([0.0, 0.0, 1.0, 1.0]),
            "dt": 0.02,
            "inner_steps": 4,
            "horizon": 3,
        },
    ]

    corpus = build_supervisor_scenario_corpus(records)

    assert isinstance(corpus, SupervisorScenarioCorpus)
    assert len(corpus.scenarios) == 2
    assert corpus.metadata == (
        {"source": "unit-replay", "stream_id": "fixture-1"},
        {},
    )
    scenario = corpus.scenarios[0]
    assert isinstance(scenario, KuramotoSupervisorScenario)
    assert scenario.phases.shape == (4,)
    assert scenario.omegas.shape == (4,)
    assert scenario.base_K.shape == (4, 4)
    assert scenario.good_mask.shape == (4,)
    assert scenario.bad_mask.shape == (4,)
    assert scenario.dt == 0.02
    assert scenario.inner_steps == 4
    assert scenario.horizon == 3
    assert jnp.issubdtype(scenario.phases.dtype, jnp.floating)
    assert jnp.all(jnp.isfinite(scenario.phases))
    assert jnp.all(jnp.isfinite(scenario.base_K))


def test_collect_supervisor_corpus_rollouts_is_deterministic_and_traces_scenarios() -> (
    None
):
    records = [
        {
            "phases": [0.0, 0.1, 2.7, 3.1],
            "omegas": [0.04, 0.03, -0.03, -0.04],
            "base_K": [
                [0.0, 0.03, 0.03, 0.03],
                [0.03, 0.0, 0.03, 0.03],
                [0.03, 0.03, 0.0, 0.03],
                [0.03, 0.03, 0.03, 0.0],
            ],
            "good_mask": [1.0, 1.0, 0.0, 0.0],
            "bad_mask": [0.0, 0.0, 1.0, 1.0],
            "dt": 0.02,
            "inner_steps": 4,
            "horizon": 3,
            "metadata": {"source": "unit-replay", "stream_id": "fixture-a"},
        },
        {
            "phases": [0.2, 0.0, 2.9, 3.0],
            "omegas": [0.05, 0.02, -0.02, -0.05],
            "base_K": np.full((4, 4), 0.02) - np.eye(4) * 0.02,
            "good_mask": [1.0, 1.0, 0.0, 0.0],
            "bad_mask": [0.0, 0.0, 1.0, 1.0],
            "dt": 0.02,
            "inner_steps": 4,
            "horizon": 3,
            "metadata": {"source": "unit-replay", "stream_id": "fixture-b"},
        },
    ]
    corpus = build_supervisor_scenario_corpus(records)
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(60))

    rollout_a = collect_supervisor_corpus_rollouts(
        policy,
        corpus,
        key=jax.random.PRNGKey(61),
        n_episodes_per_scenario=2,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.01,
    )
    rollout_b = collect_supervisor_corpus_rollouts(
        policy,
        corpus,
        key=jax.random.PRNGKey(61),
        n_episodes_per_scenario=2,
        gamma=0.97,
        gae_lambda=0.9,
        trajectory_jitter=0.01,
    )

    assert isinstance(rollout_a, SupervisorPPOCorpusRollout)
    assert rollout_a.batch.phases.shape == (12, 4)
    assert rollout_a.batch.actions.shape == (12, 4)
    assert rollout_a.batch.returns.shape == (12,)
    assert rollout_a.batch.dt == 0.02
    assert rollout_a.batch.inner_steps == 4
    assert rollout_a.batch.horizon == 3
    assert rollout_a.episode_returns.shape == (4,)
    assert rollout_a.scenario_indices.shape == (4,)
    assert jnp.array_equal(rollout_a.scenario_indices, jnp.array([0, 0, 1, 1]))
    assert rollout_a.metadata == (
        {"source": "unit-replay", "stream_id": "fixture-a"},
        {"source": "unit-replay", "stream_id": "fixture-b"},
    )
    assert jnp.isfinite(rollout_a.batch.phases).all()
    assert jnp.isfinite(rollout_a.episode_return_mean)
    assert float(rollout_a.episode_return_std) >= 0.0
    assert jnp.array_equal(rollout_a.batch.phases, rollout_b.batch.phases)
    assert jnp.array_equal(rollout_a.episode_returns, rollout_b.episode_returns)
    assert jnp.array_equal(rollout_a.scenario_indices, rollout_b.scenario_indices)


def test_collect_supervisor_corpus_rollouts_rejects_invalid_corpus_inputs() -> None:
    config = DifferentiableSupervisorConfig(n_oscillators=4, hidden_width=8)
    policy = DifferentiableSupervisorPolicy(config, key=jax.random.PRNGKey(62))
    scenario = _scenario()
    mismatched = scenario._replace(horizon=scenario.horizon + 1)
    corpus = SupervisorScenarioCorpus(
        scenarios=(scenario, mismatched),
        metadata=({}, {}),
    )

    with pytest.raises(ValueError, match="n_episodes_per_scenario"):
        collect_supervisor_corpus_rollouts(
            policy,
            SupervisorScenarioCorpus(scenarios=(scenario,), metadata=({},)),
            key=jax.random.PRNGKey(63),
            n_episodes_per_scenario=0,
        )

    with pytest.raises(ValueError, match="same dt, inner_steps, and horizon"):
        collect_supervisor_corpus_rollouts(
            policy,
            corpus,
            key=jax.random.PRNGKey(63),
            n_episodes_per_scenario=1,
        )


def test_supervisor_scenario_corpus_rejects_malformed_records() -> None:
    with pytest.raises(ValueError, match="at least one record"):
        build_supervisor_scenario_corpus([])

    valid = {
        "phases": [0.0, 0.1],
        "omegas": [0.04, 0.03],
        "base_K": [[0.0, 0.03], [0.03, 0.0]],
        "good_mask": [1.0, 0.0],
        "bad_mask": [0.0, 1.0],
        "dt": 0.02,
        "inner_steps": 4,
        "horizon": 3,
    }

    bad_shape = dict(valid, base_K=[[0.0, 0.03]])
    with pytest.raises(ValueError, match="base_K"):
        build_supervisor_scenario_corpus([bad_shape])

    bad_mask = dict(valid, good_mask=[0.0, 0.0])
    with pytest.raises(ValueError, match="good_mask"):
        build_supervisor_scenario_corpus([bad_mask])

    bad_metadata = dict(valid, metadata={"bad": np.array([1.0])})
    with pytest.raises(ValueError, match="metadata"):
        build_supervisor_scenario_corpus([bad_metadata])
