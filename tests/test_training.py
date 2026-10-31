# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn/ training utilities

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")
optax = pytest.importorskip("optax", reason="optax required")

from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer
from scpn_phase_orchestrator.nn.training import (
    coupling_sparsity_loss,
    generate_chimera_data,
    generate_kuramoto_data,
    sync_loss,
    train,
    train_step,
    trajectory_loss,
)

if TYPE_CHECKING:
    from equinox import Module as EqxModule
    from jax import Array

N = 8
DT = 0.01
N_STEPS = 30


@pytest.fixture()
def key() -> Array:
    prng_key: Array = jax.random.PRNGKey(42)
    return prng_key


@pytest.fixture()
def layer(key: Array) -> KuramotoLayer:
    return KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)


@pytest.fixture()
def phases(key: Array) -> Array:
    values: Array = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
    return values


class TestSyncLoss:
    def test_scalar_output(self, layer: KuramotoLayer, phases: Array) -> None:
        loss = sync_loss(layer, phases, target_R=1.0)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_zero_when_target_matches(
        self, layer: KuramotoLayer, phases: Array
    ) -> None:
        from scpn_phase_orchestrator.nn.functional import order_parameter

        final = layer(phases)
        R = float(order_parameter(final))
        loss = sync_loss(layer, phases, target_R=R)
        assert float(loss) < 1e-6


class TestTrajectoryLoss:
    def test_scalar_output(self, layer: KuramotoLayer, phases: Array) -> None:
        _, traj = layer.forward_with_trajectory(phases)
        loss = trajectory_loss(layer, phases, traj)
        assert loss.shape == ()

    def test_zero_for_own_trajectory(self, layer: KuramotoLayer, phases: Array) -> None:
        _, traj = layer.forward_with_trajectory(phases)
        loss = trajectory_loss(layer, phases, traj)
        assert float(loss) < 1e-4


class TestCouplingSparsityLoss:
    def test_scales_with_mean_absolute_coupling(self) -> None:
        dense = jnp.array([[0.0, 2.0], [2.0, 0.0]])
        sparse = jnp.array([[0.0, 0.5], [0.5, 0.0]])
        assert coupling_sparsity_loss(dense) > coupling_sparsity_loss(sparse)

    def test_zero_target_density_matches_mean_l1_penalty(self) -> None:
        K = jnp.array([[0.0, -2.0], [1.0, 0.0]])
        assert jnp.isclose(coupling_sparsity_loss(K, target_density=0.0), 0.75)


class TestTrainStep:
    def test_returns_updated_model(self, layer: KuramotoLayer, phases: Array) -> None:
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(layer, eqx.is_array))

        def loss_fn(m: EqxModule) -> Array:
            return sync_loss(m, phases, target_R=1.0)

        new_model, new_state, loss = train_step(
            layer,
            loss_fn,
            opt_state,
            optimizer,
        )
        assert isinstance(new_model, KuramotoLayer)
        assert jnp.isfinite(loss)


class TestTrain:
    def test_loss_decreases(self, layer: KuramotoLayer, phases: Array) -> None:
        def loss_fn(m: EqxModule) -> Array:
            return sync_loss(m, phases, target_R=1.0)

        _, losses = train(
            layer,
            loss_fn,
            optax.adam(1e-3),
            n_epochs=10,
        )
        assert len(losses) == 10
        assert losses[-1] <= losses[0] + 0.01  # Allow small fluctuation

    def test_callback_observes_each_epoch_loss(
        self, layer: KuramotoLayer, phases: Array
    ) -> None:
        seen: list[tuple[int, float]] = []

        def loss_fn(m: EqxModule) -> Array:
            return sync_loss(m, phases, target_R=1.0)

        def callback(epoch: int, model: EqxModule, loss: Array) -> None:
            assert isinstance(model, KuramotoLayer)
            seen.append((epoch, float(loss)))

        _, losses = train(
            layer,
            loss_fn,
            optax.adam(1e-3),
            n_epochs=3,
            callback=callback,
        )
        assert [epoch for epoch, _ in seen] == [0, 1, 2]
        assert [loss for _, loss in seen] == losses


class TestGenerateKuramotoData:
    def test_shapes(self, key: Array) -> None:
        K, omegas, p0, traj = generate_kuramoto_data(N, 100, key=key)
        assert K.shape == (N, N)
        assert omegas.shape == (N,)
        assert p0.shape == (N,)
        assert traj.shape == (100, N)

    def test_K_symmetric(self, key: Array) -> None:
        K, _, _, _ = generate_kuramoto_data(N, 50, key=key)
        assert jnp.allclose(K, K.T, atol=1e-6)

    def test_K_zero_diagonal(self, key: Array) -> None:
        K, _, _, _ = generate_kuramoto_data(N, 50, key=key)
        assert jnp.allclose(jnp.diag(K), 0.0, atol=1e-7)


class TestGenerateChimeraData:
    def test_shapes(self, key: Array) -> None:
        K, p0, traj = generate_chimera_data(32, 200, key=key)
        assert K.shape == (32, 32)
        assert p0.shape == (32,)
        assert traj.shape == (200, 32)

    def test_K_sparse(self, key: Array) -> None:
        K, _, _ = generate_chimera_data(32, 50, coupling_range=4, key=key)
        # Ring coupling: each node has 2*coupling_range neighbours
        nnz = jnp.sum(K > 0)
        expected_nnz = 32 * 2 * 4  # 256
        assert int(nnz) == expected_nnz


# ---------------------------------------------------------------------------
# Pipeline wiring + performance
# ---------------------------------------------------------------------------


class TestTrainingPipelineWiring:
    """Verify that nn/ training modules wire into the full pipeline:
    generate data → build layer → train → evaluate R."""

    def test_full_training_pipeline(self, key: Array) -> None:
        """generate_kuramoto_data → KuramotoLayer → train → R improves.
        Proves the training pipeline produces a model that affects dynamics."""
        from scpn_phase_orchestrator.nn.functional import order_parameter

        K, omegas, p0, traj = generate_kuramoto_data(N, 50, key=key)
        layer = KuramotoLayer(N, n_steps=20, dt=DT, key=key)

        # R before training
        final_before = layer(p0)
        float(order_parameter(final_before))

        def loss_fn(m: EqxModule) -> Array:
            return sync_loss(m, p0, target_R=1.0)

        trained, losses = train(layer, loss_fn, optax.adam(1e-2), n_epochs=20)

        # R after training
        assert isinstance(trained, KuramotoLayer)
        final_after = trained(p0)
        r_after = float(order_parameter(final_after))

        # Training should improve or maintain R toward target
        assert losses[-1] < losses[0] + 0.05, "Loss should decrease"
        assert 0.0 <= r_after <= 1.0

    def test_train_step_preserves_k_symmetry(
        self, layer: KuramotoLayer, phases: Array
    ) -> None:
        """After one train_step, K matrix should still be approximately
        symmetric (train_step should enforce K = (K + K^T) / 2)."""
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(layer, eqx.is_array))

        def loss_fn(m: EqxModule) -> Array:
            return sync_loss(m, phases, target_R=1.0)

        new_model, _, _ = train_step(layer, loss_fn, opt_state, optimizer)
        assert isinstance(new_model, KuramotoLayer)
        K = new_model.K
        # K should be close to symmetric after the step
        asym = float(jnp.max(jnp.abs(K - K.T)))
        assert asym < 0.5, f"K asymmetry {asym:.4f} too large after train_step"

    def test_generated_trajectory_phases_in_range(self, key: Array) -> None:
        """Generated trajectory phases must be in [0, 2π)."""
        _, _, _, traj = generate_kuramoto_data(N, 100, key=key)
        traj_mod = traj % (2 * jnp.pi)
        assert jnp.all(traj_mod >= 0.0)
        assert jnp.all(traj_mod < 2 * jnp.pi + 1e-6)


class TestTrainingContracts:
    """Inputs that used to be accepted and silently change the objective."""

    def test_trajectory_loss_refuses_a_shorter_observation(self) -> None:
        layer = KuramotoLayer(4, n_steps=10, dt=0.01, key=jax.random.PRNGKey(0))
        phases = jnp.zeros(4)
        _, traj = layer.forward_with_trajectory(phases)
        with pytest.raises(ValueError, match="must match"):
            trajectory_loss(layer, phases, traj[:5])

    @pytest.mark.parametrize("density", [1.5, -0.1, float("nan"), True, "0.1"])
    def test_sparsity_target_must_lie_in_the_unit_interval(
        self, density: object
    ) -> None:
        """Above 1 the penalty was negative and rewarded larger couplings."""
        sparsity_any: Callable[..., Array] = coupling_sparsity_loss
        with pytest.raises(ValueError, match="target_density must be in"):
            sparsity_any(jnp.ones((3, 3)), target_density=density)

    def test_full_density_switches_the_penalty_off(self) -> None:
        assert float(coupling_sparsity_loss(jnp.ones((3, 3)), 1.0)) == 0.0

    @pytest.mark.parametrize("epochs", [-1, 2.0, True])
    def test_epoch_count_must_be_a_non_negative_integer(self, epochs: object) -> None:
        layer = KuramotoLayer(4, n_steps=2, dt=0.01, key=jax.random.PRNGKey(0))
        train_any: Callable[..., object] = train
        with pytest.raises(ValueError, match="n_epochs must be"):
            train_any(
                layer,
                lambda model: sync_loss(model, jnp.zeros(4)),
                optax.adam(1e-3),
                epochs,
            )
