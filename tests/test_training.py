# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn/ training utilities

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")
optax = pytest.importorskip("optax", reason="optax required")

from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer
from scpn_phase_orchestrator.nn.training import (
    generate_chimera_data,
    generate_kuramoto_data,
    sync_loss,
    train,
    train_step,
    trajectory_loss,
)

N = 8
DT = 0.01
N_STEPS = 30


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def layer(key):
    return KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)


@pytest.fixture()
def phases(key):
    return jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)


class TestSyncLoss:
    def test_scalar_output(self, layer, phases):
        loss = sync_loss(layer, phases, target_R=1.0)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_zero_when_target_matches(self, layer, phases):
        from scpn_phase_orchestrator.nn.functional import order_parameter

        final = layer(phases)
        R = float(order_parameter(final))
        loss = sync_loss(layer, phases, target_R=R)
        assert float(loss) < 1e-6


class TestTrajectoryLoss:
    def test_scalar_output(self, layer, phases):
        _, traj = layer.forward_with_trajectory(phases)
        loss = trajectory_loss(layer, phases, traj)
        assert loss.shape == ()

    def test_zero_for_own_trajectory(self, layer, phases):
        _, traj = layer.forward_with_trajectory(phases)
        loss = trajectory_loss(layer, phases, traj)
        assert float(loss) < 1e-4


class TestTrainStep:
    def test_returns_updated_model(self, layer, phases):
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(layer, eqx.is_array))

        def loss_fn(m):
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
    def test_loss_decreases(self, layer, phases):
        def loss_fn(m):
            return sync_loss(m, phases, target_R=1.0)

        _, losses = train(
            layer,
            loss_fn,
            optax.adam(1e-3),
            n_epochs=10,
        )
        assert len(losses) == 10
        assert losses[-1] <= losses[0] + 0.01  # Allow small fluctuation


class TestGenerateKuramotoData:
    def test_shapes(self, key):
        K, omegas, p0, traj = generate_kuramoto_data(N, 100, key=key)
        assert K.shape == (N, N)
        assert omegas.shape == (N,)
        assert p0.shape == (N,)
        assert traj.shape == (100, N)

    def test_K_symmetric(self, key):
        K, _, _, _ = generate_kuramoto_data(N, 50, key=key)
        assert jnp.allclose(K, K.T, atol=1e-6)

    def test_K_zero_diagonal(self, key):
        K, _, _, _ = generate_kuramoto_data(N, 50, key=key)
        assert jnp.allclose(jnp.diag(K), 0.0, atol=1e-7)


class TestGenerateChimeraData:
    def test_shapes(self, key):
        K, p0, traj = generate_chimera_data(32, 200, key=key)
        assert K.shape == (32, 32)
        assert p0.shape == (32,)
        assert traj.shape == (200, 32)

    def test_K_sparse(self, key):
        K, _, _ = generate_chimera_data(32, 50, coupling_range=4, key=key)
        # Ring coupling: each node has 2*coupling_range neighbours
        nnz = jnp.sum(K > 0)
        expected_nnz = 32 * 2 * 4  # 256
        assert int(nnz) == expected_nnz
