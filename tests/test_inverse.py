# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for inverse Kuramoto pipeline

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.functional import kuramoto_forward
from scpn_phase_orchestrator.nn.inverse import (
    analytical_inverse,
    coupling_correlation,
    hybrid_inverse,
    infer_coupling,
    inverse_loss,
)

N = 4
DT = 0.02
T_STEPS = 50


@pytest.fixture()
def synthetic_data():
    """Generate synthetic Kuramoto data with known coupling."""
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)

    omegas_true = jax.random.normal(k1, (N,)) * 0.5
    raw = jax.random.normal(k2, (N, N)) * 0.3
    K_true = (raw + raw.T) / 2.0
    K_true = K_true.at[jnp.diag_indices(N)].set(0.0)
    phases0 = jax.random.uniform(k3, (N,), maxval=2.0 * jnp.pi)

    _, trajectory = kuramoto_forward(phases0, omegas_true, K_true, DT, T_STEPS)
    # Prepend initial conditions
    observed = jnp.concatenate([phases0[jnp.newaxis, :], trajectory], axis=0)
    return K_true, omegas_true, observed


class TestInverseLoss:
    def test_scalar_output(self, synthetic_data):
        K_true, omegas_true, observed = synthetic_data
        loss = inverse_loss(K_true, omegas_true, observed, DT)
        assert loss.shape == ()

    def test_true_params_low_loss(self, synthetic_data):
        K_true, omegas_true, observed = synthetic_data
        loss = inverse_loss(K_true, omegas_true, observed, DT)
        assert float(loss) < 0.01

    def test_random_params_higher_loss(self, synthetic_data):
        K_true, omegas_true, observed = synthetic_data
        key = jax.random.PRNGKey(99)
        K_rand = jax.random.normal(key, K_true.shape)
        loss_true = inverse_loss(K_true, omegas_true, observed, DT)
        loss_rand = inverse_loss(K_rand, omegas_true, observed, DT)
        assert float(loss_rand) > float(loss_true)

    def test_l1_penalty_increases_loss(self, synthetic_data):
        K_true, omegas_true, observed = synthetic_data
        loss_no_l1 = inverse_loss(K_true, omegas_true, observed, DT, l1_weight=0.0)
        loss_l1 = inverse_loss(K_true, omegas_true, observed, DT, l1_weight=1.0)
        assert float(loss_l1) >= float(loss_no_l1)

    def test_differentiable(self, synthetic_data):
        K_true, omegas_true, observed = synthetic_data

        def loss_fn(K):
            return inverse_loss(K, omegas_true, observed, DT)

        grad_K = jax.grad(loss_fn)(K_true)
        assert grad_K.shape == K_true.shape
        assert jnp.isfinite(grad_K).all()


class TestInferCoupling:
    def test_returns_correct_shapes(self, synthetic_data):
        _, _, observed = synthetic_data
        K, omegas, losses = infer_coupling(observed, DT, n_epochs=10, lr=0.01)
        assert K.shape == (N, N)
        assert omegas.shape == (N,)
        assert len(losses) == 10

    def test_loss_decreases(self, synthetic_data):
        _, _, observed = synthetic_data
        _, _, losses = infer_coupling(observed, DT, n_epochs=50, lr=0.01)
        assert losses[-1] < losses[0]

    def test_K_symmetric(self, synthetic_data):
        _, _, observed = synthetic_data
        K, _, _ = infer_coupling(observed, DT, n_epochs=20)
        assert jnp.allclose(K, K.T, atol=1e-7)

    def test_K_zero_diagonal(self, synthetic_data):
        _, _, observed = synthetic_data
        K, _, _ = infer_coupling(observed, DT, n_epochs=20)
        assert jnp.allclose(jnp.diag(K), 0.0, atol=1e-7)

    def test_inferred_K_finite_and_bounded(self, synthetic_data):
        _, _, observed = synthetic_data
        K, omegas, _ = infer_coupling(observed, DT, n_epochs=50, lr=0.01)
        assert jnp.isfinite(K).all()
        assert jnp.isfinite(omegas).all()
        assert jnp.max(jnp.abs(K)) < 100.0  # not diverged


class TestCouplingCorrelation:
    def test_perfect_correlation(self):
        K = jnp.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.3], [0.5, 0.3, 0.0]])
        corr = coupling_correlation(K, K)
        assert jnp.isclose(corr, 1.0, atol=1e-5)

    def test_scaled_correlation(self):
        K = jnp.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.3], [0.5, 0.3, 0.0]])
        corr = coupling_correlation(K, 2.0 * K)
        assert jnp.isclose(corr, 1.0, atol=1e-5)

    def test_negative_correlation(self):
        K = jnp.array([[0.0, 1.0, 0.5], [1.0, 0.0, 0.3], [0.5, 0.3, 0.0]])
        corr = coupling_correlation(K, -K)
        assert jnp.isclose(corr, -1.0, atol=1e-5)


# Noiseless data with zero omegas for analytical inverse tests
@pytest.fixture()
def noiseless_data():
    """Noiseless Kuramoto with zero omegas — ideal for analytical recovery."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key, 2)
    n = 8
    raw = jax.random.normal(k1, (n, n)) * 0.3
    K_true = (raw + raw.T) / 2.0
    K_true = K_true.at[jnp.diag_indices(n)].set(0.0)
    omegas_true = jnp.zeros(n)
    phases0 = jax.random.uniform(k2, (n,), maxval=2.0 * jnp.pi)
    _, trajectory = kuramoto_forward(phases0, omegas_true, K_true, DT, 200)
    observed = jnp.concatenate([phases0[jnp.newaxis, :], trajectory], axis=0)
    return K_true, omegas_true, observed


class TestAnalyticalInverse:
    def test_returns_correct_shapes(self, noiseless_data):
        K_true, _, observed = noiseless_data
        K, omegas = analytical_inverse(observed, DT)
        assert K.shape == K_true.shape
        assert omegas.shape == (K_true.shape[0],)

    def test_K_symmetric(self, noiseless_data):
        _, _, observed = noiseless_data
        K, _ = analytical_inverse(observed, DT)
        assert jnp.allclose(K, K.T, atol=1e-7)

    def test_K_zero_diagonal(self, noiseless_data):
        _, _, observed = noiseless_data
        K, _ = analytical_inverse(observed, DT)
        assert jnp.allclose(jnp.diag(K), 0.0, atol=1e-7)

    def test_noiseless_correlation_above_09(self, noiseless_data):
        K_true, _, observed = noiseless_data
        K, _ = analytical_inverse(observed, DT)
        corr = float(coupling_correlation(K_true, K))
        assert corr > 0.9, f"Expected >0.9, got {corr:.3f}"

    def test_recovers_omegas_near_zero(self, noiseless_data):
        _, _, observed = noiseless_data
        _, omegas = analytical_inverse(observed, DT)
        assert jnp.allclose(omegas, 0.0, atol=0.5)

    def test_tikhonov_regularisation(self, noiseless_data):
        K_true, _, observed = noiseless_data
        K_reg, _ = analytical_inverse(observed, DT, alpha=0.1)
        assert jnp.isfinite(K_reg).all()
        corr = float(coupling_correlation(K_true, K_reg))
        assert corr > 0.5

    def test_with_nonzero_omegas(self):
        key = jax.random.PRNGKey(7)
        k1, k2, k3 = jax.random.split(key, 3)
        n = 6
        K_true = jax.random.normal(k1, (n, n)) * 0.2
        K_true = (K_true + K_true.T) / 2
        K_true = K_true.at[jnp.diag_indices(n)].set(0.0)
        omegas_true = jax.random.normal(k2, (n,)) * 0.3
        p0 = jax.random.uniform(k3, (n,), maxval=2 * jnp.pi)
        _, traj = kuramoto_forward(p0, omegas_true, K_true, DT, 200)
        observed = jnp.concatenate([p0[jnp.newaxis, :], traj], axis=0)
        K, omegas = analytical_inverse(observed, DT)
        corr = float(coupling_correlation(K_true, K))
        # With nonzero omegas, omega estimation absorbs some K variance
        assert corr > 0.4, f"Expected >0.4 with omegas, got {corr:.3f}"


class TestHybridInverse:
    def test_returns_correct_shapes(self, noiseless_data):
        K_true, _, observed = noiseless_data
        K, omegas, losses = hybrid_inverse(observed, DT, n_refine=5)
        assert K.shape == K_true.shape
        assert omegas.shape == (K_true.shape[0],)
        assert len(losses) == 5

    def test_refine_zero_is_analytical(self, noiseless_data):
        _, _, observed = noiseless_data
        K_h, _, losses = hybrid_inverse(observed, DT, n_refine=0)
        K_a, _ = analytical_inverse(observed, DT)
        assert jnp.allclose(K_h, K_a, atol=1e-6)
        assert losses == []

    def test_correlation_at_least_analytical(self, noiseless_data):
        K_true, _, observed = noiseless_data
        K_a, _ = analytical_inverse(observed, DT)
        corr_a = float(coupling_correlation(K_true, K_a))
        K_h, _, _ = hybrid_inverse(observed, DT, n_refine=20)
        corr_h = float(coupling_correlation(K_true, K_h))
        # Hybrid should not degrade below analytical
        assert corr_h >= corr_a - 0.05, (
            f"Hybrid {corr_h:.3f} degraded vs analytical {corr_a:.3f}"
        )
