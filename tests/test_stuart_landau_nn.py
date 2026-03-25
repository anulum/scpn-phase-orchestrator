# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for differentiable Stuart-Landau nn/ module

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")

from scpn_phase_orchestrator.nn.functional import (
    order_parameter,
    stuart_landau_forward,
    stuart_landau_rk4_step,
    stuart_landau_step,
)
from scpn_phase_orchestrator.nn.stuart_landau_layer import StuartLandauLayer

N = 8
DT = 0.01
N_STEPS = 50


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def setup(key):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    phases = jax.random.uniform(k1, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    amplitudes = jax.random.uniform(k2, (N,), minval=0.1, maxval=1.0)
    omegas = jax.random.normal(k3, (N,))
    mu = jnp.ones(N) * 0.5
    raw = jax.random.normal(k4, (N, N))
    K = (raw + raw.T) / 2.0 * 0.1
    raw_r = jax.random.normal(k5, (N, N))
    K_r = (raw_r + raw_r.T) / 2.0 * 0.1
    return phases, amplitudes, omegas, mu, K, K_r


# --- Functional API ---


class TestStuartLandauStep:
    def test_output_shapes(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        new_p, new_r = stuart_landau_step(phases, amps, omegas, mu, K, K_r, DT)
        assert new_p.shape == (N,)
        assert new_r.shape == (N,)

    def test_phases_in_range(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        new_p, _ = stuart_landau_step(phases, amps, omegas, mu, K, K_r, DT)
        assert jnp.all(new_p >= 0.0)
        assert jnp.all(new_p < 2.0 * jnp.pi)

    def test_amplitudes_nonnegative(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        _, new_r = stuart_landau_step(phases, amps, omegas, mu, K, K_r, DT)
        assert jnp.all(new_r >= 0.0)

    def test_rk4_output_shapes(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        new_p, new_r = stuart_landau_rk4_step(phases, amps, omegas, mu, K, K_r, DT)
        assert new_p.shape == (N,)
        assert new_r.shape == (N,)

    def test_rk4_amplitudes_nonnegative(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        _, new_r = stuart_landau_rk4_step(phases, amps, omegas, mu, K, K_r, DT)
        assert jnp.all(new_r >= 0.0)


class TestStuartLandauForward:
    def test_trajectory_shapes(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        fp, fr, tp, tr = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, DT, N_STEPS
        )
        assert fp.shape == (N,)
        assert fr.shape == (N,)
        assert tp.shape == (N_STEPS, N)
        assert tr.shape == (N_STEPS, N)

    def test_final_matches_last(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        fp, fr, tp, tr = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, DT, N_STEPS
        )
        assert jnp.allclose(fp, tp[-1], atol=1e-7)
        assert jnp.allclose(fr, tr[-1], atol=1e-7)

    def test_supercritical_convergence(self):
        """Uncoupled supercritical oscillator: r -> sqrt(mu)."""
        phases = jnp.array([1.0])
        amps = jnp.array([0.01])
        omegas = jnp.array([1.0])
        mu = jnp.array([1.0])
        K = jnp.zeros((1, 1))
        K_r = jnp.zeros((1, 1))
        _, fr, _, _ = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, 0.02, 500
        )
        assert jnp.isclose(fr, jnp.sqrt(mu[0]), atol=0.1)

    def test_subcritical_decay(self):
        """Uncoupled subcritical oscillator: r -> 0."""
        phases = jnp.array([1.0])
        amps = jnp.array([0.5])
        omegas = jnp.array([1.0])
        mu = jnp.array([-1.0])
        K = jnp.zeros((1, 1))
        K_r = jnp.zeros((1, 1))
        _, fr, _, _ = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, 0.01, 500
        )
        assert fr < 0.05

    def test_euler_method(self, setup):
        phases, amps, omegas, mu, K, K_r = setup
        fp, fr, tp, tr = stuart_landau_forward(
            phases, amps, omegas, mu, K, K_r, DT, 10, method="euler"
        )
        assert tp.shape == (10, N)


# --- Gradient flow ---


class TestStuartLandauGradients:
    def test_grad_wrt_K(self, setup):
        phases, amps, omegas, mu, K, K_r = setup

        def loss_fn(K_):
            fp, _, _, _ = stuart_landau_forward(
                phases, amps, omegas, mu, K_, K_r, DT, 20
            )
            return order_parameter(fp)

        grad_K = jax.grad(loss_fn)(K)
        assert grad_K.shape == K.shape
        assert jnp.isfinite(grad_K).all()

    def test_grad_wrt_K_r(self, setup):
        phases, amps, omegas, mu, K, K_r = setup

        def loss_fn(K_r_):
            _, fr, _, _ = stuart_landau_forward(
                phases, amps, omegas, mu, K, K_r_, DT, 20
            )
            return jnp.mean(fr)

        grad_Kr = jax.grad(loss_fn)(K_r)
        assert grad_Kr.shape == K_r.shape
        assert jnp.isfinite(grad_Kr).all()

    def test_grad_wrt_mu(self, setup):
        phases, amps, omegas, mu, K, K_r = setup

        def loss_fn(mu_):
            _, fr, _, _ = stuart_landau_forward(
                phases, amps, omegas, mu_, K, K_r, DT, 20
            )
            return jnp.mean(fr)

        grad_mu = jax.grad(loss_fn)(mu)
        assert grad_mu.shape == mu.shape
        assert jnp.isfinite(grad_mu).all()

    def test_vmap_batched(self, setup):
        _, amps, omegas, mu, K, K_r = setup
        batch_p = jax.random.uniform(
            jax.random.PRNGKey(0), (4, N), maxval=2.0 * jnp.pi
        )
        batch_r = jax.random.uniform(jax.random.PRNGKey(1), (4, N), maxval=1.0)

        def run_one(p, r):
            return stuart_landau_step(p, r, omegas, mu, K, K_r, DT)

        out_p, out_r = jax.vmap(run_one)(batch_p, batch_r)
        assert out_p.shape == (4, N)
        assert out_r.shape == (4, N)


# --- StuartLandauLayer (equinox) ---


class TestStuartLandauLayer:
    def test_init(self, key):
        layer = StuartLandauLayer(N, key=key)
        assert layer.K.shape == (N, N)
        assert layer.K_r.shape == (N, N)
        assert layer.omegas.shape == (N,)
        assert layer.mu.shape == (N,)
        assert layer.n == N

    def test_forward_shapes(self, key):
        layer = StuartLandauLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        amps = jax.random.uniform(key, (N,), maxval=1.0)
        out_p, out_r = layer(phases, amps)
        assert out_p.shape == (N,)
        assert out_r.shape == (N,)

    def test_trajectory(self, key):
        layer = StuartLandauLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        amps = jax.random.uniform(key, (N,), maxval=1.0)
        fp, fr, tp, tr = layer.forward_with_trajectory(phases, amps)
        assert tp.shape == (N_STEPS, N)
        assert tr.shape == (N_STEPS, N)

    def test_sync_score(self, key):
        layer = StuartLandauLayer(N, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        amps = jax.random.uniform(key, (N,), maxval=1.0)
        R = layer.sync_score(phases, amps)
        assert R.shape == ()
        assert 0.0 <= float(R) <= 1.0

    def test_mean_amplitude(self, key):
        layer = StuartLandauLayer(N, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        amps = jax.random.uniform(key, (N,), maxval=1.0)
        ma = layer.mean_amplitude(phases, amps)
        assert ma.shape == ()
        assert float(ma) >= 0.0

    def test_gradient_through_layer(self, key):
        layer = StuartLandauLayer(N, n_steps=20, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        amps = jax.random.uniform(key, (N,), maxval=1.0)

        @eqx.filter_grad
        def grad_fn(model):
            return model.sync_score(phases, amps)

        grads = grad_fn(layer)
        assert grads.K.shape == (N, N)
        assert jnp.isfinite(grads.K).all()
        assert jnp.isfinite(grads.K_r).all()
        assert jnp.isfinite(grads.omegas).all()
        assert jnp.isfinite(grads.mu).all()

    def test_deterministic_seeding(self):
        layer1 = StuartLandauLayer(N, key=jax.random.PRNGKey(99))
        layer2 = StuartLandauLayer(N, key=jax.random.PRNGKey(99))
        assert jnp.allclose(layer1.K, layer2.K)
        assert jnp.allclose(layer1.mu, layer2.mu)

    def test_vmap_batched(self, key):
        layer = StuartLandauLayer(N, n_steps=10, dt=DT, key=key)
        batch_p = jax.random.uniform(key, (4, N), maxval=2.0 * jnp.pi)
        batch_r = jax.random.uniform(key, (4, N), maxval=1.0)

        def run_one(p, r):
            return layer(p, r)

        out_p, out_r = jax.vmap(run_one)(batch_p, batch_r)
        assert out_p.shape == (4, N)
        assert out_r.shape == (4, N)
