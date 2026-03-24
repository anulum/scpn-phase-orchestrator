# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn/ differentiable Kuramoto module

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required for nn/ tests")
jnp = pytest.importorskip("jax.numpy", reason="JAX required for nn/ tests")
eqx = pytest.importorskip("equinox", reason="equinox required for KuramotoLayer tests")

from scpn_phase_orchestrator.nn.functional import (
    kuramoto_forward,
    kuramoto_rk4_step,
    kuramoto_step,
    order_parameter,
    plv,
)
from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer

N = 8
DT = 0.01
N_STEPS = 50


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def setup(key):
    k1, k2, k3 = jax.random.split(key, 3)
    phases = jax.random.uniform(k1, (N,), minval=0.0, maxval=2.0 * jnp.pi)
    omegas = jax.random.normal(k2, (N,))
    raw = jax.random.normal(k3, (N, N))
    K = (raw + raw.T) / 2.0
    return phases, omegas, K


# --- Functional API: shapes and values ---


class TestKuramotoStep:
    def test_output_shape(self, setup):
        phases, omegas, K = setup
        out = kuramoto_step(phases, omegas, K, DT)
        assert out.shape == (N,)

    def test_phases_in_range(self, setup):
        phases, omegas, K = setup
        out = kuramoto_step(phases, omegas, K, DT)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)

    def test_rk4_output_shape(self, setup):
        phases, omegas, K = setup
        out = kuramoto_rk4_step(phases, omegas, K, DT)
        assert out.shape == (N,)

    def test_zero_coupling_free_rotation(self, setup):
        phases, omegas, _ = setup
        K_zero = jnp.zeros((N, N))
        out = kuramoto_step(phases, omegas, K_zero, DT)
        expected = (phases + DT * omegas) % (2.0 * jnp.pi)
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_euler_rk4_close_for_small_dt(self, setup):
        phases, omegas, K = setup
        small_dt = 0.001
        e = kuramoto_step(phases, omegas, K, small_dt)
        r = kuramoto_rk4_step(phases, omegas, K, small_dt)
        assert jnp.allclose(e, r, atol=1e-4)


class TestKuramotoForward:
    def test_trajectory_shape(self, setup):
        phases, omegas, K = setup
        final, traj = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last_trajectory(self, setup):
        phases, omegas, K = setup
        final, traj = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        assert jnp.allclose(final, traj[-1], atol=1e-7)

    def test_euler_method(self, setup):
        phases, omegas, K = setup
        final, traj = kuramoto_forward(phases, omegas, K, DT, 10, method="euler")
        assert traj.shape == (10, N)


class TestOrderParameter:
    def test_perfect_sync(self):
        phases = jnp.ones(N) * 1.5
        R = order_parameter(phases)
        assert jnp.isclose(R, 1.0, atol=1e-6)

    def test_incoherent(self):
        phases = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
        R = order_parameter(phases)
        assert R < 0.15

    def test_trajectory_batch(self, setup):
        phases, omegas, K = setup
        _, traj = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        R_traj = order_parameter(traj)
        assert R_traj.shape == (N_STEPS,)


class TestPLV:
    def test_output_shape(self, setup):
        phases, omegas, K = setup
        _, traj = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        P = plv(traj)
        assert P.shape == (N, N)

    def test_diagonal_ones(self, setup):
        phases, omegas, K = setup
        _, traj = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        P = plv(traj)
        assert jnp.allclose(jnp.diag(P), 1.0, atol=1e-6)

    def test_values_in_range(self, setup):
        phases, omegas, K = setup
        _, traj = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        P = plv(traj)
        assert jnp.all(P >= -1e-6)
        assert jnp.all(P <= 1.0 + 1e-6)


# --- JIT compilation ---


class TestJIT:
    def test_step_jit(self, setup):
        phases, omegas, K = setup
        jitted = jax.jit(kuramoto_step, static_argnums=(3,))
        out = jitted(phases, omegas, K, DT)
        assert out.shape == (N,)

    def test_forward_jit(self, setup):
        phases, omegas, K = setup

        @jax.jit
        def run(p, o, k):
            return kuramoto_forward(p, o, k, DT, N_STEPS)

        final, traj = run(phases, omegas, K)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_order_parameter_jit(self, setup):
        phases, _, _ = setup
        jitted = jax.jit(order_parameter)
        R = jitted(phases)
        assert R.shape == ()


# --- Gradient flow ---


class TestGradients:
    def test_grad_through_step(self, setup):
        phases, omegas, K = setup

        def loss_fn(K_):
            out = kuramoto_step(phases, omegas, K_, DT)
            return jnp.sum(out)

        grad_K = jax.grad(loss_fn)(K)
        assert grad_K.shape == K.shape
        assert not jnp.all(grad_K == 0.0)

    def test_grad_through_forward(self, setup):
        phases, omegas, K = setup

        def loss_fn(K_):
            final, _ = kuramoto_forward(phases, omegas, K_, DT, N_STEPS)
            return order_parameter(final)

        grad_K = jax.grad(loss_fn)(K)
        assert grad_K.shape == K.shape
        assert jnp.isfinite(grad_K).all()

    def test_grad_wrt_phases(self, setup):
        phases, omegas, K = setup

        def loss_fn(p):
            final, _ = kuramoto_forward(p, omegas, K, DT, N_STEPS)
            return order_parameter(final)

        grad_p = jax.grad(loss_fn)(phases)
        assert grad_p.shape == phases.shape
        assert jnp.isfinite(grad_p).all()

    def test_grad_wrt_omegas(self, setup):
        phases, omegas, K = setup

        def loss_fn(o):
            final, _ = kuramoto_forward(phases, o, K, DT, N_STEPS)
            return order_parameter(final)

        grad_o = jax.grad(loss_fn)(omegas)
        assert grad_o.shape == omegas.shape
        assert jnp.isfinite(grad_o).all()

    def test_grad_finite_difference_agreement(self, setup):
        """Verify autodiff matches finite differences (the gold test)."""
        phases, omegas, K = setup

        def loss_fn(K_flat):
            K_ = K_flat.reshape(N, N)
            final, _ = kuramoto_forward(phases, omegas, K_, DT, 10)
            return order_parameter(final)

        K_flat = K.flatten()
        auto_grad = jax.grad(loss_fn)(K_flat)

        eps = 1e-4
        fd_grad = jnp.zeros_like(K_flat)
        for i in range(min(5, len(K_flat))):
            e = jnp.zeros_like(K_flat).at[i].set(eps)
            fd_grad = fd_grad.at[i].set(
                (loss_fn(K_flat + e) - loss_fn(K_flat - e)) / (2.0 * eps)
            )

        # Check first 5 elements agree
        assert jnp.allclose(auto_grad[:5], fd_grad[:5], atol=1e-2)


# --- vmap batching ---


class TestVmap:
    def test_vmap_over_phases(self, setup):
        _, omegas, K = setup
        batch = jax.random.uniform(
            jax.random.PRNGKey(0), (4, N), minval=0.0, maxval=2.0 * jnp.pi
        )
        batched_step = jax.vmap(kuramoto_step, in_axes=(0, None, None, None))
        out = batched_step(batch, omegas, K, DT)
        assert out.shape == (4, N)

    def test_vmap_forward(self, setup):
        _, omegas, K = setup
        batch = jax.random.uniform(
            jax.random.PRNGKey(0), (4, N), minval=0.0, maxval=2.0 * jnp.pi
        )

        def run_one(p):
            return kuramoto_forward(p, omegas, K, DT, N_STEPS)

        finals, trajs = jax.vmap(run_one)(batch)
        assert finals.shape == (4, N)
        assert trajs.shape == (4, N_STEPS, N)


# --- KuramotoLayer (equinox) ---


class TestKuramotoLayer:
    def test_init(self, key):
        layer = KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        assert layer.K.shape == (N, N)
        assert layer.omegas.shape == (N,)
        assert layer.n == N

    def test_K_symmetric(self, key):
        layer = KuramotoLayer(N, key=key)
        assert jnp.allclose(layer.K, layer.K.T, atol=1e-7)

    def test_forward_shape(self, key):
        layer = KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        out = layer(phases)
        assert out.shape == (N,)

    def test_trajectory(self, key):
        layer = KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        final, traj = layer.forward_with_trajectory(phases)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_sync_score_scalar(self, key):
        layer = KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        R = layer.sync_score(phases)
        assert R.shape == ()
        assert 0.0 <= float(R) <= 1.0

    def test_gradient_through_layer(self, key):
        layer = KuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)

        @eqx.filter_grad
        def grad_fn(model):
            return model.sync_score(phases)

        grads = grad_fn(layer)
        assert grads.K.shape == (N, N)
        assert jnp.isfinite(grads.K).all()
        assert jnp.isfinite(grads.omegas).all()

    def test_deterministic_seeding(self):
        layer1 = KuramotoLayer(N, key=jax.random.PRNGKey(99))
        layer2 = KuramotoLayer(N, key=jax.random.PRNGKey(99))
        assert jnp.allclose(layer1.K, layer2.K)
        assert jnp.allclose(layer1.omegas, layer2.omegas)

    def test_different_seeds_differ(self):
        layer1 = KuramotoLayer(N, key=jax.random.PRNGKey(0))
        layer2 = KuramotoLayer(N, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(layer1.K, layer2.K)

    def test_vmap_batched_forward(self, key):
        layer = KuramotoLayer(N, n_steps=20, dt=DT, key=key)
        batch = jax.random.uniform(key, (4, N), maxval=2.0 * jnp.pi)
        batched = jax.vmap(layer)
        out = batched(batch)
        assert out.shape == (4, N)
