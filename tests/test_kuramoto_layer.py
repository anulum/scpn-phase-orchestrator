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
    coupling_laplacian,
    kuramoto_forward,
    kuramoto_rk4_step,
    kuramoto_step,
    order_parameter,
    plv,
    saf_loss,
    saf_order_parameter,
    simplicial_forward,
    simplicial_rk4_step,
    simplicial_step,
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


# --- Simplicial (3-body) Kuramoto ---


SIGMA2 = 0.5


class TestSimplicialStep:
    def test_output_shape(self, setup):
        phases, omegas, K = setup
        out = simplicial_step(phases, omegas, K, DT, SIGMA2)
        assert out.shape == (N,)

    def test_phases_in_range(self, setup):
        phases, omegas, K = setup
        out = simplicial_step(phases, omegas, K, DT, SIGMA2)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)

    def test_rk4_output_shape(self, setup):
        phases, omegas, K = setup
        out = simplicial_rk4_step(phases, omegas, K, DT, SIGMA2)
        assert out.shape == (N,)

    def test_sigma2_zero_reduces_to_kuramoto(self, setup):
        phases, omegas, K = setup
        std = kuramoto_step(phases, omegas, K, DT)
        simp = simplicial_step(phases, omegas, K, DT, sigma2=0.0)
        assert jnp.allclose(std, simp, atol=1e-7)

    def test_sigma2_zero_rk4_reduces_to_kuramoto(self, setup):
        phases, omegas, K = setup
        std = kuramoto_rk4_step(phases, omegas, K, DT)
        simp = simplicial_rk4_step(phases, omegas, K, DT, sigma2=0.0)
        assert jnp.allclose(std, simp, atol=1e-7)

    def test_nonzero_sigma2_differs_from_standard(self, setup):
        phases, omegas, K = setup
        std = kuramoto_step(phases, omegas, K, DT)
        simp = simplicial_step(phases, omegas, K, DT, sigma2=1.0)
        assert not jnp.allclose(std, simp)


class TestSimplicialForward:
    def test_trajectory_shape(self, setup):
        phases, omegas, K = setup
        final, traj = simplicial_forward(phases, omegas, K, DT, N_STEPS, SIGMA2)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last_trajectory(self, setup):
        phases, omegas, K = setup
        final, traj = simplicial_forward(phases, omegas, K, DT, N_STEPS, SIGMA2)
        assert jnp.allclose(final, traj[-1], atol=1e-7)

    def test_euler_method(self, setup):
        phases, omegas, K = setup
        final, traj = simplicial_forward(
            phases, omegas, K, DT, 10, SIGMA2, method="euler"
        )
        assert traj.shape == (10, N)


class TestSimplicialGradients:
    def test_grad_through_step(self, setup):
        phases, omegas, K = setup

        def loss_fn(K_):
            out = simplicial_step(phases, omegas, K_, DT, SIGMA2)
            return jnp.sum(out)

        grad_K = jax.grad(loss_fn)(K)
        assert grad_K.shape == K.shape
        assert not jnp.all(grad_K == 0.0)

    def test_grad_wrt_sigma2(self, setup):
        phases, omegas, K = setup

        def loss_fn(s2):
            final, _ = simplicial_forward(phases, omegas, K, DT, 20, s2)
            return order_parameter(final)

        grad_s2 = jax.grad(loss_fn)(1.0)
        assert jnp.isfinite(grad_s2)

    def test_grad_through_forward(self, setup):
        phases, omegas, K = setup

        def loss_fn(K_):
            final, _ = simplicial_forward(phases, omegas, K_, DT, N_STEPS, SIGMA2)
            return order_parameter(final)

        grad_K = jax.grad(loss_fn)(K)
        assert grad_K.shape == K.shape
        assert jnp.isfinite(grad_K).all()

    def test_vmap_batched(self, setup):
        _, omegas, K = setup
        batch = jax.random.uniform(
            jax.random.PRNGKey(0), (4, N), minval=0.0, maxval=2.0 * jnp.pi
        )
        batched = jax.vmap(simplicial_step, in_axes=(0, None, None, None, None))
        out = batched(batch, omegas, K, DT, SIGMA2)
        assert out.shape == (4, N)


# --- Spectral Alignment Function (SAF) ---


class TestCouplingLaplacian:
    def test_shape(self, setup):
        _, _, K = setup
        L = coupling_laplacian(jnp.abs(K))
        assert L.shape == (N, N)

    def test_row_sums_zero(self, setup):
        _, _, K = setup
        L = coupling_laplacian(jnp.abs(K))
        row_sums = jnp.sum(L, axis=1)
        assert jnp.allclose(row_sums, 0.0, atol=1e-6)

    def test_symmetric(self, setup):
        _, _, K = setup
        K_sym = jnp.abs((K + K.T) / 2.0)
        L = coupling_laplacian(K_sym)
        assert jnp.allclose(L, L.T, atol=1e-7)


class TestSAFOrderParameter:
    def test_scalar_output(self, setup):
        _, omegas, K = setup
        K_pos = jnp.abs(K)
        r = saf_order_parameter(K_pos, omegas)
        assert r.shape == ()

    def test_in_range(self, setup):
        _, omegas, K = setup
        K_pos = jnp.abs(K)
        r = saf_order_parameter(K_pos, omegas)
        assert 0.0 <= float(r) <= 1.0

    def test_strong_coupling_high_r(self, key):
        omegas = jax.random.normal(key, (N,)) * 0.1
        K = jnp.ones((N, N)) * 5.0
        K = K.at[jnp.diag_indices(N)].set(0.0)
        r = saf_order_parameter(K, omegas)
        assert float(r) > 0.8

    def test_identical_frequencies_perfect_sync(self, key):
        omegas = jnp.zeros(N)
        K = jnp.ones((N, N))
        K = K.at[jnp.diag_indices(N)].set(0.0)
        r = saf_order_parameter(K, omegas)
        assert jnp.isclose(r, 1.0, atol=0.01)

    def test_differentiable_wrt_K(self, setup):
        _, omegas, K = setup
        K_pos = jnp.abs(K)

        def loss(K_):
            return saf_order_parameter(K_, omegas)

        grad_K = jax.grad(loss)(K_pos)
        assert grad_K.shape == K.shape
        assert jnp.isfinite(grad_K).all()

    def test_differentiable_wrt_omegas(self, setup):
        _, omegas, K = setup
        K_pos = jnp.abs(K)

        def loss(o):
            return saf_order_parameter(K_pos, o)

        grad_o = jax.grad(loss)(omegas)
        assert grad_o.shape == omegas.shape
        assert jnp.isfinite(grad_o).all()


class TestSAFLoss:
    def test_scalar_output(self, setup):
        _, omegas, K = setup
        K_pos = jnp.abs(K)
        loss = saf_loss(K_pos, omegas)
        assert loss.shape == ()

    def test_budget_penalty(self, setup):
        _, omegas, K = setup
        K_pos = jnp.abs(K)
        loss_no_budget = saf_loss(K_pos, omegas, budget=0.0)
        loss_tight = saf_loss(K_pos, omegas, budget=0.01, budget_weight=10.0)
        assert float(loss_tight) > float(loss_no_budget)

    def test_gradient_descent_improves_r(self, key):
        """Optimizing K via SAF gradient should increase order parameter."""
        k1, k2 = jax.random.split(key)
        # Small frequency spread so SAF r starts positive
        omegas = jax.random.normal(k1, (N,)) * 0.1
        # Moderate coupling — gives r in (0, 1) range, not clipped
        raw = jax.random.uniform(k2, (N, N), minval=0.5, maxval=1.5)
        K = (raw + raw.T) / 2.0
        K = K.at[jnp.diag_indices(N)].set(0.0)

        r_before = saf_order_parameter(K, omegas)
        assert float(r_before) > 0.0, "Need r_before > 0 for meaningful test"

        # Unclipped SAF for smooth gradients
        def raw_saf(K_):
            N_ = K_.shape[0]
            L = coupling_laplacian(K_)
            evals, evecs = jnp.linalg.eigh(L)
            lam = evals[1:]
            V = evecs[:, 1:]
            proj = (V.T @ omegas) ** 2
            return 1.0 - jnp.sum(proj / (lam**2 + 1e-8)) / (2.0 * N_)

        grad_fn = jax.grad(raw_saf)
        for _ in range(10):
            g = grad_fn(K)
            K = K + 0.05 * g  # ascend to maximize r
            K = (K + K.T) / 2.0
            K = jnp.maximum(K, 0.01)
            K = K.at[jnp.diag_indices(N)].set(0.0)

        r_after = saf_order_parameter(K, omegas)
        assert jnp.isfinite(r_after)
        assert float(r_after) >= float(r_before)


# ---------------------------------------------------------------------------
# Pipeline wiring + performance
# ---------------------------------------------------------------------------


class TestKuramotoPipelineWiring:
    """Verify that nn/ Kuramoto modules wire into the full analysis pipeline
    and meet performance targets — proving they are not decorative."""

    def test_layer_to_order_parameter_pipeline(self, key):
        """KuramotoLayer → order_parameter → R∈[0,1].
        Proves the layer output is valid input for analysis."""
        k1, k2 = jax.random.split(key)
        layer = KuramotoLayer(8, n_steps=50, dt=0.01, key=k1)
        phases = jax.random.uniform(k2, (8,), maxval=2.0 * jnp.pi)
        final = layer(phases)
        r = float(order_parameter(final))
        assert 0.0 <= r <= 1.0, f"R={r} must be in [0,1]"

    def test_layer_to_plv_pipeline(self, key):
        """KuramotoLayer → PLV matrix: trajectory feeds PLV analysis."""
        k1, k2 = jax.random.split(key)
        layer = KuramotoLayer(8, n_steps=50, dt=0.01, key=k1)
        phases = jax.random.uniform(k2, (8,), maxval=2.0 * jnp.pi)
        _, traj = layer.forward_with_trajectory(phases)
        plv_mat = plv(traj)
        assert plv_mat.shape == (8, 8)
        assert jnp.all(plv_mat >= 0.0) and jnp.all(plv_mat <= 1.0 + 1e-6)

    def test_step_performance_n64(self, key):
        """kuramoto_step(N=64) must complete in under 1ms (regression guard)."""
        import time

        phases = jax.random.uniform(key, (64,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(64)
        K = 0.3 * jnp.ones((64, 64))
        K = K.at[jnp.diag_indices(64)].set(0.0)

        # Warm up JIT
        kuramoto_step(phases, omegas, K, 0.01)

        t0 = time.perf_counter()
        for _ in range(100):
            kuramoto_step(phases, omegas, K, 0.01)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.001, (
            f"kuramoto_step(64) took {elapsed * 1000:.2f}ms, limit 1ms"
        )

    def test_forward_trajectory_feeds_training(self, key):
        """forward_with_trajectory output wires into trajectory_loss."""
        from scpn_phase_orchestrator.nn.training import trajectory_loss

        k1, k2 = jax.random.split(key)
        layer = KuramotoLayer(8, n_steps=30, dt=0.01, key=k1)
        phases = jax.random.uniform(k2, (8,), maxval=2.0 * jnp.pi)
        _, traj = layer.forward_with_trajectory(phases)
        loss = trajectory_loss(layer, phases, traj)
        assert jnp.isfinite(loss)
        assert float(loss) < 1e-4, "Own trajectory should give near-zero loss"
