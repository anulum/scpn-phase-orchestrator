# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for UDE-Kuramoto module

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")

from scpn_phase_orchestrator.nn.functional import order_parameter
from scpn_phase_orchestrator.nn.ude import (
    CouplingResidual,
    UDEKuramotoLayer,
    ude_kuramoto_forward,
    ude_kuramoto_step,
)

N = 6
DT = 0.01
N_STEPS = 20


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def setup(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
    omegas = jax.random.normal(k2, (N,))
    raw = jax.random.normal(k3, (N, N)) * 0.1
    K = (raw + raw.T) / 2.0
    residual = CouplingResidual(hidden=8, key=k4)
    return phases, omegas, K, residual


class TestCouplingResidual:
    def test_scalar_output(self, key):
        res = CouplingResidual(hidden=8, key=key)
        out = res(jnp.array(0.5))
        assert out.shape == ()

    def test_zero_input(self, key):
        res = CouplingResidual(hidden=8, key=key)
        out = res(jnp.array(0.0))
        assert jnp.isfinite(out)

    def test_vmap_over_array(self, key):
        res = CouplingResidual(hidden=8, key=key)
        inputs = jnp.linspace(-jnp.pi, jnp.pi, 10)
        outputs = jax.vmap(res)(inputs)
        assert outputs.shape == (10,)


class TestUDEKuramotoStep:
    def test_output_shape(self, setup):
        phases, omegas, K, residual = setup
        out = ude_kuramoto_step(phases, omegas, K, residual, DT)
        assert out.shape == (N,)

    def test_phases_in_range(self, setup):
        phases, omegas, K, residual = setup
        out = ude_kuramoto_step(phases, omegas, K, residual, DT)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)


class TestUDEKuramotoForward:
    def test_trajectory_shape(self, setup):
        phases, omegas, K, residual = setup
        final, traj = ude_kuramoto_forward(phases, omegas, K, residual, DT, N_STEPS)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last(self, setup):
        phases, omegas, K, residual = setup
        final, traj = ude_kuramoto_forward(phases, omegas, K, residual, DT, N_STEPS)
        assert jnp.allclose(final, traj[-1], atol=1e-6)


class TestUDEGradients:
    def test_grad_wrt_K(self, setup):
        phases, omegas, K, residual = setup

        def loss_fn(K_):
            final, _ = ude_kuramoto_forward(phases, omegas, K_, residual, DT, N_STEPS)
            return order_parameter(final)

        grad_K = jax.grad(loss_fn)(K)
        assert grad_K.shape == K.shape
        assert jnp.isfinite(grad_K).all()

    def test_grad_through_residual(self, setup):
        phases, omegas, K, residual = setup

        @eqx.filter_grad
        def grad_fn(res):
            final, _ = ude_kuramoto_forward(phases, omegas, K, res, DT, N_STEPS)
            return order_parameter(final)

        grads = grad_fn(residual)
        # Check that at least some parameters have non-zero gradients
        flat_grads = jax.tree.leaves(grads)
        has_nonzero = any(not jnp.all(g == 0.0) for g in flat_grads)
        assert has_nonzero


class TestUDEKuramotoLayer:
    def test_init(self, key):
        layer = UDEKuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        assert layer.K.shape == (N, N)
        assert layer.omegas.shape == (N,)
        assert layer.n == N

    def test_forward_shape(self, key):
        layer = UDEKuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        out = layer(phases)
        assert out.shape == (N,)

    def test_sync_score(self, key):
        layer = UDEKuramotoLayer(N, n_steps=N_STEPS, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        R = layer.sync_score(phases)
        assert R.shape == ()
        assert 0.0 <= float(R) <= 1.0

    def test_gradient_through_layer(self, key):
        layer = UDEKuramotoLayer(N, n_steps=10, dt=DT, hidden=8, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)

        @eqx.filter_grad
        def grad_fn(model):
            return model.sync_score(phases)

        grads = grad_fn(layer)
        assert grads.K.shape == (N, N)
        assert jnp.isfinite(grads.K).all()
        assert jnp.isfinite(grads.omegas).all()

    def test_deterministic(self):
        l1 = UDEKuramotoLayer(N, key=jax.random.PRNGKey(99))
        l2 = UDEKuramotoLayer(N, key=jax.random.PRNGKey(99))
        assert jnp.allclose(l1.K, l2.K)


class TestUDEPipelineWiring:
    """Verify UDE Kuramoto wires into analysis pipeline."""

    def test_ude_to_order_parameter(self, key):
        from scpn_phase_orchestrator.nn.functional import order_parameter

        layer = UDEKuramotoLayer(N, n_steps=50, dt=DT, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        final = layer(phases)
        r = float(order_parameter(final))
        assert 0.0 <= r <= 1.0, f"UDE output must give valid R, got {r}"

    def test_residual_bounded(self, key):
        """CouplingResidual output should be bounded (no explosion)."""
        residual = CouplingResidual(hidden=8, key=key)
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        # CouplingResidual maps phase differences → corrections
        # Feed pairwise differences
        diff = phases[:, None] - phases[None, :]
        correction = jax.vmap(jax.vmap(residual))(diff)
        assert jnp.all(jnp.isfinite(correction))
        assert jnp.all(jnp.abs(correction) < 10.0)
