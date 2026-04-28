# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for masked (sparse) coupling

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.functional import (
    kuramoto_forward,
    kuramoto_forward_masked,
    kuramoto_rk4_step,
    kuramoto_rk4_step_masked,
    kuramoto_step,
    kuramoto_step_masked,
    order_parameter,
)

N = 8
DT = 0.01
N_STEPS = 50


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def setup(key):
    k1, k2, k3 = jax.random.split(key, 3)
    phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
    omegas = jax.random.normal(k2, (N,))
    raw = jax.random.normal(k3, (N, N))
    K = (raw + raw.T) / 2.0
    return phases, omegas, K


class TestAllOnesMaskMatchesDense:
    """Mask of all ones should produce identical output to dense."""

    def test_euler_step(self, setup):
        phases, omegas, K = setup
        mask = jnp.ones((N, N))
        dense = kuramoto_step(phases, omegas, K, DT)
        masked = kuramoto_step_masked(phases, omegas, K, mask, DT)
        assert jnp.allclose(dense, masked, atol=1e-6)

    def test_rk4_step(self, setup):
        phases, omegas, K = setup
        mask = jnp.ones((N, N))
        dense = kuramoto_rk4_step(phases, omegas, K, DT)
        masked = kuramoto_rk4_step_masked(phases, omegas, K, mask, DT)
        assert jnp.allclose(dense, masked, atol=1e-6)

    def test_forward(self, setup):
        phases, omegas, K = setup
        mask = jnp.ones((N, N))
        f_d, t_d = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        f_m, t_m = kuramoto_forward_masked(phases, omegas, K, mask, DT, N_STEPS)
        assert jnp.allclose(f_d, f_m, atol=1e-5)
        assert jnp.allclose(t_d, t_m, atol=1e-5)


class TestZeroMaskDecouples:
    """Zero mask should produce free rotation (no coupling)."""

    def test_zero_mask_step(self, setup):
        phases, omegas, K = setup
        mask = jnp.zeros((N, N))
        out = kuramoto_step_masked(phases, omegas, K, mask, DT)
        expected = (phases + DT * omegas) % (2.0 * jnp.pi)
        assert jnp.allclose(out, expected, atol=1e-6)


class TestSparseMask:
    """Random sparse mask should produce valid but different output."""

    def test_sparse_output_shape(self, setup, key):
        phases, omegas, K = setup
        mask = (jax.random.uniform(key, (N, N)) < 0.3).astype(jnp.float32)
        mask = (mask + mask.T).clip(max=1.0)  # Symmetric
        out = kuramoto_step_masked(phases, omegas, K, mask, DT)
        assert out.shape == (N,)
        assert jnp.isfinite(out).all()

    def test_sparse_forward_shape(self, setup, key):
        phases, omegas, K = setup
        mask = (jax.random.uniform(key, (N, N)) < 0.3).astype(jnp.float32)
        mask = (mask + mask.T).clip(max=1.0)
        final, traj = kuramoto_forward_masked(
            phases,
            omegas,
            K,
            mask,
            DT,
            N_STEPS,
        )
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_sparse_differs_from_dense(self, setup, key):
        phases, omegas, K = setup
        mask = (jax.random.uniform(key, (N, N)) < 0.3).astype(jnp.float32)
        mask = (mask + mask.T).clip(max=1.0)
        f_d, _ = kuramoto_forward(phases, omegas, K, DT, N_STEPS)
        f_m, _ = kuramoto_forward_masked(phases, omegas, K, mask, DT, N_STEPS)
        assert not jnp.allclose(f_d, f_m, atol=1e-3)


class TestGradients:
    def test_grad_through_masked_step(self, setup, key):
        phases, omegas, K = setup
        mask = (jax.random.uniform(key, (N, N)) < 0.5).astype(jnp.float32)

        def loss(k):
            final, _ = kuramoto_forward_masked(
                phases,
                omegas,
                k,
                mask,
                DT,
                N_STEPS,
            )
            return order_parameter(final)

        grad = jax.grad(loss)(K)
        assert jnp.isfinite(grad).all()
        # Gradient should be zero where mask is zero
        zero_mask = mask == 0
        assert jnp.allclose(grad[zero_mask], 0.0, atol=1e-6)


class TestJIT:
    def test_jit_masked_forward(self, setup):
        phases, omegas, K = setup
        mask = jnp.ones((N, N))

        @jax.jit
        def run(p):
            f, _ = kuramoto_forward_masked(p, omegas, K, mask, DT, N_STEPS)
            return f

        out1 = run(phases)
        out2 = run(phases)
        assert jnp.allclose(out1, out2, atol=1e-6)


class TestVmap:
    def test_vmap_over_phases(self, setup, key):
        _, omegas, K = setup
        mask = jnp.ones((N, N))
        keys = jax.random.split(key, 4)
        batch = jax.vmap(
            lambda k: jax.random.uniform(k, (N,), maxval=2.0 * jnp.pi),
        )(keys)

        def run(p):
            f, _ = kuramoto_forward_masked(p, omegas, K, mask, DT, N_STEPS)
            return f

        results = jax.vmap(run)(batch)
        assert results.shape == (4, N)


class TestSparseCouplingPipelineWiring:
    """Pipeline: masked forward → order_parameter."""

    def test_masked_forward_to_order_parameter(self, setup, key):
        """kuramoto_forward_masked → order_parameter R∈[0,1]."""
        phases, omegas, K = setup
        mask = (jax.random.uniform(key, (N, N)) < 0.5).astype(jnp.float32)
        mask = (mask + mask.T).clip(max=1.0)
        final, _ = kuramoto_forward_masked(
            phases,
            omegas,
            K,
            mask,
            DT,
            100,
        )
        r = float(order_parameter(final))
        assert 0.0 <= r <= 1.0
