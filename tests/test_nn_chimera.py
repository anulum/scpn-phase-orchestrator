# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn/ chimera detection (JAX)

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.chimera import (
    chimera_index,
    detect_chimera,
    local_order_parameter,
)

N = 16


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def ring_K():
    K = jnp.zeros((N, N))
    for i in range(N):
        K = K.at[i, (i + 1) % N].set(1.0)
        K = K.at[i, (i - 1) % N].set(1.0)
    return K


class TestLocalOrderParameter:
    def test_shape(self, key, ring_K):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        R = local_order_parameter(phases, ring_K)
        assert R.shape == (N,)

    def test_range(self, key, ring_K):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        R = local_order_parameter(phases, ring_K)
        assert jnp.all(R >= 0.0)
        assert jnp.all(R <= 1.0 + 1e-6)

    def test_perfect_sync_all_ones(self, ring_K):
        phases = jnp.zeros(N)
        R = local_order_parameter(phases, ring_K)
        assert jnp.allclose(R, 1.0, atol=1e-5)

    def test_no_coupling_returns_ones(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        K_zero = jnp.zeros((N, N))
        R = local_order_parameter(phases, K_zero)
        # No neighbours → n_neighbours clipped to 1, R = |0/1| = 0
        assert jnp.all(R <= 1e-6)


class TestChimeraIndex:
    def test_scalar_output(self, key, ring_K):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        idx = chimera_index(phases, ring_K)
        assert idx.shape == ()

    def test_zero_for_uniform_sync(self, ring_K):
        phases = jnp.zeros(N)
        idx = chimera_index(phases, ring_K)
        assert float(idx) < 1e-6

    def test_differentiable(self, key, ring_K):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)

        def loss(p):
            return chimera_index(p, ring_K)

        grad = jax.grad(loss)(phases)
        assert jnp.isfinite(grad).all()


class TestDetectChimera:
    def test_returns_two_masks(self, key, ring_K):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        coh, incoh = detect_chimera(phases, ring_K)
        assert coh.shape == (N,)
        assert incoh.shape == (N,)

    def test_perfect_sync_all_coherent(self, ring_K):
        phases = jnp.zeros(N)
        coh, incoh = detect_chimera(phases, ring_K)
        assert jnp.all(coh)
        assert not jnp.any(incoh)


class TestNNChimeraPipelineWiring:
    """Pipeline: KuramotoLayer → phases → chimera_index."""

    def test_kuramoto_layer_to_chimera_index(self, key):
        """KuramotoLayer → phases → chimera_index∈[0,1]."""
        from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer

        k1, k2 = jax.random.split(key)
        layer = KuramotoLayer(N, n_steps=100, dt=0.01, key=k1)
        phases = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)
        final = layer(phases)
        K = jnp.abs(layer.K)
        idx = float(chimera_index(final, K))
        assert 0.0 <= idx <= 1.0
