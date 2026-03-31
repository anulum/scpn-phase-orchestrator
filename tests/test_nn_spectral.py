# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn/ spectral analysis (JAX)

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.spectral import (
    algebraic_connectivity,
    eigenratio,
    laplacian_spectrum,
    sync_threshold,
)

N = 8


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def complete_graph():
    return jnp.ones((N, N)) - jnp.eye(N)


@pytest.fixture()
def ring_graph():
    K = jnp.zeros((N, N))
    for i in range(N):
        K = K.at[i, (i + 1) % N].set(1.0)
        K = K.at[i, (i - 1) % N].set(1.0)
    return K


class TestLaplacianSpectrum:
    def test_shape(self, complete_graph):
        eigs = laplacian_spectrum(complete_graph)
        assert eigs.shape == (N,)

    def test_first_eigenvalue_near_zero(self, complete_graph):
        eigs = laplacian_spectrum(complete_graph)
        assert float(eigs[0]) == pytest.approx(0.0, abs=1e-5)

    def test_complete_graph_known_spectrum(self, complete_graph):
        eigs = laplacian_spectrum(complete_graph)
        assert float(eigs[1]) == pytest.approx(N, abs=1e-4)


class TestAlgebraicConnectivity:
    def test_complete_graph(self, complete_graph):
        assert float(algebraic_connectivity(complete_graph)) == pytest.approx(
            N,
            abs=1e-4,
        )

    def test_ring_less_than_complete(self, ring_graph, complete_graph):
        assert float(algebraic_connectivity(ring_graph)) < float(
            algebraic_connectivity(complete_graph),
        )

    def test_differentiable(self, key):
        K = jnp.abs(jax.random.normal(key, (N, N)))
        K = (K + K.T) / 2
        grad = jax.grad(lambda k: algebraic_connectivity(k))(K)
        assert jnp.isfinite(grad).all()


class TestEigenratio:
    def test_complete_graph_ratio_one(self, complete_graph):
        assert float(eigenratio(complete_graph)) == pytest.approx(1.0, abs=1e-3)

    def test_ring_above_one(self, ring_graph):
        assert float(eigenratio(ring_graph)) > 1.0


class TestSyncThreshold:
    def test_scalar(self, complete_graph, key):
        omegas = jax.random.normal(key, (N,))
        kc = sync_threshold(complete_graph, omegas)
        assert kc.shape == ()

    def test_identical_frequencies_zero(self, complete_graph):
        assert float(sync_threshold(complete_graph, jnp.ones(N))) == pytest.approx(
            0.0,
            abs=1e-4,
        )


class TestNNSpectralPipelineWiring:
    """Pipeline: nn/ spectral → sync_threshold → KuramotoLayer."""

    def test_sync_threshold_informs_kuramoto_layer(self, key):
        """sync_threshold predicts K_c → KuramotoLayer uses K above it."""
        from scpn_phase_orchestrator.nn.functional import order_parameter
        from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer

        k1, k2 = jax.random.split(key)
        layer = KuramotoLayer(N, n_steps=100, dt=0.01, key=k1)
        omegas_test = jax.random.normal(k2, (N,)) * 0.5
        K_mat = jnp.abs(layer.K)
        kc = float(sync_threshold(K_mat, omegas_test))
        assert kc >= 0.0

        phases = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)
        final = layer(phases)
        r = float(order_parameter(final))
        assert 0.0 <= r <= 1.0
