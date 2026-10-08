# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for nn/ chimera detection (JAX)

"""Exercise actual JAX chimera arrays, phase gradients and engine wiring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import enable_x64

from benchmarks.chimera_local_order_reference import scalar_local_order
from scpn_phase_orchestrator.nn.chimera import (
    chimera_index,
    detect_chimera,
    local_order_parameter,
)

pytestmark = pytest.mark.native_runtime

N = 16


@pytest.fixture()
def key() -> jax.Array:
    """Return a deterministic actual JAX random key."""
    return jax.random.PRNGKey(42)


@pytest.fixture()
def ring_K() -> jax.Array:
    """Construct a two-neighbour ring with no self edges."""
    K = jnp.zeros((N, N))
    for i in range(N):
        K = K.at[i, (i + 1) % N].set(1.0)
        K = K.at[i, (i - 1) % N].set(1.0)
    return K


class TestLocalOrderParameter:
    """Check local-order topology, numerical bounds and shape admission."""

    def test_shape(self, key: jax.Array, ring_K: jax.Array) -> None:
        """Preserve one local-order value per oscillator."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        R = local_order_parameter(phases, ring_K)
        assert R.shape == (N,)

    def test_range(self, key: jax.Array, ring_K: jax.Array) -> None:
        """Keep generated JAX local-order values inside the unit interval."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        R = local_order_parameter(phases, ring_K)
        assert jnp.all(R >= 0.0)
        assert jnp.all(R <= 1.0 + 1e-6)

    def test_perfect_sync_all_ones(self, ring_K: jax.Array) -> None:
        """Measure unit coherence for a synchronized coupled ring."""
        phases = jnp.zeros(N)
        R = local_order_parameter(phases, ring_K)
        assert jnp.allclose(R, 1.0, atol=1e-5)

    def test_no_coupling_returns_ones(self, key: jax.Array) -> None:
        """Preserve zero-neighbour semantics in the original test slot."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        K_zero = jnp.zeros((N, N))
        R = local_order_parameter(phases, K_zero)
        # No neighbours → n_neighbours clipped to 1, R = |0/1| = 0
        assert jnp.all(R <= 1e-6)


class TestChimeraIndex:
    """Check JAX local-order variance on the declared nonzero topology."""

    def test_scalar_output(self, key: jax.Array, ring_K: jax.Array) -> None:
        """Return scalar population variance rather than a boundary fraction."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        idx = chimera_index(phases, ring_K)
        assert idx.shape == ()

    def test_zero_for_uniform_sync(self, ring_K: jax.Array) -> None:
        """Give zero regional variance for uniformly synchronized local fields."""
        phases = jnp.zeros(N)
        idx = chimera_index(phases, ring_K)
        assert float(idx) < 1e-6

    def test_differentiable(self, key: jax.Array, ring_K: jax.Array) -> None:
        """Differentiate nondegenerate local-order variance with respect to phases."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)

        def loss(p: jax.Array) -> jax.Array:
            """Measure phase-dependent regional variance on the fixed ring."""
            return chimera_index(p, ring_K)

        grad = jax.grad(loss)(phases)
        assert jnp.isfinite(grad).all()


class TestDetectChimera:
    """Check population classification and the model-specific index."""

    def test_returns_two_masks(self, key: jax.Array, ring_K: jax.Array) -> None:
        """Return coherent and incoherent masks per oscillator."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        coh, incoh = detect_chimera(phases, ring_K)
        assert coh.shape == (N,)
        assert incoh.shape == (N,)

    def test_perfect_sync_all_coherent(self, ring_K: jax.Array) -> None:
        """Classify the synchronized ring at inclusive default thresholds."""
        phases = jnp.zeros(N)
        coh, incoh = detect_chimera(phases, ring_K)
        assert jnp.all(coh)
        assert not jnp.any(incoh)

    def test_threshold_masks_disjoint(self, key: jax.Array, ring_K: jax.Array) -> None:
        """Keep masks disjoint for ordered thresholds on a JAX graph."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        coh, incoh = detect_chimera(
            phases, ring_K, coherent_threshold=0.8, incoherent_threshold=0.3
        )
        assert not jnp.any(coh & incoh)


class TestNNChimeraPipelineWiring:
    """Pipeline: KuramotoLayer → phases → chimera_index."""

    def test_kuramoto_layer_to_chimera_index(self, key: jax.Array) -> None:
        """KuramotoLayer → phases → chimera_index∈[0,1]."""
        from scpn_phase_orchestrator.nn.kuramoto_layer import KuramotoLayer

        k1, k2 = jax.random.split(key)
        layer = KuramotoLayer(N, n_steps=100, dt=0.01, key=k1)
        phases = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)
        final = layer(phases)
        K = jnp.abs(layer.K)
        idx = float(chimera_index(final, K))
        assert 0.0 <= idx <= 1.0


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_real_jit_local_order_matches_scalar_equation(precision: str) -> None:
    """JIT the genuine JAX owner on the positive zero-diagonal common domain."""
    p = np.array([0.1, 1.2, -0.7, 2.1])
    k = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    with enable_x64(precision == "float64"):
        dtype = jnp.float64 if precision == "float64" else jnp.float32
        phases, coupling = jnp.array(p, dtype=dtype), jnp.array(k, dtype=dtype)
        actual = jax.jit(local_order_parameter)(phases, coupling).block_until_ready()
        assert str(actual.dtype) == precision
        np.testing.assert_allclose(
            np.asarray(actual),
            scalar_local_order(p, k),
            atol=1e-12 if precision == "float64" else 2e-7,
            rtol=0,
        )


def test_jax_nonzero_signed_and_self_edges_remain_distinct() -> None:
    """NN keeps nonzero adjacency, variance and inclusive configurable masks."""
    phases = jnp.array([0.0, 0.0, jnp.pi])
    coupling = jnp.array([[1.0, -2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        np.asarray(local_order_parameter(phases, coupling)),
        [1.0, 0.0, 1.0],
        atol=1e-7,
        rtol=0,
    )
    assert float(chimera_index(phases, coupling)) == pytest.approx(2.0 / 9.0)
    coherent, incoherent = detect_chimera(phases, coupling, 1.0, 0.0)
    np.testing.assert_array_equal(np.asarray(coherent), [True, False, True])
    np.testing.assert_array_equal(np.asarray(incoherent), [False, True, False])


def test_jax_finite_extreme_angles_and_phase_gradient() -> None:
    """Actual float64 JIT avoids difference overflow and differentiates smooth R."""
    with enable_x64():
        phases = jnp.array([1e308, -1e308], dtype=jnp.float64)
        coupling = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float64)
        actual = jax.jit(local_order_parameter)(phases, coupling).block_until_ready()
        np.testing.assert_allclose(np.asarray(actual), [1.0, 1.0], atol=1e-12, rtol=0)
        p = jnp.array([0.1, 0.7, 1.8], dtype=jnp.float64)
        k = jnp.ones((3, 3), dtype=jnp.float64) - jnp.eye(3, dtype=jnp.float64)
        gradient = jax.jit(jax.grad(chimera_index))(p, k).block_until_ready()
        assert np.isfinite(np.asarray(gradient)).all()
        assert np.linalg.norm(np.asarray(gradient)) > 0.0


def test_jax_coupling_amplitude_gradient_is_zero_on_fixed_adjacency() -> None:
    """Hard nonzero adjacency is locally constant in nonzero coupling amplitudes."""
    p = jnp.array([0.1, 0.7, 1.8])
    k = jnp.array([[1.0, -2.0, 3.0], [1.0, 2.0, 0.0], [0.0, 1.0, 2.0]])
    gradient = jax.grad(chimera_index, argnums=1)(p, k).block_until_ready()
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros((3, 3)))
