# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Kuramoto reservoir computing

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.reservoir import (
    reservoir_drive,
    reservoir_features,
    reservoir_predict,
    ridge_readout,
)

N = 8
D_IN = 2
D_OUT = 1
DT = 0.01
N_STEPS = 5
T = 20


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def setup(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
    omegas = jax.random.normal(k2, (N,))
    raw = jax.random.normal(k3, (N, N)) * 0.3
    K = (raw + raw.T) / 2.0
    W_in = jax.random.normal(k4, (N, D_IN)) * 0.1
    return phases, omegas, K, W_in


class TestReservoirFeatures:
    def test_output_shape(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        feat = reservoir_features(phases)
        assert feat.shape == (2 * N + 1,)

    def test_last_element_is_R(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        feat = reservoir_features(phases)
        from scpn_phase_orchestrator.nn.functional import order_parameter

        R = order_parameter(phases)
        assert jnp.isclose(feat[-1], R, atol=1e-6)

    def test_cos_sin_structure(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        feat = reservoir_features(phases)
        assert jnp.allclose(feat[:N], jnp.cos(phases), atol=1e-6)
        assert jnp.allclose(feat[N : 2 * N], jnp.sin(phases), atol=1e-6)


class TestReservoirDrive:
    def test_output_shape(self, setup, key):
        phases, omegas, K, W_in = setup
        u = jax.random.normal(key, (T, D_IN))
        feat = reservoir_drive(phases, omegas, K, W_in, u, DT, N_STEPS)
        assert feat.shape == (T, 2 * N + 1)

    def test_different_inputs_different_features(self, setup, key):
        phases, omegas, K, W_in = setup
        k1, k2 = jax.random.split(key)
        u1 = jax.random.normal(k1, (T, D_IN))
        u2 = jax.random.normal(k2, (T, D_IN)) * 5.0
        feat1 = reservoir_drive(phases, omegas, K, W_in, u1, DT, N_STEPS)
        feat2 = reservoir_drive(phases, omegas, K, W_in, u2, DT, N_STEPS)
        assert not jnp.allclose(feat1, feat2)

    def test_zero_input_still_evolves(self, setup):
        phases, omegas, K, W_in = setup
        u = jnp.zeros((T, D_IN))
        feat = reservoir_drive(phases, omegas, K, W_in, u, DT, N_STEPS)
        assert not jnp.allclose(feat[0], feat[-1])


class TestRidgeReadout:
    def test_output_shape(self, key):
        D_feat = 2 * N + 1
        features = jax.random.normal(key, (T, D_feat))
        targets = jax.random.normal(key, (T, D_OUT))
        W_out = ridge_readout(features, targets)
        assert W_out.shape == (D_feat, D_OUT)

    def test_perfect_linear_fit(self, key):
        """Ridge regression should perfectly fit a linear relationship."""
        D_feat = 5
        k1, k2 = jax.random.split(key)
        W_true = jax.random.normal(k1, (D_feat, 1))
        features = jax.random.normal(k2, (50, D_feat))
        targets = features @ W_true
        W_out = ridge_readout(features, targets, alpha=1e-8)
        preds = reservoir_predict(features, W_out)
        assert jnp.allclose(preds, targets, atol=1e-3)


class TestReservoirPredict:
    def test_output_shape(self, key):
        D_feat = 2 * N + 1
        features = jax.random.normal(key, (T, D_feat))
        W_out = jax.random.normal(key, (D_feat, D_OUT))
        preds = reservoir_predict(features, W_out)
        assert preds.shape == (T, D_OUT)


class TestEndToEnd:
    def test_sine_wave_regression(self, setup, key):
        """Full pipeline: drive reservoir with sine, fit readout to cosine."""
        phases, omegas, K, W_in = setup
        t = jnp.linspace(0, 4.0 * jnp.pi, T)
        u = jnp.sin(t)[:, jnp.newaxis]  # (T, 1)
        targets = jnp.cos(t)[:, jnp.newaxis]  # (T, 1)

        W_in_1d = W_in[:, :1]
        feat = reservoir_drive(phases, omegas, K, W_in_1d, u, DT, N_STEPS)
        W_out = ridge_readout(feat, targets)
        preds = reservoir_predict(feat, W_out)

        # Not expecting perfect fit with N=8 reservoir, just non-trivial
        mse = float(jnp.mean((preds - targets) ** 2))
        baseline_mse = float(jnp.mean(targets**2))
        assert mse < baseline_mse  # better than predicting zero


# Pipeline wiring: TestEndToEnd.test_sine_wave_regression drives the Kuramoto
# reservoir with input, fits readout, and verifies prediction — the full
# reservoir computing pipeline.
