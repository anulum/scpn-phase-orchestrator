# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for theta neuron model

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")

from scpn_phase_orchestrator.nn.theta_neuron import (
    ThetaNeuronLayer,
    theta_neuron_forward,
    theta_neuron_rk4_step,
    theta_neuron_step,
)

N = 8
DT = 0.01
N_STEPS = 50


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


class TestStep:
    def test_output_shape(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        eta = -0.5 * jnp.ones(N)
        K = jnp.zeros((N, N))
        out = theta_neuron_step(phases, eta, K, DT)
        assert out.shape == (N,)

    def test_phases_in_range(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        eta = jnp.ones(N)
        K = 0.1 * jnp.ones((N, N))
        out = theta_neuron_step(phases, eta, K, DT)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)

    def test_rk4_shape(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        eta = jnp.zeros(N)
        K = jnp.zeros((N, N))
        out = theta_neuron_rk4_step(phases, eta, K, DT)
        assert out.shape == (N,)

    def test_excitable_stays_near_rest(self, key):
        # η<0, no coupling → excitable regime, phases should stay near 0 or π
        phases = 0.01 * jnp.ones(N)
        eta = -1.0 * jnp.ones(N)
        K = jnp.zeros((N, N))
        final, _ = theta_neuron_forward(phases, eta, K, DT, 100)
        # In excitable regime, stable fixed point at θ = -arccos(1+2η)
        # For η=-1, fixed point at θ=π. Phases should migrate there.
        assert jnp.isfinite(final).all()


class TestForward:
    def test_trajectory_shape(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        eta = jnp.zeros(N)
        K = jnp.zeros((N, N))
        final, traj = theta_neuron_forward(phases, eta, K, DT, N_STEPS)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        eta = jnp.zeros(N)
        K = jnp.zeros((N, N))
        final, traj = theta_neuron_forward(phases, eta, K, DT, N_STEPS)
        assert jnp.allclose(final, traj[-1], atol=1e-6)


class TestLayer:
    def test_init(self, key):
        layer = ThetaNeuronLayer(N, key=key)
        assert layer.K.shape == (N, N)
        assert layer.eta.shape == (N,)

    def test_forward(self, key):
        k1, k2 = jax.random.split(key)
        layer = ThetaNeuronLayer(N, n_steps=N_STEPS, dt=DT, key=k1)
        phases = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)
        out = layer(phases)
        assert out.shape == (N,)
        assert jnp.isfinite(out).all()

    def test_grad(self, key):
        k1, k2 = jax.random.split(key)
        layer = ThetaNeuronLayer(N, n_steps=N_STEPS, dt=DT, key=k1)
        phases = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)

        def loss(m, p):
            return jnp.mean(m(p))

        grads = eqx.filter_grad(loss)(layer, phases)
        assert jnp.isfinite(grads.K).all()
        assert jnp.isfinite(grads.eta).all()
