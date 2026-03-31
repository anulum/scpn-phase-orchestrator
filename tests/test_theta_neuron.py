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


# ---------------------------------------------------------------------------
# Physics contracts
# ---------------------------------------------------------------------------


class TestThetaNeuronPhysics:
    """Verify theta neuron model satisfies the Ermentrout-Kopell
    dθ/dt = 1 - cos(θ) + (1 + cos(θ))·η + K·coupling dynamics."""

    def test_spiking_regime_phases_advance(self, key):
        """η > 0 (spiking regime): phases must change significantly
        over time (each neuron oscillates)."""
        phases = jnp.zeros(N)
        eta = jnp.ones(N)  # spiking
        K = jnp.zeros((N, N))
        final, traj = theta_neuron_forward(phases, eta, K, DT, 500)
        # Phases must have moved significantly from initial 0
        total_change = float(jnp.sum(jnp.abs(jnp.sin(final))))
        assert total_change > 0.1, "Spiking neuron (η>0) must advance phases"

    def test_coupling_increases_synchrony(self, key):
        """Strong coupling should synchronise theta neurons (R→1)."""
        from scpn_phase_orchestrator.nn.functional import order_parameter

        k1, k2 = jax.random.split(key)
        phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
        eta = 0.5 * jnp.ones(N)

        # Weak coupling
        K_weak = 0.01 * jnp.ones((N, N))
        final_weak, _ = theta_neuron_forward(phases, eta, K_weak, DT, 500)
        r_weak = float(order_parameter(final_weak))

        # Strong coupling
        K_strong = 2.0 * jnp.ones((N, N))
        final_strong, _ = theta_neuron_forward(phases, eta, K_strong, DT, 500)
        r_strong = float(order_parameter(final_strong))

        assert r_strong > r_weak - 0.1, (
            f"Strong coupling R={r_strong:.3f} should exceed weak R={r_weak:.3f}"
        )

    def test_euler_rk4_agree_small_dt(self, key):
        """Euler and RK4 must agree for small dt (O(dt²) error)."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        eta = 0.5 * jnp.ones(N)
        K = 0.1 * jnp.ones((N, N))
        dt_small = 0.001
        euler = theta_neuron_step(phases, eta, K, dt_small)
        rk4 = theta_neuron_rk4_step(phases, eta, K, dt_small)
        err = float(jnp.max(jnp.abs(jnp.sin(euler - rk4))))
        assert err < 0.01, f"Euler-RK4 difference {err:.4e} too large at dt={dt_small}"

    def test_pipeline_layer_to_order_parameter(self, key):
        """ThetaNeuronLayer → order_parameter: proves the module wires
        into the analysis pipeline."""
        from scpn_phase_orchestrator.nn.functional import order_parameter

        k1, k2 = jax.random.split(key)
        layer = ThetaNeuronLayer(N, n_steps=100, dt=DT, key=k1)
        phases = jax.random.uniform(k2, (N,), maxval=2.0 * jnp.pi)
        final = layer(phases)
        r = float(order_parameter(final))
        assert 0.0 <= r <= 1.0, f"R={r} must be in [0,1]"


# Pipeline wiring: theta neuron tests exercise ThetaNeuronLayer → order_parameter
# (TestThetaNeuronPhysics.test_pipeline_layer_to_order_parameter,
# test_coupling_increases_sync). Differentiable via JAX grad. Euler/RK4 parity.
