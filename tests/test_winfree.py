# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Winfree model

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.functional import (
    winfree_forward,
    winfree_rk4_step,
    winfree_step,
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
        omegas = jnp.ones(N)
        out = winfree_step(phases, omegas, 1.0, DT)
        assert out.shape == (N,)

    def test_phases_in_range(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        out = winfree_step(phases, omegas, 1.0, DT)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)

    def test_zero_coupling_free_rotation(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        out = winfree_step(phases, omegas, 0.0, DT)
        expected = (phases + DT * omegas) % (2.0 * jnp.pi)
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_rk4_shape(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        out = winfree_rk4_step(phases, omegas, 1.0, DT)
        assert out.shape == (N,)


class TestForward:
    def test_trajectory_shape(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        final, traj = winfree_forward(phases, omegas, 1.0, DT, N_STEPS)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        final, traj = winfree_forward(phases, omegas, 1.0, DT, N_STEPS)
        assert jnp.allclose(final, traj[-1], atol=1e-6)

    def test_differentiable(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)

        def loss(p):
            f, _ = winfree_forward(p, omegas, 1.0, DT, N_STEPS)
            return jnp.mean(f)

        grad = jax.grad(loss)(phases)
        assert jnp.isfinite(grad).all()


class TestWinfreePhysics:
    """Verify Winfree model physics: dθ_i/dt = ω_i + K·Q(θ_i)·Σ P(θ_j).
    Coupling should synchronise oscillators (R→1 for strong K)."""

    def test_strong_coupling_synchronises(self, key):
        """Strong coupling K should drive R toward 1."""
        from scpn_phase_orchestrator.nn.functional import order_parameter

        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.zeros(N)  # pure coupling
        final, _ = winfree_forward(phases, omegas, 5.0, DT, 500)
        r = float(order_parameter(final))
        assert r > 0.5, f"Strong Winfree coupling should synchronise: R={r:.3f}"

    def test_euler_rk4_agree_small_dt(self, key):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        euler = winfree_step(phases, omegas, 1.0, 0.001)
        rk4 = winfree_rk4_step(phases, omegas, 1.0, 0.001)
        err = float(jnp.max(jnp.abs(jnp.sin(euler - rk4))))
        assert err < 0.01, f"Euler-RK4 error {err:.4e} too large at dt=0.001"

    def test_pipeline_to_order_parameter(self, key):
        """Winfree output must be valid input for order_parameter."""
        from scpn_phase_orchestrator.nn.functional import order_parameter

        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        omegas = jnp.ones(N)
        final, _ = winfree_forward(phases, omegas, 0.5, DT, 100)
        r = float(order_parameter(final))
        assert 0.0 <= r <= 1.0


# Pipeline wiring: Winfree tests exercise winfree_forward →
# order_parameter (test_pipeline_to_order_parameter,
# test_strong_coupling_synchronises). JAX grad + Euler/RK4.
