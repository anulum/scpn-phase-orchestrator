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
