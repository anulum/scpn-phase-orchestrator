# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for the diffrax continuous-adjoint solver

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")
eqx = pytest.importorskip("equinox", reason="equinox required")
diffrax = pytest.importorskip("diffrax", reason="diffrax required")

from scpn_phase_orchestrator.nn.neural_ode import solve_ude_adjoint
from scpn_phase_orchestrator.nn.ude import CouplingResidual, ude_kuramoto_forward

N = 4


def _angular_distance(a: jax.Array, b: jax.Array) -> jax.Array:
    """Wrap-aware distance between two phase vectors, in ``[0, π]``."""
    return jnp.abs(jnp.angle(jnp.exp(1j * (a - b))))


@pytest.fixture()
def setup():
    key = jax.random.PRNGKey(7)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    phases = jax.random.uniform(k1, (N,), maxval=2.0 * jnp.pi)
    omegas = jax.random.normal(k2, (N,)) * 0.5
    raw = jax.random.normal(k3, (N, N)) * 0.1
    coupling = (raw + raw.T) / 2.0
    residual = CouplingResidual(hidden=8, key=k4)
    return phases, omegas, coupling, residual


class TestForwardConvergence:
    def test_matches_fine_euler_within_tol(self, setup):
        """Adaptive Tsit5 agrees with a fine explicit-Euler roll-out."""
        phases, omegas, coupling, residual = setup
        n_steps = 500
        dt = 1e-3
        t1 = n_steps * dt
        euler_final, _ = ude_kuramoto_forward(
            phases, omegas, coupling, residual, dt, n_steps
        )
        ode_final = solve_ude_adjoint(
            phases, omegas, coupling, residual, t1=t1, dt0=dt, rtol=1e-9, atol=1e-9
        )
        assert float(jnp.max(_angular_distance(euler_final, ode_final))) < 1e-2

    def test_final_shape_and_range(self, setup):
        phases, omegas, coupling, residual = setup
        final = solve_ude_adjoint(phases, omegas, coupling, residual, t1=0.5)
        assert final.shape == (N,)
        assert float(jnp.min(final)) >= 0.0
        assert float(jnp.max(final)) < 2.0 * jnp.pi

    def test_unwrapped_output_can_exceed_two_pi(self, setup):
        """With ``wrap=False`` the raw integrated phase is returned."""
        phases, omegas, coupling, residual = setup
        wrapped = solve_ude_adjoint(phases, omegas, coupling, residual, t1=2.0)
        unwrapped = solve_ude_adjoint(
            phases, omegas, coupling, residual, t1=2.0, wrap=False
        )
        assert jnp.allclose(wrapped, unwrapped % (2.0 * jnp.pi), atol=1e-5)


class TestTrajectory:
    def test_saveat_returns_trajectory(self, setup):
        phases, omegas, coupling, residual = setup
        ts = jnp.linspace(0.0, 1.0, 5)
        traj = solve_ude_adjoint(
            phases, omegas, coupling, residual, t1=1.0, saveat_ts=ts
        )
        assert traj.shape == (5, N)


class TestSolversAndAdjoints:
    def test_custom_solver_dopri5(self, setup):
        phases, omegas, coupling, residual = setup
        final = solve_ude_adjoint(
            phases, omegas, coupling, residual, t1=0.5, solver=diffrax.Dopri5()
        )
        assert final.shape == (N,)

    def test_default_checkpoint_adjoint_gradient_is_finite(self, setup):
        phases, omegas, coupling, residual = setup

        def loss(coupling_matrix: jax.Array) -> jax.Array:
            final = solve_ude_adjoint(phases, omegas, coupling_matrix, residual, t1=0.5)
            return jnp.sum(jnp.sin(final))

        grad = jax.grad(loss)(coupling)
        assert grad.shape == (N, N)
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_backsolve_adjoint_gradient_is_finite(self, setup):
        phases, omegas, coupling, residual = setup

        def loss(coupling_matrix: jax.Array) -> jax.Array:
            final = solve_ude_adjoint(
                phases,
                omegas,
                coupling_matrix,
                residual,
                t1=0.5,
                adjoint=diffrax.BacksolveAdjoint(),
            )
            return jnp.sum(jnp.sin(final))

        grad = jax.grad(loss)(coupling)
        assert bool(jnp.all(jnp.isfinite(grad)))


class TestValidation:
    def test_non_1d_phases_rejected(self, setup):
        _, omegas, coupling, residual = setup
        bad = jnp.zeros((N, 1))
        with pytest.raises(ValueError, match="one-dimensional"):
            solve_ude_adjoint(bad, omegas, coupling, residual, t1=0.5)

    def test_non_positive_t1_rejected(self, setup):
        phases, omegas, coupling, residual = setup
        with pytest.raises(ValueError, match="t1 must be positive"):
            solve_ude_adjoint(phases, omegas, coupling, residual, t1=0.0)

    def test_non_positive_dt0_rejected(self, setup):
        phases, omegas, coupling, residual = setup
        with pytest.raises(ValueError, match="dt0 must be positive"):
            solve_ude_adjoint(phases, omegas, coupling, residual, t1=0.5, dt0=0.0)

    def test_non_positive_rtol_rejected(self, setup):
        phases, omegas, coupling, residual = setup
        with pytest.raises(ValueError, match="rtol and atol must be positive"):
            solve_ude_adjoint(phases, omegas, coupling, residual, t1=0.5, rtol=0.0)

    def test_non_positive_atol_rejected(self, setup):
        phases, omegas, coupling, residual = setup
        with pytest.raises(ValueError, match="rtol and atol must be positive"):
            solve_ude_adjoint(phases, omegas, coupling, residual, t1=0.5, atol=-1.0)

    def test_non_positive_max_steps_rejected(self, setup):
        phases, omegas, coupling, residual = setup
        with pytest.raises(ValueError, match="max_steps must be positive"):
            solve_ude_adjoint(phases, omegas, coupling, residual, t1=0.5, max_steps=0)


class TestLazyExport:
    def test_symbol_resolves_through_package(self):
        from scpn_phase_orchestrator import nn

        assert nn.solve_ude_adjoint is solve_ude_adjoint
