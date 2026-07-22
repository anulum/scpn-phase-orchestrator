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
optax = pytest.importorskip("optax", reason="optax required")

from scpn_phase_orchestrator.nn.neural_ode import solve_ude_adjoint
from scpn_phase_orchestrator.nn.training import (
    generate_kuramoto_data,
    train,
    trajectory_loss,
)
from scpn_phase_orchestrator.nn.ude import (
    CouplingResidual,
    UDEKuramotoLayer,
    ude_kuramoto_forward,
)

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


class TestBackendSwitch:
    def test_euler_default_is_unchanged(self):
        """The Euler default still returns the exact reproducible roll-out."""
        key = jax.random.PRNGKey(3)
        layer = UDEKuramotoLayer(n=4, n_steps=10, dt=0.02, hidden=8, key=key)
        phases = jax.random.uniform(jax.random.PRNGKey(4), (4,), maxval=2.0 * jnp.pi)
        final, traj = layer.forward_with_trajectory(phases)
        ref_final, ref_traj = ude_kuramoto_forward(
            phases, layer.omegas, layer.K, layer.residual, layer.dt, layer.n_steps
        )
        assert jnp.array_equal(final, ref_final)
        assert jnp.array_equal(traj, ref_traj)

    def test_diffrax_backend_shape_and_convergence(self):
        """The diffrax backend samples the same grid and tracks fine Euler."""
        key = jax.random.PRNGKey(5)
        layer = UDEKuramotoLayer(n=4, n_steps=200, dt=1e-3, hidden=8, key=key)
        phases = jax.random.uniform(jax.random.PRNGKey(6), (4,), maxval=2.0 * jnp.pi)
        final, traj = layer.forward_with_trajectory(phases, backend="diffrax")
        assert traj.shape == (200, 4)
        euler_final, _ = layer.forward_with_trajectory(phases)
        assert float(jnp.max(_angular_distance(final, euler_final))) < 1e-2

    def test_invalid_backend_rejected(self):
        key = jax.random.PRNGKey(7)
        layer = UDEKuramotoLayer(n=3, n_steps=5, dt=0.02, hidden=4, key=key)
        phases = jnp.zeros(3)
        with pytest.raises(ValueError, match="backend must be"):
            layer.forward_with_trajectory(phases, backend="rk4")


class TestCheckpointedAdjointTrainingPath:
    def test_diffrax_training_reduces_loss(self):
        """End-to-end: the checkpointed-adjoint path trains the UDE layer down."""
        data_key = jax.random.PRNGKey(11)
        k_true, omegas_true, phases0, observed = generate_kuramoto_data(
            N=4, T=12, dt=0.05, K_scale=0.3, key=data_key
        )
        del k_true, omegas_true
        layer = UDEKuramotoLayer(
            n=4, n_steps=12, dt=0.05, hidden=8, key=jax.random.PRNGKey(12)
        )

        def loss_fn(model: UDEKuramotoLayer) -> jax.Array:
            return trajectory_loss(model, phases0, observed, backend="diffrax")

        initial = float(loss_fn(layer))
        trained, history = train(layer, loss_fn, optax.adam(5e-2), n_epochs=25)
        assert len(history) == 25
        assert history[-1] < initial

    def test_trajectory_loss_euler_backend_matches_direct_call(self):
        """The default backend leaves ``trajectory_loss`` byte-for-byte."""
        data_key = jax.random.PRNGKey(21)
        _, _, phases0, observed = generate_kuramoto_data(
            N=4, T=10, dt=0.05, key=data_key
        )
        layer = UDEKuramotoLayer(
            n=4, n_steps=10, dt=0.05, hidden=8, key=jax.random.PRNGKey(22)
        )
        via_default = trajectory_loss(layer, phases0, observed)
        via_euler = trajectory_loss(layer, phases0, observed, backend="euler")
        assert jnp.array_equal(via_default, via_euler)


class TestStiffnessGuard:
    def test_max_steps_exhaustion_raises_by_default(self, setup):
        """A solve that cannot reach ``t1`` within ``max_steps`` raises."""
        phases, omegas, coupling, residual = setup
        with pytest.raises(RuntimeError):
            solve_ude_adjoint(
                phases, omegas, coupling, residual, t1=5.0, dt0=1e-3, max_steps=1
            )

    def test_throw_false_recovers_the_solution(self, setup):
        """``throw=False`` returns the (incomplete) solution instead of raising."""
        phases, omegas, coupling, residual = setup
        out = solve_ude_adjoint(
            phases,
            omegas,
            coupling,
            residual,
            t1=5.0,
            dt0=1e-3,
            max_steps=1,
            throw=False,
        )
        assert out.shape == (N,)


class TestLazyExport:
    def test_symbol_resolves_through_package(self):
        from scpn_phase_orchestrator import nn

        assert nn.solve_ude_adjoint is solve_ude_adjoint
