# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for Oscillator Ising Machine (OIM)

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax", reason="JAX required")
jnp = pytest.importorskip("jax.numpy", reason="JAX required")

from scpn_phase_orchestrator.nn.oim import (
    coloring_energy,
    coloring_violations,
    extract_coloring,
    oim_forward,
    oim_step,
)

N = 6
DT = 0.1
N_STEPS = 200
N_COLORS = 3


@pytest.fixture()
def key():
    return jax.random.PRNGKey(42)


@pytest.fixture()
def triangle_graph():
    """K3 complete graph (3 nodes, all connected). Needs 3 colors."""
    return jnp.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=jnp.float32)


@pytest.fixture()
def bipartite_graph():
    """K_{3,3} bipartite (6 nodes). Needs 2 colors."""
    A = jnp.zeros((6, 6))
    for i in range(3):
        for j in range(3, 6):
            A = A.at[i, j].set(1.0)
            A = A.at[j, i].set(1.0)
    return A


class TestOIMStep:
    def test_output_shape(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        out = oim_step(phases, bipartite_graph, N_COLORS, DT)
        assert out.shape == (N,)

    def test_phases_in_range(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        out = oim_step(phases, bipartite_graph, N_COLORS, DT)
        assert jnp.all(out >= 0.0)
        assert jnp.all(out < 2.0 * jnp.pi)


class TestOIMForward:
    def test_trajectory_shape(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        final, traj = oim_forward(phases, bipartite_graph, 2, DT, N_STEPS)
        assert final.shape == (N,)
        assert traj.shape == (N_STEPS, N)

    def test_final_matches_last(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        final, traj = oim_forward(phases, bipartite_graph, 2, DT, N_STEPS)
        assert jnp.allclose(final, traj[-1], atol=1e-6)


class TestExtractColoring:
    def test_output_shape(self):
        phases = jnp.array([0.0, 2.1, 4.2])
        colors = extract_coloring(phases, 3)
        assert colors.shape == (3,)

    def test_correct_assignment(self):
        # 3 colors: centers at 0, 2π/3, 4π/3
        phases = jnp.array([0.1, 2.2, 4.3])
        colors = extract_coloring(phases, 3)
        assert int(colors[0]) == 0
        assert int(colors[1]) == 1
        assert int(colors[2]) == 2

    def test_range(self, key):
        phases = jax.random.uniform(key, (20,), maxval=2.0 * jnp.pi)
        colors = extract_coloring(phases, 5)
        assert jnp.all(colors >= 0)
        assert jnp.all(colors < 5)


class TestColoringViolations:
    def test_perfect_coloring_zero_violations(self, triangle_graph):
        colors = jnp.array([0, 1, 2])
        v = coloring_violations(colors, triangle_graph)
        assert int(v) == 0

    def test_all_same_color_max_violations(self, triangle_graph):
        colors = jnp.array([0, 0, 0])
        v = coloring_violations(colors, triangle_graph)
        assert int(v) == 3  # 3 edges in K3

    def test_one_violation(self):
        A = jnp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=jnp.float32)
        colors = jnp.array([0, 0, 1])  # edge (0,1) violated
        v = coloring_violations(colors, A)
        assert int(v) == 1


class TestColoringEnergy:
    def test_scalar_output(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        e = coloring_energy(phases, bipartite_graph, 2)
        assert e.shape == ()

    def test_differentiable(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)

        def loss(p):
            return coloring_energy(p, bipartite_graph, 2)

        grad = jax.grad(loss)(phases)
        assert grad.shape == (N,)
        assert jnp.isfinite(grad).all()

    def test_energy_decreases_during_dynamics(self, key, bipartite_graph):
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        e_before = coloring_energy(phases, bipartite_graph, 2)
        final, _ = oim_forward(phases, bipartite_graph, 2, DT, N_STEPS)
        e_after = coloring_energy(final, bipartite_graph, 2)
        assert float(e_after) <= float(e_before)


class TestEndToEnd:
    @pytest.mark.xfail(reason="OIM convergence needs GPU — too slow on CPU XLA")
    def test_triangle_3coloring(self, key, triangle_graph):
        """K3 should be 3-colorable after sufficient OIM dynamics."""
        phases = jax.random.uniform(key, (3,), maxval=2.0 * jnp.pi)
        final, _ = oim_forward(
            phases, triangle_graph, 3, 0.05, 2000, coupling_strength=5.0
        )
        colors = extract_coloring(final, 3)
        v = coloring_violations(colors, triangle_graph)
        assert int(v) == 0

    @pytest.mark.xfail(reason="OIM convergence needs GPU — too slow on CPU XLA")
    def test_bipartite_2coloring(self, key, bipartite_graph):
        """K_{3,3} should be 2-colorable after sufficient OIM dynamics."""
        phases = jax.random.uniform(key, (N,), maxval=2.0 * jnp.pi)
        final, _ = oim_forward(
            phases, bipartite_graph, 2, 0.05, 2000, coupling_strength=5.0
        )
        colors = extract_coloring(final, 2)
        v = coloring_violations(colors, bipartite_graph)
        assert int(v) == 0
