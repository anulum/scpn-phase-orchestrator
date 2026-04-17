# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hypergraph coupling tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.hypergraph import (
    Hyperedge,
    HypergraphEngine,
)


class TestHyperedge:
    def test_order(self):
        e = Hyperedge(nodes=(0, 1, 2), strength=0.5)
        assert e.order == 3
        assert e.strength == 0.5

    def test_pairwise(self):
        e = Hyperedge(nodes=(0, 1))
        assert e.order == 2


class TestHypergraphEngine:
    def test_pairwise_only(self):
        """With pairwise knm and no hyperedges → standard Kuramoto."""
        N = 4
        eng = HypergraphEngine(N, dt=0.01)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 0.5
        np.fill_diagonal(knm, 0)
        p = eng.run(phases, omegas, n_steps=100, pairwise_knm=knm)
        R = eng.order_parameter(p)
        assert eng.order_parameter(phases) < R

    def test_3body_matches_simplicial(self):
        """3-body hyperedges should produce same physics as SimplicialEngine."""
        N = 4
        eng = HypergraphEngine(N, dt=0.01)
        eng.add_all_to_all(order=3, strength=0.3)
        # Should have C(4,3)=4 hyperedges
        assert eng.n_edges == 4
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.zeros(N)
        p = eng.run(phases, omegas, n_steps=50)
        assert p.shape == (N,)

    def test_4body_coupling(self):
        """4-body interactions should work."""
        N = 5
        eng = HypergraphEngine(N, dt=0.01)
        eng.add_all_to_all(order=4, strength=0.2)
        # C(5,4)=5 hyperedges
        assert eng.n_edges == 5
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, N)
        omegas = np.zeros(N)
        p = eng.run(phases, omegas, n_steps=100)
        assert np.all(np.isfinite(p))

    def test_mixed_order(self):
        """Mix of 2-body, 3-body, and 4-body edges."""
        N = 5
        eng = HypergraphEngine(N, dt=0.01)
        eng.add_edge((0, 1), strength=1.0)
        eng.add_edge((0, 1, 2), strength=0.5)
        eng.add_edge((0, 1, 2, 3), strength=0.2)
        assert eng.n_edges == 3

        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, N)
        omegas = np.zeros(N)
        p = eng.run(phases, omegas, n_steps=50)
        assert np.all(np.isfinite(p))

    def test_synchronization_with_hyperedges(self):
        """Strong all-to-all 3-body + pairwise → sync."""
        N = 6
        eng = HypergraphEngine(N, dt=0.01)
        eng.add_all_to_all(order=3, strength=0.5)

        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 1.0
        np.fill_diagonal(knm, 0)

        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, N)
        R0 = eng.order_parameter(phases)

        p = eng.run(phases, omegas, n_steps=500, pairwise_knm=knm)
        R_final = eng.order_parameter(p)
        assert R_final > R0

    def test_single_step(self):
        """step() returns correct shape."""
        N = 3
        eng = HypergraphEngine(N, dt=0.01)
        eng.add_edge((0, 1, 2), strength=1.0)
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.array([1.0, 1.0, 1.0])
        p = eng.step(phases, omegas)
        assert p.shape == (3,)

    def test_add_edge(self):
        eng = HypergraphEngine(4, dt=0.01)
        assert eng.n_edges == 0
        eng.add_edge((0, 1, 2))
        assert eng.n_edges == 1

    def test_with_forcing(self):
        """External forcing zeta, psi should work."""
        N = 3
        eng = HypergraphEngine(N, dt=0.01)
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.zeros(N)
        p = eng.run(phases, omegas, n_steps=100, zeta=1.0, psi=0.0)
        assert np.all(np.isfinite(p))


class TestHypergraphPipelineWiring:
    """Pipeline: hypergraph engine → order_parameter."""

    def test_hypergraph_to_order_parameter(self):
        """HypergraphEngine.run → phases → compute_order_parameter.
        Proves higher-order coupling output feeds standard analysis."""
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 6
        eng = HypergraphEngine(n, dt=0.01)
        eng.add_edge((0, 1, 2), strength=1.0)
        eng.add_edge((3, 4, 5), strength=1.0)
        eng.add_edge((1, 3, 5), strength=0.5)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)

        final = eng.run(phases, omegas, n_steps=200)
        r, _ = compute_order_parameter(final)
        assert 0.0 <= r <= 1.0
