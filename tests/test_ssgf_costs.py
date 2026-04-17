# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF cost tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.ssgf.costs import SSGFCosts, compute_ssgf_costs


class TestSSGFCosts:
    def test_synced_low_c1(self):
        W = np.full((4, 4), 0.5)
        np.fill_diagonal(W, 0.0)
        phases = np.zeros(4)
        costs = compute_ssgf_costs(W, phases)
        assert costs.c1_sync < 0.01

    def test_spread_high_c1(self):
        W = np.full((4, 4), 0.5)
        np.fill_diagonal(W, 0.0)
        phases = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        costs = compute_ssgf_costs(W, phases)
        assert costs.c1_sync > 0.5

    def test_connected_negative_c2(self):
        W = np.full((4, 4), 1.0)
        np.fill_diagonal(W, 0.0)
        costs = compute_ssgf_costs(W, np.zeros(4))
        assert costs.c2_spectral_gap < 0  # -λ₂ < 0 for connected graph

    def test_sparse_low_c3(self):
        W = np.zeros((4, 4))
        W[0, 1] = W[1, 0] = 0.5
        costs = compute_ssgf_costs(W, np.zeros(4))
        dense_W = np.full((4, 4), 0.5)
        np.fill_diagonal(dense_W, 0.0)
        costs_dense = compute_ssgf_costs(dense_W, np.zeros(4))
        assert costs.c3_sparsity < costs_dense.c3_sparsity

    def test_symmetric_zero_c4(self):
        W = np.full((4, 4), 0.5)
        np.fill_diagonal(W, 0.0)
        costs = compute_ssgf_costs(W, np.zeros(4))
        assert abs(costs.c4_symmetry) < 1e-12

    def test_asymmetric_positive_c4(self):
        W = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
        costs = compute_ssgf_costs(W, np.zeros(3))
        assert costs.c4_symmetry > 0

    def test_u_total_is_weighted_sum(self):
        W = np.full((3, 3), 0.5)
        np.fill_diagonal(W, 0.0)
        phases = np.array([0.0, 0.5, 1.0])
        weights = (2.0, 1.0, 0.5, 0.3)
        costs = compute_ssgf_costs(W, phases, weights=weights)
        expected = (
            2.0 * costs.c1_sync
            + 1.0 * costs.c2_spectral_gap
            + 0.5 * costs.c3_sparsity
            + 0.3 * costs.c4_symmetry
        )
        assert abs(costs.u_total - expected) < 1e-10

    def test_returns_dataclass(self):
        costs = compute_ssgf_costs(np.eye(3), np.zeros(3))
        assert isinstance(costs, SSGFCosts)


class TestSSGFCostsPipelineWiring:
    """Pipeline: engine K_nm + phases → SSGF costs → SSGF optimisation."""

    def test_engine_state_to_ssgf_costs(self):
        """UPDEEngine → phases + K_nm → compute_ssgf_costs → u_total.
        Proves SSGF cost function consumes engine state."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 6
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        costs = compute_ssgf_costs(knm, phases)
        assert isinstance(costs, SSGFCosts)
        assert np.isfinite(costs.u_total)
        assert 0.0 <= costs.c1_sync <= 1.0  # sync cost bounded
