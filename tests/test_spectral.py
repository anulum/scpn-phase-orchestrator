# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral coupling tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.spectral import (
    critical_coupling,
    fiedler_partition,
    fiedler_value,
    fiedler_vector,
    graph_laplacian,
    spectral_gap,
    sync_convergence_rate,
)


def _chain_knm(n: int, k: float = 1.0) -> np.ndarray:
    """Chain graph: K_ij = k for |i-j|=1, else 0."""
    knm = np.zeros((n, n))
    for i in range(n - 1):
        knm[i, i + 1] = k
        knm[i + 1, i] = k
    return knm


def _complete_knm(n: int, k: float = 1.0) -> np.ndarray:
    """Complete graph: K_ij = k for all i≠j."""
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestGraphLaplacian:
    def test_row_sums_zero(self):
        L = graph_laplacian(_chain_knm(4))
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    def test_symmetric(self):
        L = graph_laplacian(_chain_knm(5))
        np.testing.assert_allclose(L, L.T)

    def test_complete_graph(self):
        n = 4
        L = graph_laplacian(_complete_knm(n))
        expected = n * np.eye(n) - np.ones((n, n))
        np.testing.assert_allclose(L, expected)


class TestFiedlerValue:
    def test_chain_4(self):
        # λ₂ of 4-node chain = 2 - 2*cos(π/4) ≈ 0.586
        lam2 = fiedler_value(_chain_knm(4))
        expected = 2.0 - 2.0 * np.cos(np.pi / 4)
        assert abs(lam2 - expected) < 1e-10

    def test_complete_graph(self):
        # λ₂ of complete graph K_n with weight k = n*k
        n, k = 5, 0.5
        lam2 = fiedler_value(_complete_knm(n, k))
        assert abs(lam2 - n * k) < 1e-10

    def test_disconnected_returns_zero(self):
        knm = np.zeros((4, 4))
        assert fiedler_value(knm) < 1e-12

    def test_two_nodes(self):
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert abs(fiedler_value(knm) - 2.0) < 1e-12


class TestFiedlerVector:
    def test_chain_bisects_at_midpoint(self):
        v2 = fiedler_vector(_chain_knm(6))
        signs = np.sign(v2)
        # Chain should partition roughly in half
        assert sum(signs > 0) >= 2
        assert sum(signs < 0) >= 2

    def test_length_matches(self):
        knm = _complete_knm(5)
        v2 = fiedler_vector(knm)
        assert len(v2) == 5


class TestCriticalCoupling:
    def test_identical_omegas(self):
        omegas = np.ones(4)
        kc = critical_coupling(omegas, _chain_knm(4))
        assert kc == 0.0

    def test_spread_omegas(self):
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        kc = critical_coupling(omegas, _chain_knm(4))
        lam2 = 2.0 - 2.0 * np.cos(np.pi / 4)
        assert abs(kc - 3.0 / lam2) < 1e-10

    def test_disconnected_returns_inf(self):
        omegas = np.array([1.0, 2.0, 3.0])
        kc = critical_coupling(omegas, np.zeros((3, 3)))
        assert kc == float("inf")


class TestFiedlerPartition:
    def test_two_groups(self):
        pos, neg = fiedler_partition(_chain_knm(6))
        assert len(pos) + len(neg) == 6
        assert len(pos) >= 1
        assert len(neg) >= 1

    def test_complete_graph_partitions(self):
        pos, neg = fiedler_partition(_complete_knm(4))
        assert len(pos) + len(neg) == 4


class TestSpectralGap:
    def test_complete_graph_zero_gap(self):
        # Complete graph: all eigenvalues above λ₁=0 are equal → gap = 0
        gap = spectral_gap(_complete_knm(4))
        assert abs(gap) < 1e-10

    def test_chain_positive_gap(self):
        gap = spectral_gap(_chain_knm(6))
        assert gap > 0

    def test_two_nodes(self):
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert spectral_gap(knm) == 0.0  # only 2 eigenvalues


class TestConvergenceRate:
    def test_positive_for_synced(self):
        knm = _complete_knm(4, 1.0)
        omegas = np.ones(4)
        mu = sync_convergence_rate(knm, omegas, gamma_max=0.0)
        assert mu > 0

    def test_decreases_with_gamma(self):
        knm = _complete_knm(4, 1.0)
        omegas = np.ones(4)
        mu0 = sync_convergence_rate(knm, omegas, gamma_max=0.0)
        mu1 = sync_convergence_rate(knm, omegas, gamma_max=1.0)
        assert mu1 < mu0

    def test_empty(self):
        assert sync_convergence_rate(np.zeros((0, 0)), np.array([])) == 0.0
