# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based Hodge & spectral proofs

"""Hypothesis-driven invariant proofs for Hodge decomposition and
spectral graph analysis (Laplacian, Fiedler, spectral gap).

Each test is a computational theorem enforcing algebraic identities
from spectral graph theory and Hodge theory on simplicial complexes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.hodge import hodge_decomposition
from scpn_phase_orchestrator.coupling.spectral import (
    critical_coupling,
    fiedler_partition,
    fiedler_value,
    fiedler_vector,
    graph_laplacian,
    spectral_gap,
    sync_convergence_rate,
)

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


def _asymmetric_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    knm = rng.uniform(0.1, 1.0, (n, n)) * strength
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Graph Laplacian ──────────────────────────────────────────────────


class TestGraphLaplacianInvariants:
    """L = D - W: row sums = 0, positive semi-definite, symmetric for symmetric K."""

    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=60, deadline=None)
    def test_row_sums_zero(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        L = graph_laplacian(knm)
        np.testing.assert_allclose(L.sum(axis=1), 0.0, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=60, deadline=None)
    def test_symmetric_for_symmetric_k(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        L = graph_laplacian(knm)
        np.testing.assert_allclose(L, L.T, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_positive_semidefinite(self, n: int, seed: int) -> None:
        """All eigenvalues ≥ 0."""
        knm = _connected_knm(n, seed=seed)
        L = graph_laplacian(knm)
        eigs = np.linalg.eigvalsh(L)
        assert np.all(eigs >= -1e-10)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_smallest_eigenvalue_zero(self, n: int, seed: int) -> None:
        """Connected graph: λ₁ ≈ 0."""
        knm = _connected_knm(n, seed=seed)
        L = graph_laplacian(knm)
        eigs = np.linalg.eigvalsh(L)
        assert abs(eigs[0]) < 1e-10

    @given(n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_zero_knm_all_zeros(self, n: int) -> None:
        L = graph_laplacian(np.zeros((n, n)))
        np.testing.assert_allclose(L, 0.0, atol=1e-15)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_diagonal_nonnegative(self, n: int, seed: int) -> None:
        """Diagonal = degree ≥ 0."""
        knm = _connected_knm(n, seed=seed)
        L = graph_laplacian(knm)
        assert np.all(np.diag(L) >= -1e-12)


# ── 2. Fiedler value (algebraic connectivity) ───────────────────────────


class TestFiedlerValueInvariants:
    """λ₂(L) ≥ 0 always; > 0 iff connected; = 0 iff disconnected."""

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_nonnegative(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        assert fiedler_value(knm) >= -1e-10

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_connected_positive(self, n: int, seed: int) -> None:
        """All-to-all → λ₂ > 0."""
        knm = _connected_knm(n, seed=seed)
        assert fiedler_value(knm) > 1e-6

    @given(n=st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_disconnected_zero(self, n: int) -> None:
        assert abs(fiedler_value(np.zeros((n, n)))) < 1e-10

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_stronger_coupling_higher_lambda2(self, n: int, seed: int) -> None:
        """Doubling all K → doubles λ₂ (Laplacian is linear in K)."""
        knm1 = _connected_knm(n, strength=1.0, seed=seed)
        knm2 = _connected_knm(n, strength=2.0, seed=seed)
        l2_1 = fiedler_value(knm1)
        l2_2 = fiedler_value(knm2)
        assert abs(l2_2 / l2_1 - 2.0) < 1e-6


# ── 3. Fiedler vector & partition ────────────────────────────────────────


class TestFiedlerVectorPartition:
    """Fiedler vector properties and partition completeness."""

    @given(
        n=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_fiedler_vector_length(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        v2 = fiedler_vector(knm)
        assert len(v2) == n

    @given(
        n=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_partition_covers_all(self, n: int, seed: int) -> None:
        """Partition groups must cover all N oscillators."""
        knm = _connected_knm(n, seed=seed)
        pos, neg = fiedler_partition(knm)
        assert sorted(pos + neg) == list(range(n))

    @given(
        n=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_partition_disjoint(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        pos, neg = fiedler_partition(knm)
        assert len(set(pos) & set(neg)) == 0


# ── 4. Spectral gap ─────────────────────────────────────────────────────


class TestSpectralGapInvariants:

    @given(
        n=st.integers(min_value=3, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_nonnegative(self, n: int, seed: int) -> None:
        """λ₃ ≥ λ₂ → gap ≥ 0."""
        knm = _connected_knm(n, seed=seed)
        gap = spectral_gap(knm)
        assert gap >= -1e-10

    @pytest.mark.parametrize("n", [2])
    def test_n2_returns_zero(self, n: int) -> None:
        """N=2 has only 2 eigenvalues → gap = 0."""
        knm = _connected_knm(n)
        assert spectral_gap(knm) == 0.0

    @given(
        n=st.integers(min_value=3, max_value=8),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_scales_linearly(self, n: int, seed: int) -> None:
        """Doubling K doubles the spectral gap."""
        knm1 = _connected_knm(n, strength=1.0, seed=seed)
        knm2 = _connected_knm(n, strength=2.0, seed=seed)
        g1 = spectral_gap(knm1)
        g2 = spectral_gap(knm2)
        if g1 > 1e-8:
            assert abs(g2 / g1 - 2.0) < 1e-6


# ── 5. Critical coupling ────────────────────────────────────────────────


class TestCriticalCouplingInvariants:

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_positive_for_connected(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        kc = critical_coupling(omegas, knm)
        assert kc > 0 or kc == float("inf")
        assert kc < float("inf")

    @given(n=st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_disconnected_inf(self, n: int) -> None:
        omegas = np.linspace(-1, 1, n)
        assert critical_coupling(omegas, np.zeros((n, n))) == float("inf")

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_identical_freqs_zero(self, n: int, seed: int) -> None:
        """Identical ω → spread=0 → K_c=0."""
        omegas = np.full(n, 3.0)
        knm = _connected_knm(n, seed=seed)
        kc = critical_coupling(omegas, knm)
        assert abs(kc) < 1e-12

    def test_finite(self) -> None:
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = _connected_knm(4)
        kc = critical_coupling(omegas, knm)
        assert np.isfinite(kc)


# ── 6. Sync convergence rate ────────────────────────────────────────────


class TestSyncConvergenceRate:

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_nonnegative_small_gamma(self, n: int, seed: int) -> None:
        """γ < π/2 → cos(γ) > 0 → rate ≥ 0."""
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        rate = sync_convergence_rate(knm, omegas, gamma_max=0.5)
        assert rate >= -1e-12

    def test_empty_zero(self) -> None:
        assert sync_convergence_rate(np.zeros((0, 0)), np.array([])) == 0.0

    def test_zero_knm_zero(self) -> None:
        omegas = np.array([1.0, 2.0])
        assert sync_convergence_rate(np.zeros((2, 2)), omegas) == 0.0


# ── 7. Hodge decomposition ──────────────────────────────────────────────


class TestHodgeDecompositionInvariants:
    """gradient + curl + harmonic = total coupling force."""

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=60, deadline=None)
    def test_sum_equals_total(self, n: int, seed: int) -> None:
        """Reconstruction: gradient + curl + harmonic = total."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _asymmetric_knm(n, seed=seed)
        res = hodge_decomposition(knm, phases)
        total = np.sum(knm * np.cos(phases[np.newaxis, :] - phases[:, np.newaxis]), axis=1)
        np.testing.assert_allclose(
            res.gradient + res.curl + res.harmonic, total, atol=1e-10
        )

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_symmetric_k_zero_curl(self, n: int, seed: int) -> None:
        """Symmetric K → K_anti = 0 → curl = 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _connected_knm(n, seed=seed)
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_harmonic_near_zero(self, n: int, seed: int) -> None:
        """For exact sym/anti split, harmonic = numerical residual ≈ 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _asymmetric_knm(n, seed=seed)
        res = hodge_decomposition(knm, phases)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-10)

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_all_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _asymmetric_knm(n, seed=seed)
        res = hodge_decomposition(knm, phases)
        assert np.all(np.isfinite(res.gradient))
        assert np.all(np.isfinite(res.curl))
        assert np.all(np.isfinite(res.harmonic))

    @given(
        n=st.integers(min_value=2, max_value=10),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_length_n(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        knm = _asymmetric_knm(n, seed=seed)
        res = hodge_decomposition(knm, phases)
        assert len(res.gradient) == n
        assert len(res.curl) == n
        assert len(res.harmonic) == n

    def test_empty_phases(self) -> None:
        res = hodge_decomposition(np.zeros((0, 0)), np.array([]))
        assert len(res.gradient) == 0
        assert len(res.curl) == 0
        assert len(res.harmonic) == 0

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_knm_all_zero(self, n: int, seed: int) -> None:
        """K=0 → all components = 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        res = hodge_decomposition(np.zeros((n, n)), phases)
        np.testing.assert_allclose(res.gradient, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.curl, 0.0, atol=1e-15)
        np.testing.assert_allclose(res.harmonic, 0.0, atol=1e-15)
