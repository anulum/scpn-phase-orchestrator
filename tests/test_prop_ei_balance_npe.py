# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based EI balance & NPE proofs

"""Hypothesis-driven invariant proofs for E/I balance and
Normalized Persistent Entropy (NPE).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.ei_balance import (
    adjust_ei_ratio,
    compute_ei_balance,
)
from scpn_phase_orchestrator.monitor.npe import compute_npe, phase_distance_matrix

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Phase distance matrix ────────────────────────────────────────────


class TestPhaseDistanceMatrix:
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_symmetric(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        D = phase_distance_matrix(phases)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_diagonal_zero(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        D = phase_distance_matrix(phases)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_values_in_zero_pi(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        D = phase_distance_matrix(phases)
        assert np.all(D >= -1e-12)
        assert np.all(np.pi + 1e-12 >= D)


# ── 2. NPE ──────────────────────────────────────────────────────────────


class TestNPEInvariants:
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_npe_in_unit(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        npe = compute_npe(phases)
        assert -1e-12 <= npe <= 1.0 + 1e-12

    @given(n=st.integers(min_value=2, max_value=16))
    @settings(max_examples=30, deadline=None)
    def test_sync_near_zero(self, n: int) -> None:
        phases = np.full(n, 1.5)
        npe = compute_npe(phases)
        assert npe < 0.01

    def test_n1_returns_zero(self) -> None:
        assert compute_npe(np.array([1.0])) == 0.0

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=30, deadline=None)
    def test_random_high(self, seed: int) -> None:
        """Random phases (large N) → NPE close to 1."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, 50)
        npe = compute_npe(phases)
        assert npe > 0.5

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        assert np.isfinite(compute_npe(phases))


# ── 3. E/I balance ──────────────────────────────────────────────────────


class TestEIBalanceInvariants:
    @given(
        n=st.integers(min_value=4, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_ratio_nonneg(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        e_idx = list(range(n // 2))
        i_idx = list(range(n // 2, n))
        bal = compute_ei_balance(knm, e_idx, i_idx)
        assert bal.ratio >= 0.0

    @given(
        n=st.integers(min_value=4, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_strengths_nonneg(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        e_idx = list(range(n // 2))
        i_idx = list(range(n // 2, n))
        bal = compute_ei_balance(knm, e_idx, i_idx)
        assert bal.excitatory_strength >= 0.0
        assert bal.inhibitory_strength >= 0.0

    @given(n=st.integers(min_value=4, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_balanced_flag(self, n: int) -> None:
        """Uniform K → ratio ≈ 1 → is_balanced = True."""
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        e_idx = list(range(n // 2))
        i_idx = list(range(n // 2, n))
        bal = compute_ei_balance(knm, e_idx, i_idx)
        assert abs(bal.ratio - 1.0) < 1e-10
        assert bal.is_balanced is True

    def test_empty_inhibitory_inf(self) -> None:
        knm = _connected_knm(4)
        bal = compute_ei_balance(knm, [0, 1, 2, 3], [])
        assert bal.ratio == float("inf") or bal.ratio == 1.0

    @given(
        n=st.integers(min_value=4, max_value=8),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_adjust_preserves_shape(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        e_idx = list(range(n // 2))
        i_idx = list(range(n // 2, n))
        adjusted = adjust_ei_ratio(knm, e_idx, i_idx, target_ratio=1.0)
        assert adjusted.shape == knm.shape


# Pipeline wiring: hypothesis-driven invariant proofs exercise the pipeline.
