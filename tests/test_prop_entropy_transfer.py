# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based entropy & transfer entropy proofs

"""Hypothesis-driven invariant proofs for entropy production rate,
phase transfer entropy, TE matrix, and TE-adaptive coupling.

Each test is a computational theorem enforcing information-theoretic
bounds that must hold for all valid inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling
from scpn_phase_orchestrator.monitor.entropy_prod import entropy_production_rate
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    phase_transfer_entropy,
    transfer_entropy_matrix,
)

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Entropy production rate ──────────────────────────────────────────


class TestEntropyProductionRate:
    """Thermodynamic dissipation: Σ (dθ/dt)² · dt ≥ 0."""

    @given(
        n=st.integers(min_value=2, max_value=12),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=60, deadline=None)
    def test_nonnegative(self, n: int, seed: int) -> None:
        """Sum of squares times positive dt is always ≥ 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-3, 3, n)
        knm = _connected_knm(n, seed=seed)
        epr = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        assert epr >= -1e-15

    @given(n=st.integers(min_value=2, max_value=12))
    @settings(max_examples=30, deadline=None)
    def test_locked_phases_low(self, n: int) -> None:
        """Identical phases + identical frequencies → near-zero dissipation."""
        phases = np.full(n, 1.0)
        omegas = np.zeros(n)
        knm = _connected_knm(n)
        epr = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        assert epr < 1e-10

    def test_empty_phases_zero(self) -> None:
        epr = entropy_production_rate(
            np.array([]), np.array([]), np.zeros((0, 0)), 1.0, 0.01
        )
        assert epr == 0.0

    def test_zero_dt_zero(self) -> None:
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.array([1.0, 2.0, 3.0])
        knm = _connected_knm(3)
        epr = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.0)
        assert epr == 0.0

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_scales_with_dt(self, n: int, seed: int) -> None:
        """EPR scales linearly with dt."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-2, 2, n)
        knm = _connected_knm(n, seed=seed)
        epr1 = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        epr2 = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.02)
        if epr1 > 1e-15:
            assert abs(epr2 / epr1 - 2.0) < 1e-10

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-5, 5, n)
        knm = _connected_knm(n, seed=seed)
        epr = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        assert np.isfinite(epr)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_zero_coupling_equals_omega_squared(self, seed: int) -> None:
        """With K=0 and α=anything, dθ/dt = ω, so EPR = Σω²·dt."""
        n = 4
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = rng.uniform(-3, 3, n)
        knm = np.zeros((n, n))
        dt = 0.01
        epr = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=dt)
        expected = float(np.sum(omegas**2)) * dt
        assert abs(epr - expected) < 1e-12


# ── 2. Phase transfer entropy ───────────────────────────────────────────


class TestPhaseTransferEntropy:
    """TE(X→Y) ≥ 0, information-theoretic bound."""

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=40, deadline=None)
    def test_nonnegative(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        source = rng.uniform(0, TWO_PI, 100)
        target = rng.uniform(0, TWO_PI, 100)
        te = phase_transfer_entropy(source, target)
        assert te >= -1e-15

    def test_short_series_zero(self) -> None:
        """Series with < 3 points → 0."""
        assert phase_transfer_entropy(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
        assert phase_transfer_entropy(np.array([1.0]), np.array([1.0])) == 0.0

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_self_te_low(self, seed: int) -> None:
        """TE(X→X) should be low — past of X is redundant given X."""
        rng = np.random.default_rng(seed)
        signal = rng.uniform(0, TWO_PI, 200)
        te = phase_transfer_entropy(signal, signal)
        assert te < 0.5

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_finite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        source = rng.uniform(0, TWO_PI, 80)
        target = rng.uniform(0, TWO_PI, 80)
        te = phase_transfer_entropy(source, target)
        assert np.isfinite(te)

    @pytest.mark.parametrize("n_bins", [4, 8, 16, 32])
    def test_nonneg_various_bins(self, n_bins: int) -> None:
        rng = np.random.default_rng(42)
        source = rng.uniform(0, TWO_PI, 100)
        target = rng.uniform(0, TWO_PI, 100)
        te = phase_transfer_entropy(source, target, n_bins=n_bins)
        assert te >= -1e-15


# ── 3. Transfer entropy matrix ──────────────────────────────────────────


class TestTEMatrix:
    """TE matrix: shape, diagonal, non-negativity."""

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_shape(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (n, 100))
        te = transfer_entropy_matrix(traj)
        assert te.shape == (n, n)

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_diagonal_zero(self, n: int, seed: int) -> None:
        """TE matrix diagonal = 0 (no self-transfer computed)."""
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (n, 100))
        te = transfer_entropy_matrix(traj)
        np.testing.assert_array_equal(np.diag(te), 0.0)

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_nonnegative(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (n, 100))
        te = transfer_entropy_matrix(traj)
        assert np.all(te >= -1e-15)

    @given(
        n=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_all_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        traj = rng.uniform(0, TWO_PI, (n, 80))
        te = transfer_entropy_matrix(traj)
        assert np.all(np.isfinite(te))


# ── 4. TE adaptive coupling ─────────────────────────────────────────────


class TestTEAdaptCoupling:
    """te_adapt_coupling preserves structural invariants."""

    @given(
        n=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_zero_diagonal(self, n: int, seed: int) -> None:
        """Output K_nm must have zero diagonal (no self-coupling)."""
        rng = np.random.default_rng(seed)
        knm = _connected_knm(n, seed=seed)
        history = rng.uniform(0, TWO_PI, (n, 80))
        updated = te_adapt_coupling(knm, history, lr=0.1)
        np.testing.assert_array_equal(np.diag(updated), 0.0)

    @given(
        n=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_nonnegative(self, n: int, seed: int) -> None:
        """Output K_nm ≥ 0 (clipped by np.maximum)."""
        rng = np.random.default_rng(seed)
        knm = _connected_knm(n, seed=seed)
        history = rng.uniform(0, TWO_PI, (n, 80))
        updated = te_adapt_coupling(knm, history, lr=0.1)
        assert np.all(updated >= -1e-15)

    @given(
        n=st.integers(min_value=2, max_value=5),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_shape_preserved(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        knm = _connected_knm(n, seed=seed)
        history = rng.uniform(0, TWO_PI, (n, 80))
        updated = te_adapt_coupling(knm, history, lr=0.1)
        assert updated.shape == knm.shape

    def test_zero_lr_no_change(self) -> None:
        """lr=0 + decay=0 → output equals input (clamped to ≥0)."""
        n = 4
        knm = _connected_knm(n)
        rng = np.random.default_rng(0)
        history = rng.uniform(0, TWO_PI, (n, 80))
        updated = te_adapt_coupling(knm, history, lr=0.0, decay=0.0)
        np.testing.assert_allclose(updated, knm, atol=1e-12)

    def test_full_decay_zeros_knm(self) -> None:
        """decay=1.0, lr=0 → output ≈ 0 (old K fully decayed, no TE added)."""
        n = 3
        knm = _connected_knm(n, strength=5.0)
        rng = np.random.default_rng(0)
        history = rng.uniform(0, TWO_PI, (n, 80))
        updated = te_adapt_coupling(knm, history, lr=0.0, decay=1.0)
        assert np.all(updated < 1e-12)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_finite(self, seed: int) -> None:
        n = 3
        rng = np.random.default_rng(seed)
        knm = _connected_knm(n, seed=seed)
        history = rng.uniform(0, TWO_PI, (n, 60))
        updated = te_adapt_coupling(knm, history, lr=0.05, decay=0.01)
        assert np.all(np.isfinite(updated))



# Pipeline wiring: hypothesis-driven invariant proofs exercise the pipeline.
