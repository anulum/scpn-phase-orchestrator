# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based plasticity & stochastic proofs

"""Hypothesis-driven invariant proofs for Hebbian eligibility,
three-factor plasticity, and stochastic noise injection.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)
from scpn_phase_orchestrator.upde.stochastic import StochasticInjector

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Eligibility trace ────────────────────────────────────────────────


class TestEligibilityInvariants:
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_zero_diagonal(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        elig = compute_eligibility(phases)
        np.testing.assert_array_equal(np.diag(elig), 0.0)

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_symmetric(self, n: int, seed: int) -> None:
        """cos(θ_j - θ_i) = cos(θ_i - θ_j) → symmetric."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        elig = compute_eligibility(phases)
        np.testing.assert_allclose(elig, elig.T, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=50, deadline=None)
    def test_values_in_neg1_pos1(self, n: int, seed: int) -> None:
        """cos ∈ [-1, 1]."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        elig = compute_eligibility(phases)
        assert np.all(elig >= -1.0 - 1e-12)
        assert np.all(elig <= 1.0 + 1e-12)

    @given(n=st.integers(min_value=2, max_value=16))
    @settings(max_examples=20, deadline=None)
    def test_sync_all_ones(self, n: int) -> None:
        """Synchronized → cos(0) = 1 everywhere (except diagonal)."""
        phases = np.full(n, 2.0)
        elig = compute_eligibility(phases)
        expected = np.ones((n, n))
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_allclose(elig, expected, atol=1e-12)

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_shape(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        elig = compute_eligibility(phases)
        assert elig.shape == (n, n)


# ── 2. Three-factor plasticity ──────────────────────────────────────────


class TestThreeFactorUpdateInvariants:
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_gate_false_no_change(self, n: int, seed: int) -> None:
        """phase_gate=False → output equals input."""
        knm = _connected_knm(n, seed=seed)
        elig = compute_eligibility(np.random.default_rng(seed).uniform(0, TWO_PI, n))
        updated = three_factor_update(knm, elig, modulator=1.0, phase_gate=False)
        np.testing.assert_array_equal(updated, knm)

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_modulator_no_change(self, n: int, seed: int) -> None:
        """modulator=0 → no update."""
        knm = _connected_knm(n, seed=seed)
        elig = compute_eligibility(np.random.default_rng(seed).uniform(0, TWO_PI, n))
        updated = three_factor_update(knm, elig, modulator=0.0, phase_gate=True)
        np.testing.assert_allclose(updated, knm, atol=1e-15)

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_shape_preserved(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        elig = compute_eligibility(np.random.default_rng(seed).uniform(0, TWO_PI, n))
        updated = three_factor_update(
            knm, elig, modulator=1.0, phase_gate=True, lr=0.01
        )
        assert updated.shape == knm.shape

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_finite(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        elig = compute_eligibility(np.random.default_rng(seed).uniform(0, TWO_PI, n))
        updated = three_factor_update(
            knm, elig, modulator=2.0, phase_gate=True, lr=0.05
        )
        assert np.all(np.isfinite(updated))

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=20, deadline=None)
    def test_does_not_mutate_input(self, n: int, seed: int) -> None:
        knm = _connected_knm(n, seed=seed)
        knm_copy = knm.copy()
        elig = compute_eligibility(np.random.default_rng(seed).uniform(0, TWO_PI, n))
        three_factor_update(knm, elig, modulator=1.0, phase_gate=True, lr=0.1)
        np.testing.assert_array_equal(knm, knm_copy)


# ── 3. Stochastic injector ──────────────────────────────────────────────


class TestStochasticInjectorInvariants:
    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_output_in_zero_twopi(self, n: int, seed: int) -> None:
        """Output phases wrapped to [0, 2π)."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        inj = StochasticInjector(D=0.5, seed=seed)
        out = inj.inject(phases, dt=0.01)
        assert np.all(out >= 0.0)
        assert np.all(out < TWO_PI + 1e-10)

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_d_zero_unchanged(self, n: int, seed: int) -> None:
        """D=0 → no noise → phases unchanged."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        inj = StochasticInjector(D=0.0, seed=seed)
        out = inj.inject(phases, dt=0.01)
        np.testing.assert_array_equal(out, phases)

    def test_negative_d_raises(self) -> None:
        with pytest.raises(ValueError):
            StochasticInjector(D=-1.0)

    def test_set_negative_d_raises(self) -> None:
        inj = StochasticInjector(D=1.0)
        with pytest.raises(ValueError):
            inj.D = -0.5

    @given(
        n=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_shape_preserved(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        inj = StochasticInjector(D=1.0, seed=seed)
        out = inj.inject(phases, dt=0.01)
        assert out.shape == phases.shape

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_finite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, 10)
        inj = StochasticInjector(D=2.0, seed=seed)
        out = inj.inject(phases, dt=0.01)
        assert np.all(np.isfinite(out))

    def test_noise_magnitude_scales_with_d(self) -> None:
        """Higher D → larger perturbation (statistically)."""
        n = 500
        phases = np.ones(n) * np.pi
        inj_lo = StochasticInjector(D=0.01, seed=42)
        inj_hi = StochasticInjector(D=5.0, seed=42)
        out_lo = inj_lo.inject(phases.copy(), dt=0.1)
        out_hi = inj_hi.inject(phases.copy(), dt=0.1)
        # Circular variance: higher D → more spread
        spread_lo = np.var(out_lo)
        spread_hi = np.var(out_hi)
        assert spread_hi > spread_lo



# Pipeline wiring: hypothesis-driven plasticity + stochastic proofs exercise
# the three_factor_update + engine pipeline.
