# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based free energy & SSGF cost proofs

"""Hypothesis-driven invariant proofs for Boltzmann weights,
effective temperature, Langevin noise, and SSGF cost terms.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs
from scpn_phase_orchestrator.ssgf.free_energy import (
    add_langevin_noise,
    boltzmann_weight,
    effective_temperature,
)

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. Boltzmann weight ─────────────────────────────────────────────────


class TestBoltzmannWeightInvariants:
    @given(
        u=st.floats(min_value=0, max_value=100),
        t=st.floats(min_value=0.01, max_value=100),
    )
    @settings(max_examples=60, deadline=None)
    def test_in_zero_one_for_nonneg_u(self, u: float, t: float) -> None:
        """exp(-U/T) ∈ (0, 1] for U ≥ 0, T > 0."""
        w = boltzmann_weight(u, t)
        assert 0.0 < w <= 1.0 + 1e-12

    @given(t=st.floats(min_value=0.01, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_zero_energy_is_one(self, t: float) -> None:
        """U = 0 → w = exp(0) = 1."""
        assert abs(boltzmann_weight(0.0, t) - 1.0) < 1e-12

    @given(u=st.floats(min_value=0.01, max_value=50))
    @settings(max_examples=30, deadline=None)
    def test_monotonic_decreasing_in_u(self, u: float) -> None:
        """Higher U → lower Boltzmann weight (at fixed T)."""
        t = 1.0
        w_lo = boltzmann_weight(u, t)
        w_hi = boltzmann_weight(u + 1.0, t)
        assert w_hi <= w_lo + 1e-12

    @given(u=st.floats(min_value=0.1, max_value=10))
    @settings(max_examples=30, deadline=None)
    def test_monotonic_increasing_in_t(self, u: float) -> None:
        """Higher T → higher weight (less penalisation)."""
        w_lo = boltzmann_weight(u, 0.5)
        w_hi = boltzmann_weight(u, 5.0)
        assert w_hi >= w_lo - 1e-12

    def test_zero_temp_u_positive_gives_zero(self) -> None:
        assert boltzmann_weight(1.0, 0.0) == 0.0

    def test_zero_temp_u_zero_gives_one(self) -> None:
        assert boltzmann_weight(0.0, 0.0) == 1.0

    def test_zero_temp_u_negative_gives_one(self) -> None:
        assert boltzmann_weight(-1.0, 0.0) == 1.0

    @given(
        u=st.floats(min_value=-100, max_value=100),
        t=st.floats(min_value=0.01, max_value=100),
    )
    @settings(max_examples=40, deadline=None)
    def test_finite(self, u: float, t: float) -> None:
        w = boltzmann_weight(u, t)
        assert np.isfinite(w)


# ── 2. Effective temperature ─────────────────────────────────────────────


class TestEffectiveTemperature:
    def test_constant_cost_zero(self) -> None:
        """No fluctuations → T_eff ≈ 0."""
        costs = np.full(50, 3.14)
        assert effective_temperature(costs) < 1e-20

    def test_short_series_zero(self) -> None:
        assert effective_temperature(np.array([1.0])) == 0.0
        assert effective_temperature(np.array([])) == 0.0

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=40, deadline=None)
    def test_nonnegative(self, seed: int) -> None:
        """Var ≥ 0 and |mean| ≥ 0, so T_eff ≥ 0."""
        rng = np.random.default_rng(seed)
        costs = rng.uniform(0.1, 10, 50)
        t_eff = effective_temperature(costs)
        assert t_eff >= -1e-12

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=30, deadline=None)
    def test_finite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        costs = rng.uniform(0.1, 5, 30)
        assert np.isfinite(effective_temperature(costs))

    def test_higher_variance_higher_teff(self) -> None:
        """More fluctuation → higher T_eff."""
        rng = np.random.default_rng(42)
        base = 5.0 + rng.standard_normal(100) * 0.1
        noisy = 5.0 + rng.standard_normal(100) * 2.0
        t_calm = effective_temperature(base)
        t_noisy = effective_temperature(noisy)
        assert t_noisy > t_calm


# ── 3. Langevin noise ───────────────────────────────────────────────────


class TestLangevinNoise:
    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=30, deadline=None)
    def test_shape_preserved(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(10)
        z_new = add_langevin_noise(z, temperature=1.0, dt=0.01, rng=rng)
        assert z_new.shape == z.shape

    def test_zero_temp_no_change(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        z_new = add_langevin_noise(z, temperature=0.0, dt=0.01)
        np.testing.assert_array_equal(z_new, z)

    def test_zero_dt_no_change(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        z_new = add_langevin_noise(z, temperature=1.0, dt=0.0)
        np.testing.assert_array_equal(z_new, z)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_noise_has_correct_scale(self, seed: int) -> None:
        """σ = sqrt(2·T·dt). Over many samples, std ≈ σ."""
        T, dt = 2.0, 0.05
        expected_sigma = np.sqrt(2.0 * T * dt)
        rng = np.random.default_rng(seed)
        z = np.zeros(1000)
        deltas = add_langevin_noise(z, temperature=T, dt=dt, rng=rng) - z
        measured_std = np.std(deltas)
        assert abs(measured_std - expected_sigma) < 0.15 * expected_sigma

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=30, deadline=None)
    def test_finite(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(20)
        z_new = add_langevin_noise(z, temperature=1.0, dt=0.01, rng=rng)
        assert np.all(np.isfinite(z_new))


# ── 4. SSGF costs ───────────────────────────────────────────────────────


class TestSSGFCostInvariants:
    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_c1_sync_in_unit(self, n: int, seed: int) -> None:
        """c1 = 1 - R ∈ [0, 1] since R ∈ [0, 1]."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        W = _connected_knm(n, seed=seed)
        costs = compute_ssgf_costs(W, phases)
        assert -1e-12 <= costs.c1_sync <= 1.0 + 1e-12

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_c3_sparsity_nonneg(self, n: int, seed: int) -> None:
        """||W||₁ / N² ≥ 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        W = _connected_knm(n, seed=seed)
        costs = compute_ssgf_costs(W, phases)
        assert costs.c3_sparsity >= -1e-12

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_c4_symmetry_zero_for_symmetric(self, n: int, seed: int) -> None:
        """Symmetric W → c4 = 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        W = _connected_knm(n, seed=seed)
        costs = compute_ssgf_costs(W, phases)
        assert abs(costs.c4_symmetry) < 1e-12

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=40, deadline=None)
    def test_c4_symmetry_nonneg(self, n: int, seed: int) -> None:
        """Frobenius norm ≥ 0."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        W = rng.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0.0)
        costs = compute_ssgf_costs(W, phases)
        assert costs.c4_symmetry >= -1e-12

    @given(n=st.integers(min_value=2, max_value=8))
    @settings(max_examples=20, deadline=None)
    def test_sync_phases_c1_zero(self, n: int) -> None:
        """Synchronized phases → R = 1 → c1 = 0."""
        phases = np.full(n, 2.0)
        W = _connected_knm(n)
        costs = compute_ssgf_costs(W, phases)
        assert abs(costs.c1_sync) < 1e-10

    @given(
        n=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(max_examples=30, deadline=None)
    def test_all_finite(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, TWO_PI, n)
        W = _connected_knm(n, seed=seed)
        costs = compute_ssgf_costs(W, phases)
        assert np.isfinite(costs.c1_sync)
        assert np.isfinite(costs.c2_spectral_gap)
        assert np.isfinite(costs.c3_sparsity)
        assert np.isfinite(costs.c4_symmetry)
        assert np.isfinite(costs.u_total)

    def test_zero_weight_all_terms_zero_total(self) -> None:
        """All weights = 0 → u_total = 0."""
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        W = _connected_knm(4)
        costs = compute_ssgf_costs(W, phases, weights=(0.0, 0.0, 0.0, 0.0))
        assert abs(costs.u_total) < 1e-15
