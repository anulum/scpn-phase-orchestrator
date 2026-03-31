# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Property-based basin stability proofs

"""Hypothesis-driven invariant proofs for basin stability estimation.

Each test is a computational theorem proving mathematical bounds on
basin stability S_B and multi-basin analysis. Menck et al. 2013.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
)

TWO_PI = 2.0 * np.pi


def _connected_knm(n: int, strength: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, (n, n)) * strength
    knm = 0.5 * (raw + raw.T)
    np.fill_diagonal(knm, 0.0)
    return knm


# ── 1. S_B ∈ [0, 1] — universal bound ──────────────────────────────────


class TestBasinStabilityBounds:
    """S_B is a probability, so must lie in [0, 1] for all inputs."""

    @given(
        n=st.integers(min_value=2, max_value=6),
        strength=st.floats(min_value=0.0, max_value=10.0),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=40, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_sb_in_unit_interval(self, n: int, strength: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, strength=strength, seed=seed)
        result = basin_stability(
            omegas,
            knm,
            n_samples=20,
            n_transient=100,
            n_measure=50,
            seed=seed,
        )
        assert 0.0 <= result.S_B <= 1.0

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_n_converged_leq_n_samples(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        result = basin_stability(
            omegas,
            knm,
            n_samples=20,
            n_transient=100,
            n_measure=50,
            seed=seed,
        )
        assert result.n_converged <= result.n_samples

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_sb_equals_ratio(self, n: int, seed: int) -> None:
        """S_B must equal n_converged / n_samples exactly."""
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        result = basin_stability(
            omegas,
            knm,
            n_samples=20,
            n_transient=100,
            n_measure=50,
            seed=seed,
        )
        assert abs(result.S_B - result.n_converged / result.n_samples) < 1e-15

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_r_final_shape(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        n_samp = 15
        result = basin_stability(
            omegas,
            knm,
            n_samples=n_samp,
            n_transient=100,
            n_measure=50,
            seed=seed,
        )
        assert result.R_final.shape == (n_samp,)

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_r_final_bounded(self, n: int, seed: int) -> None:
        """R ∈ [0, 1] for all trials."""
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        result = basin_stability(
            omegas,
            knm,
            n_samples=20,
            n_transient=100,
            n_measure=50,
            seed=seed,
        )
        assert np.all(result.R_final >= -1e-12)
        assert np.all(result.R_final <= 1.0 + 1e-12)


# ── 2. Physical limits ──────────────────────────────────────────────────


class TestBasinStabilityPhysics:
    """Physics-grounded invariants: strong coupling → sync, zero → desync."""

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_strong_coupling_high_sb(self, n: int) -> None:
        """Identical frequencies + strong K → S_B ≈ 1."""
        omegas = np.zeros(n)
        knm = _connected_knm(n, strength=20.0)
        result = basin_stability(
            omegas,
            knm,
            n_samples=30,
            n_transient=300,
            n_measure=100,
        )
        assert result.S_B > 0.8

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_zero_coupling_low_sb(self, n: int) -> None:
        """Zero coupling + spread frequencies → low S_B."""
        omegas = np.linspace(-3, 3, n)
        knm = np.zeros((n, n))
        result = basin_stability(
            omegas,
            knm,
            n_samples=30,
            n_transient=200,
            n_measure=100,
        )
        assert result.S_B < 0.5

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_identical_freqs_high_sb(self, seed: int) -> None:
        """Identical frequencies always sync (any connected K)."""
        n = 4
        omegas = np.full(n, 2.0)
        knm = _connected_knm(n, strength=5.0, seed=seed)
        result = basin_stability(
            omegas,
            knm,
            n_samples=20,
            n_transient=200,
            n_measure=100,
            seed=seed,
        )
        assert result.S_B > 0.6

    def test_deterministic_seed(self) -> None:
        """Same seed → identical results."""
        n = 4
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        knm = _connected_knm(n, strength=3.0)
        r1 = basin_stability(
            omegas, knm, n_samples=20, n_transient=100, n_measure=50, seed=99
        )
        r2 = basin_stability(
            omegas, knm, n_samples=20, n_transient=100, n_measure=50, seed=99
        )
        np.testing.assert_array_equal(r1.R_final, r2.R_final)
        assert r1.S_B == r2.S_B


# ── 3. Multi-basin: threshold monotonicity ──────────────────────────────


class TestMultiBasinMonotonicity:
    """S_B(lower threshold) ≥ S_B(higher threshold). More trials pass a
    lower bar than a higher one.
    """

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_monotonic_thresholds(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, strength=3.0, seed=seed)
        thresholds = (0.2, 0.4, 0.6, 0.8, 0.95)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=20,
            n_transient=100,
            n_measure=50,
            R_thresholds=thresholds,
            seed=seed,
        )
        sb_values = [results[f"R>={t:.2f}"].S_B for t in thresholds]
        for i in range(len(sb_values) - 1):
            assert sb_values[i] >= sb_values[i + 1] - 1e-12

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_multi_returns_all_thresholds(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        thresholds = (0.3, 0.6, 0.9)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=15,
            n_transient=80,
            n_measure=40,
            R_thresholds=thresholds,
            seed=seed,
        )
        for t in thresholds:
            key = f"R>={t:.2f}"
            assert key in results
            assert isinstance(results[key], BasinStabilityResult)

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_multi_all_sb_bounded(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=15,
            n_transient=80,
            n_measure=40,
            seed=seed,
        )
        for result in results.values():
            assert 0.0 <= result.S_B <= 1.0
            assert result.n_converged <= result.n_samples

    @given(
        n=st.integers(min_value=2, max_value=6),
        seed=st.integers(min_value=0, max_value=200),
    )
    @settings(
        max_examples=30, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_multi_shared_r_final(self, n: int, seed: int) -> None:
        """All thresholds see the same R_final (same simulation runs)."""
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=15,
            n_transient=80,
            n_measure=40,
            seed=seed,
        )
        r_arrays = [r.R_final for r in results.values()]
        for r in r_arrays[1:]:
            np.testing.assert_array_equal(r, r_arrays[0])


# ── 4. Custom threshold behaviour ───────────────────────────────────────


class TestCustomThreshold:
    """R_threshold stored correctly and affects classification."""

    @pytest.mark.parametrize("thresh", [0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
    def test_threshold_stored(self, thresh: float) -> None:
        n = 3
        omegas = np.ones(n)
        knm = _connected_knm(n)
        result = basin_stability(
            omegas,
            knm,
            n_samples=10,
            n_transient=50,
            n_measure=30,
            R_threshold=thresh,
        )
        assert result.R_threshold == thresh

    def test_lower_threshold_more_converged(self) -> None:
        """Lower R_threshold → more trials classified as converged."""
        n = 4
        omegas = np.array([1.0, 1.2, 1.5, 2.0])
        knm = _connected_knm(n, strength=3.0)
        r_low = basin_stability(
            omegas,
            knm,
            n_samples=30,
            n_transient=200,
            n_measure=100,
            R_threshold=0.3,
        )
        r_high = basin_stability(
            omegas,
            knm,
            n_samples=30,
            n_transient=200,
            n_measure=100,
            R_threshold=0.9,
        )
        assert r_low.n_converged >= r_high.n_converged


# ── 5. Alpha (phase lag) effects ─────────────────────────────────────────


class TestPhaseLagEffects:
    """Phase lag α affects synchronisability."""

    def test_zero_alpha_default(self) -> None:
        """alpha=None should behave identically to alpha=zeros."""
        n = 4
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        knm = _connected_knm(n, strength=5.0)
        r_none = basin_stability(
            omegas,
            knm,
            alpha=None,
            n_samples=20,
            n_transient=100,
            n_measure=50,
            seed=42,
        )
        r_zero = basin_stability(
            omegas,
            knm,
            alpha=np.zeros((n, n)),
            n_samples=20,
            n_transient=100,
            n_measure=50,
            seed=42,
        )
        np.testing.assert_array_equal(r_none.R_final, r_zero.R_final)

    @given(seed=st.integers(min_value=0, max_value=100))
    @settings(
        max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None
    )
    def test_alpha_results_bounded(self, seed: int) -> None:
        """With arbitrary alpha, S_B still ∈ [0, 1]."""
        n = 3
        rng = np.random.default_rng(seed)
        omegas = rng.uniform(-1, 1, n)
        knm = _connected_knm(n, seed=seed)
        alpha = rng.uniform(-np.pi, np.pi, (n, n))
        result = basin_stability(
            omegas,
            knm,
            alpha=alpha,
            n_samples=15,
            n_transient=80,
            n_measure=40,
            seed=seed,
        )
        assert 0.0 <= result.S_B <= 1.0


# Pipeline wiring: property-based tests use UPDEEngine + basin_stability
# internally — hypothesis-driven invariant proofs exercise the pipeline.
