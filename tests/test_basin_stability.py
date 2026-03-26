# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.basin_stability import (
    BasinStabilityResult,
    basin_stability,
    multi_basin_stability,
)


class TestBasinStability:
    def test_identical_frequencies_high_stability(self):
        """Identical omegas + strong coupling → S_B ≈ 1."""
        N = 6
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 2.0
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas, knm, n_samples=20, n_transient=200, n_measure=50
        )
        assert isinstance(result, BasinStabilityResult)
        assert result.S_B > 0.5

    def test_zero_coupling_low_stability(self):
        """Zero coupling + spread frequencies → S_B ≈ 0."""
        N = 6
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 2.0, N)
        knm = np.zeros((N, N))
        result = basin_stability(
            omegas, knm, n_samples=20, n_transient=200, n_measure=50
        )
        assert result.S_B < 0.5

    def test_result_fields(self):
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N))
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas, knm, n_samples=10, n_transient=100, n_measure=50
        )
        assert result.n_samples == 10
        assert len(result.R_final) == 10
        assert 0 <= result.S_B <= 1.0
        assert result.R_threshold == 0.8

    def test_custom_threshold(self):
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 3.0
        np.fill_diagonal(knm, 0)
        result = basin_stability(
            omegas,
            knm,
            n_samples=10,
            n_transient=200,
            n_measure=50,
            R_threshold=0.5,
        )
        assert result.R_threshold == 0.5


class TestMultiBasinStability:
    def test_returns_dict(self):
        N = 4
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 2.0
        np.fill_diagonal(knm, 0)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=10,
            n_transient=100,
            n_measure=50,
        )
        assert isinstance(results, dict)
        assert "R>=0.30" in results
        assert "R>=0.60" in results
        assert "R>=0.80" in results

    def test_monotonic_thresholds(self):
        """S_B at lower threshold >= S_B at higher threshold."""
        N = 6
        rng = np.random.default_rng(0)
        omegas = rng.normal(0, 0.5, N)
        knm = np.ones((N, N)) * 1.5
        np.fill_diagonal(knm, 0)
        results = multi_basin_stability(
            omegas,
            knm,
            n_samples=15,
            n_transient=200,
            n_measure=50,
        )
        assert results["R>=0.30"].S_B >= results["R>=0.80"].S_B
