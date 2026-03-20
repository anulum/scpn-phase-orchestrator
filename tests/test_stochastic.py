# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stochastic noise tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.stochastic import (
    NoiseProfile,
    StochasticInjector,
    _self_consistency_R,
    find_optimal_noise,
    optimal_D,
)


class TestStochasticInjector:
    def test_zero_noise_identity(self):
        inj = StochasticInjector(D=0.0)
        phases = np.array([0.0, 1.0, 2.0])
        result = inj.inject(phases, dt=0.01)
        np.testing.assert_array_equal(result, phases)

    def test_nonzero_noise_changes_phases(self):
        inj = StochasticInjector(D=1.0, seed=42)
        phases = np.zeros(10)
        result = inj.inject(phases, dt=0.01)
        assert not np.allclose(result, phases)

    def test_output_in_range(self):
        inj = StochasticInjector(D=5.0, seed=42)
        phases = np.zeros(100)
        result = inj.inject(phases, dt=0.1)
        assert np.all(result >= 0.0)
        assert np.all(result < 2 * np.pi)

    def test_negative_D_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            StochasticInjector(D=-1.0)

    def test_D_setter(self):
        inj = StochasticInjector(D=0.0)
        inj.D = 0.5
        assert inj.D == 0.5

    def test_D_setter_negative_raises(self):
        inj = StochasticInjector(D=0.0)
        with pytest.raises(ValueError):
            inj.D = -1.0

    def test_reproducible_with_seed(self):
        inj1 = StochasticInjector(D=1.0, seed=99)
        inj2 = StochasticInjector(D=1.0, seed=99)
        phases = np.zeros(5)
        r1 = inj1.inject(phases, dt=0.01)
        r2 = inj2.inject(phases, dt=0.01)
        np.testing.assert_array_equal(r1, r2)


class TestSelfConsistency:
    def test_zero_K(self):
        assert _self_consistency_R(0.0, 1.0) == 0.0

    def test_zero_D(self):
        assert _self_consistency_R(1.0, 0.0) == 1.0

    def test_intermediate(self):
        R = _self_consistency_R(2.0, 0.5)
        assert 0.0 < R < 1.0

    def test_large_K(self):
        R = _self_consistency_R(100.0, 1.0)
        assert R > 0.9


class TestOptimalD:
    def test_formula(self):
        assert abs(optimal_D(2.0, 0.8) - 0.8) < 1e-10

    def test_zero_R(self):
        assert optimal_D(1.0, 0.0) == 0.0


class TestFindOptimalNoise:
    def test_returns_profile(self):
        n = 6
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = find_optimal_noise(
            engine, phases, omegas, knm, alpha,
            D_range=np.array([0.0, 0.1, 0.5]),
            n_steps=50,
        )
        assert isinstance(result, NoiseProfile)
        assert result.D >= 0.0
        assert 0.0 <= result.R_achieved <= 1.0
        assert 0.0 <= result.R_deterministic <= 1.0
