# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for free energy and Langevin dynamics

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.ssgf.free_energy import (
    add_langevin_noise,
    boltzmann_weight,
    effective_temperature,
)


class TestAddLangevinNoise:
    def test_zero_temperature_no_change(self):
        z = np.array([1.0, 2.0, 3.0])
        result = add_langevin_noise(z, temperature=0.0, dt=0.01)
        np.testing.assert_array_equal(result, z)

    def test_zero_dt_no_change(self):
        z = np.array([1.0, 2.0])
        result = add_langevin_noise(z, temperature=1.0, dt=0.0)
        np.testing.assert_array_equal(result, z)

    def test_noise_has_correct_variance(self):
        """Statistical test: noise variance ≈ 2·T·dt for large samples."""
        rng = np.random.default_rng(42)
        z = np.zeros(100_000)
        T, dt = 2.0, 0.05
        result = add_langevin_noise(z, temperature=T, dt=dt, rng=rng)
        expected_var = 2.0 * T * dt
        actual_var = np.var(result)
        assert actual_var == pytest.approx(expected_var, rel=0.05)

    def test_reproducible_with_rng(self):
        z = np.array([0.0, 0.0, 0.0])
        r1 = add_langevin_noise(z, 1.0, 0.01, rng=np.random.default_rng(7))
        r2 = add_langevin_noise(z, 1.0, 0.01, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1, r2)

    def test_does_not_modify_input(self):
        z = np.array([1.0, 2.0])
        original = z.copy()
        add_langevin_noise(z, temperature=1.0, dt=0.01)
        np.testing.assert_array_equal(z, original)

    def test_negative_temperature_no_change(self):
        z = np.array([1.0, 2.0])
        result = add_langevin_noise(z, temperature=-1.0, dt=0.01)
        np.testing.assert_array_equal(result, z)


class TestBoltzmannWeight:
    def test_zero_energy_unity(self):
        assert boltzmann_weight(0.0, temperature=1.0) == pytest.approx(1.0)

    def test_positive_energy_less_than_one(self):
        w = boltzmann_weight(1.0, temperature=1.0)
        assert 0.0 < w < 1.0
        assert w == pytest.approx(np.exp(-1.0))

    def test_negative_energy_greater_than_one(self):
        w = boltzmann_weight(-1.0, temperature=1.0)
        assert w > 1.0

    def test_high_temperature_flattens(self):
        """At high T, all weights approach 1."""
        w = boltzmann_weight(10.0, temperature=1e6)
        assert w == pytest.approx(1.0, abs=1e-4)

    def test_zero_temperature_positive_energy(self):
        assert boltzmann_weight(1.0, temperature=0.0) == 0.0

    def test_zero_temperature_zero_energy(self):
        assert boltzmann_weight(0.0, temperature=0.0) == 1.0

    def test_zero_temperature_negative_energy(self):
        assert boltzmann_weight(-1.0, temperature=0.0) == 1.0

    def test_extreme_exponent_clamped(self):
        """Very large U/T doesn't cause overflow."""
        w = boltzmann_weight(1e10, temperature=1e-10)
        assert w == 0.0 or np.isfinite(w)


class TestEffectiveTemperature:
    def test_constant_cost_zero_temp(self):
        costs = np.array([5.0, 5.0, 5.0, 5.0])
        assert effective_temperature(costs) == pytest.approx(0.0)

    def test_positive_for_fluctuating_costs(self):
        rng = np.random.default_rng(42)
        costs = 10.0 + rng.standard_normal(1000)
        t_eff = effective_temperature(costs)
        assert t_eff > 0.0

    def test_higher_variance_higher_temperature(self):
        rng = np.random.default_rng(42)
        costs_low = 10.0 + 0.1 * rng.standard_normal(1000)
        rng2 = np.random.default_rng(42)
        costs_high = 10.0 + 10.0 * rng2.standard_normal(1000)
        assert effective_temperature(costs_high) > effective_temperature(costs_low)

    def test_single_value_returns_zero(self):
        assert effective_temperature(np.array([3.0])) == 0.0

    def test_empty_returns_zero(self):
        assert effective_temperature(np.array([])) == 0.0

    def test_zero_mean_returns_zero(self):
        costs = np.array([1.0, -1.0, 1.0, -1.0])
        assert effective_temperature(costs) == pytest.approx(0.0)
