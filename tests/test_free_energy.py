# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for free energy and Langevin dynamics

from __future__ import annotations

import math
import statistics
from typing import get_type_hints

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.ssgf import free_energy as free_energy_mod
from scpn_phase_orchestrator.ssgf.free_energy import (
    add_langevin_noise,
    boltzmann_weight,
    effective_temperature,
)
from tests.typing_contracts import assert_precise_ndarray_hint


class TestAddLangevinNoise:
    """Langevin noise: edge semantics, variance, reproducibility, input safety."""

    def test_zero_temperature_no_change(self) -> None:
        """Zero temperature returns an unchanged copy."""
        z = np.array([1.0, 2.0, 3.0])
        result = add_langevin_noise(z, temperature=0.0, dt=0.01)
        np.testing.assert_array_equal(result, z)

    def test_zero_dt_no_change(self) -> None:
        """A zero step returns an unchanged copy."""
        z = np.array([1.0, 2.0])
        result = add_langevin_noise(z, temperature=1.0, dt=0.0)
        np.testing.assert_array_equal(result, z)

    def test_noise_has_correct_variance(self) -> None:
        """Statistical test: noise variance ≈ 2·T·dt for large samples."""
        rng = np.random.default_rng(42)
        z = np.zeros(100_000)
        T, dt = 2.0, 0.05
        result = add_langevin_noise(z, temperature=T, dt=dt, rng=rng)
        expected_var = 2.0 * T * dt
        actual_var = np.var(result)
        assert actual_var == pytest.approx(expected_var, rel=0.05)

    def test_reproducible_with_rng(self) -> None:
        """The same generator seed gives the same noise."""
        z = np.array([0.0, 0.0, 0.0])
        r1 = add_langevin_noise(z, 1.0, 0.01, rng=np.random.default_rng(7))
        r2 = add_langevin_noise(z, 1.0, 0.01, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(r1, r2)

    def test_does_not_modify_input(self) -> None:
        """The input vector is left unchanged."""
        z = np.array([1.0, 2.0])
        original = z.copy()
        add_langevin_noise(z, temperature=1.0, dt=0.01)
        np.testing.assert_array_equal(z, original)

    def test_negative_temperature_no_change(self) -> None:
        """Negative temperature returns an unchanged copy."""
        z = np.array([1.0, 2.0])
        result = add_langevin_noise(z, temperature=-1.0, dt=0.01)
        np.testing.assert_array_equal(result, z)

    def test_rust_langevin_path_used_without_rng(self) -> None:
        """The compiled kernel keeps the shape, adds N(0, 2·T·dt) noise, reseeds."""
        if not free_energy_mod._HAS_RUST:
            pytest.skip("Rust backend not built")
        z = np.zeros((200, 150), dtype=np.float64)
        temperature, dt = 0.5, 0.25

        first = add_langevin_noise(z, temperature, dt, backend="rust")
        second = add_langevin_noise(z, temperature, dt, backend="rust")

        assert first.shape == z.shape
        assert first.dtype == np.float64
        # 30 000 draws: the sample variance is within 3 % of 2·T·dt with
        # overwhelming probability, and the mean within 0.02.
        assert float(np.var(first)) == pytest.approx(2 * temperature * dt, rel=0.03)
        assert abs(float(np.mean(first))) < 0.02
        assert not np.array_equal(first, second)
        assert not np.any(z)


class TestBoltzmannWeight:
    """Boltzmann weights: values, limits and the exponent clamp."""

    def test_zero_energy_unity(self) -> None:
        """Zero energy has unit weight."""
        assert boltzmann_weight(0.0, temperature=1.0) == pytest.approx(1.0)

    def test_positive_energy_less_than_one(self) -> None:
        """Positive energy weighs exp(-U/T) below one."""
        w = boltzmann_weight(1.0, temperature=1.0)
        assert 0.0 < w < 1.0
        assert w == pytest.approx(np.exp(-1.0))

    def test_negative_energy_greater_than_one(self) -> None:
        """Negative energy weighs above one."""
        w = boltzmann_weight(-1.0, temperature=1.0)
        assert w > 1.0

    def test_high_temperature_flattens(self) -> None:
        """At high T, all weights approach 1."""
        w = boltzmann_weight(10.0, temperature=1e6)
        assert w == pytest.approx(1.0, abs=1e-4)

    def test_zero_temperature_positive_energy(self) -> None:
        """At T = 0 positive energy has zero weight."""
        assert boltzmann_weight(1.0, temperature=0.0) == 0.0

    def test_zero_temperature_zero_energy(self) -> None:
        """At T = 0 zero energy has unit weight."""
        assert boltzmann_weight(0.0, temperature=0.0) == 1.0

    def test_zero_temperature_negative_energy(self) -> None:
        """At T = 0 negative energy has unit weight."""
        assert boltzmann_weight(-1.0, temperature=0.0) == 1.0

    def test_extreme_exponent_clamped(self) -> None:
        """Very large U/T doesn't cause overflow."""
        w = boltzmann_weight(1e10, temperature=1e-10)
        assert w == 0.0 or np.isfinite(w)

    def test_extreme_negative_exponent_clamped(self) -> None:
        """Very negative U/T is clamped to a finite maximum."""
        w = boltzmann_weight(-1e10, temperature=1e-10)
        assert np.isfinite(w)
        assert w > 1e100

    def test_rust_boltzmann_path_used(self) -> None:
        """The compiled kernel returns exp(-U/T), with the clamp and T = 0 law."""
        if not free_energy_mod._HAS_RUST:
            pytest.skip("Rust backend not built")
        for u_total, temperature in ((2.0, 8.0), (-3.0, 0.5), (0.0, 1.0)):
            got = boltzmann_weight(u_total, temperature, backend="rust")
            assert got == pytest.approx(math.exp(-u_total / temperature), rel=1e-15)
        assert boltzmann_weight(1.0, 0.0, backend="rust") == 0.0
        assert boltzmann_weight(0.0, 0.0, backend="rust") == 1.0
        assert boltzmann_weight(-1e10, 1e-10, backend="rust") == math.exp(700.0)


class TestEffectiveTemperature:
    """Effective temperature from cost fluctuations."""

    def test_constant_cost_zero_temp(self) -> None:
        """Constant costs give zero temperature."""
        costs = np.array([5.0, 5.0, 5.0, 5.0])
        assert effective_temperature(costs) == pytest.approx(0.0)

    def test_positive_for_fluctuating_costs(self) -> None:
        """Fluctuating costs give a positive temperature."""
        rng = np.random.default_rng(42)
        costs = 10.0 + rng.standard_normal(1000)
        t_eff = effective_temperature(costs)
        assert t_eff > 0.0

    def test_higher_variance_higher_temperature(self) -> None:
        """Larger fluctuations give a higher temperature."""
        rng = np.random.default_rng(42)
        costs_low = 10.0 + 0.1 * rng.standard_normal(1000)
        rng2 = np.random.default_rng(42)
        costs_high = 10.0 + 10.0 * rng2.standard_normal(1000)
        assert effective_temperature(costs_high) > effective_temperature(costs_low)

    def test_single_value_returns_zero(self) -> None:
        """A single cost gives zero temperature."""
        assert effective_temperature(np.array([3.0])) == 0.0

    def test_empty_returns_zero(self) -> None:
        """An empty history gives zero temperature."""
        assert effective_temperature(np.array([])) == 0.0

    def test_zero_mean_returns_zero(self) -> None:
        """A zero-mean history gives zero temperature."""
        costs = np.array([1.0, -1.0, 1.0, -1.0])
        assert effective_temperature(costs) == pytest.approx(0.0)

    def test_rust_effective_temperature_path_uses_flattened_costs(self) -> None:
        """The compiled kernel reads a 2-D history in row order."""
        if not free_energy_mod._HAS_RUST:
            pytest.skip("Rust backend not built")
        costs = np.array([[1.0, 2.0], [3.0, 7.0]], dtype=np.float64)
        flat = [1.0, 2.0, 3.0, 7.0]
        expected = statistics.variance(flat) / (2.0 * abs(statistics.fmean(flat)))

        got = effective_temperature(costs, backend="rust")

        assert got == pytest.approx(expected, rel=1e-14)


class TestFreeEnergyTypeHints:
    """Public array parameters carry precise float64 hints."""

    def test_public_array_contracts_are_parameterised(self) -> None:
        """Array parameters and returns are float64 ndarrays."""
        for hint in [
            get_type_hints(add_langevin_noise)["z"],
            get_type_hints(add_langevin_noise)["return"],
            get_type_hints(effective_temperature)["costs_history"],
        ]:
            assert_precise_ndarray_hint(hint)
            assert "float64" in str(hint)


class TestFreeEnergyPipelineWiring:
    """Pipeline: SSGF cost history → effective temperature → Langevin."""

    def test_ssgf_cost_history_to_temperature(self) -> None:
        """GeometryCarrier runs → cost history → effective_temperature
        measures thermodynamic fluctuations of the SSGF loop."""
        from scpn_phase_orchestrator.ssgf.carrier import GeometryCarrier

        gc = GeometryCarrier(4, z_dim=3, lr=0.05, seed=42)

        def cost_fn(W: NDArray[np.float64]) -> float:
            return float(np.sum((W - 0.5) ** 2))

        costs = []
        for _ in range(30):
            W = gc.decode()
            c = cost_fn(W)
            costs.append(c)
            gc.update(cost=c, cost_fn=cost_fn)

        t_eff = effective_temperature(np.array(costs))
        assert t_eff >= 0.0

    def test_langevin_noise_on_engine_phases(self) -> None:
        """add_langevin_noise on engine-produced phases."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(50):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))

        noisy = add_langevin_noise(phases, temperature=0.1, dt=0.01)
        assert noisy.shape == phases.shape
        assert not np.allclose(noisy, phases), "Noise must perturb"
