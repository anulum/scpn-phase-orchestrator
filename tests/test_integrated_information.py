# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for integrated-information monitor

from __future__ import annotations

from typing import Any, get_type_hints

import numpy as np
import pytest

import scpn_phase_orchestrator.monitor as monitor
from scpn_phase_orchestrator.monitor.information_integration import (
    IntegratedInformationBenchmarkReport,
    IntegratedInformationResult,
    benchmark_integrated_information_approximations,
    integrated_information,
)

TWO_PI = 2.0 * np.pi


class TestIntegratedInformationContracts:
    def test_public_array_contract_is_parameterised(self) -> None:
        hints = get_type_hints(integrated_information)

        assert "numpy.ndarray" in str(hints["phase_series"])
        assert "float64" in str(hints["phase_series"])
        assert hints["return"] is IntegratedInformationResult
        assert callable(monitor.integrated_information)
        assert callable(monitor.benchmark_integrated_information_approximations)
        assert "integrated_information" in dir(monitor)
        assert "IntegratedInformationResult" in dir(monitor)
        assert "IntegratedInformationBenchmarkReport" in dir(monitor)

    def test_invalid_shape_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            integrated_information(np.array([0.0, 1.0]))

    def test_single_oscillator_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two oscillators"):
            integrated_information(np.zeros((1, 8), dtype=np.float64))

    def test_single_sample_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least two samples"):
            integrated_information(np.zeros((2, 1), dtype=np.float64))

    def test_non_finite_values_are_rejected(self) -> None:
        phases = np.zeros((2, 8), dtype=np.float64)
        phases[0, 3] = np.nan

        with pytest.raises(ValueError, match="finite"):
            integrated_information(phases)

    def test_boolean_phase_series_is_rejected(self) -> None:
        phases = np.array([[True, False], [False, True]])

        with pytest.raises(ValueError, match="phase_series"):
            integrated_information(phases)

    @pytest.mark.parametrize("n_bins", [False, 1, 1.5])
    def test_invalid_bin_count_is_rejected(self, n_bins: Any) -> None:
        phases = np.zeros((2, 8), dtype=np.float64)

        with pytest.raises(ValueError, match="n_bins"):
            integrated_information(phases, n_bins=n_bins)


class TestIntegratedInformationBehaviour:
    def test_independent_phase_streams_are_finite_and_bounded(self) -> None:
        rng = np.random.default_rng(7)
        phases = rng.uniform(0.0, TWO_PI, size=(5, 256)).astype(np.float64)

        result = integrated_information(phases, n_bins=8)

        assert result.phi >= 0.0
        assert 0.0 <= result.normalised_phi <= 1.0
        assert result.total_integration >= 0.0
        assert result.pairwise_mi.shape == (5, 5)
        assert np.allclose(result.pairwise_mi, result.pairwise_mi.T)
        assert np.allclose(np.diag(result.pairwise_mi), 0.0)

    def test_locked_streams_raise_phi_above_independent_baseline(self) -> None:
        rng = np.random.default_rng(11)
        independent = rng.uniform(0.0, TWO_PI, size=(4, 256)).astype(np.float64)
        base = np.linspace(0.0, 6.0 * TWO_PI, 256, dtype=np.float64) % TWO_PI
        locked = np.vstack(
            [
                base,
                (base + 0.02) % TWO_PI,
                (base + 0.04) % TWO_PI,
                (base + 0.06) % TWO_PI,
            ]
        ).astype(np.float64)

        independent_result = integrated_information(independent, n_bins=8)
        locked_result = integrated_information(locked, n_bins=8)

        assert locked_result.phi > independent_result.phi
        assert locked_result.normalised_phi > independent_result.normalised_phi

    def test_minimum_partition_finds_independent_bridge(self) -> None:
        rng = np.random.default_rng(13)
        base_a = np.linspace(0.0, 4.0 * TWO_PI, 320, dtype=np.float64) % TWO_PI
        base_b = rng.uniform(0.0, TWO_PI, size=320).astype(np.float64)
        phases = np.vstack(
            [
                base_a,
                (base_a + 0.01) % TWO_PI,
                base_b,
                (base_b + 0.01) % TWO_PI,
            ]
        ).astype(np.float64)

        result = integrated_information(phases, n_bins=10)
        left, right = result.minimum_partition

        assert set(left).isdisjoint(right)
        assert set(left) | set(right) == {0, 1, 2, 3}
        assert result.phi <= result.total_integration

    def test_audit_record_is_json_ready(self) -> None:
        phases = np.tile(
            np.linspace(0.0, TWO_PI, 128, dtype=np.float64),
            (3, 1),
        )

        record = integrated_information(phases, n_bins=6).to_audit_record()

        assert record["monitor"] == "integrated_information"
        assert record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
        assert isinstance(record["minimum_partition"][0], list)
        assert isinstance(record["pairwise_mi"], list)
        assert record["n_bins"] == 6


class TestIntegratedInformationPipelineWiring:
    def test_engine_trajectory_to_integrated_information(self) -> None:
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 6
        engine = UPDEEngine(n_oscillators=n, dt=0.01, method="euler")
        rng = np.random.default_rng(23)
        phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
        omegas = np.linspace(0.9, 1.1, n, dtype=np.float64)
        knm = np.full((n, n), 0.08, dtype=np.float64)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n), dtype=np.float64)
        trajectory: list[np.ndarray] = []

        for _ in range(160):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())

        phase_series = np.asarray(trajectory, dtype=np.float64).T
        result = integrated_information(phase_series, n_bins=8)

        assert result.pairwise_mi.shape == (n, n)
        assert np.isfinite(result.phi)
        assert np.isfinite(result.total_integration)


class TestIntegratedInformationApproximationBenchmarks:
    def test_benchmark_report_documents_expected_synthetic_ordering(self) -> None:
        report = benchmark_integrated_information_approximations(
            n_samples=256,
            n_bins=8,
        )

        assert isinstance(report, IntegratedInformationBenchmarkReport)
        assert report.expected_ordering_passed is True
        assert report.locked_phi_margin > 0.0
        assert report.modular_total_margin > 0.0
        assert report.noisy_lock_phi_margin > 0.0
        assert report.phase_lag_total_margin > 0.0
        assert [case.name for case in report.cases] == [
            "independent",
            "modular",
            "phase_lag_chain",
            "noisy_locked",
            "locked",
        ]

        by_name = {case.name: case.result for case in report.cases}
        assert by_name["locked"].phi > by_name["independent"].phi
        assert by_name["locked"].phi > by_name["modular"].phi
        assert by_name["locked"].phi > by_name["noisy_locked"].phi
        assert by_name["noisy_locked"].phi > by_name["independent"].phi
        assert by_name["modular"].total_integration > (
            by_name["independent"].total_integration
        )
        assert by_name["phase_lag_chain"].total_integration > (
            by_name["independent"].total_integration
        )

    def test_benchmark_report_is_json_ready_and_keeps_claim_boundary(self) -> None:
        record = benchmark_integrated_information_approximations(
            n_samples=128,
            n_bins=8,
        ).to_audit_record()

        assert record["monitor"] == "integrated_information"
        assert record["benchmark"] == "deterministic_synthetic_approximation_cases"
        assert record["claim_boundary"] == "engineering_proxy_not_theoretical_iit"
        assert record["expected_ordering_passed"] is True
        assert record["noisy_lock_phi_margin"] > 0.0
        assert record["phase_lag_total_margin"] > 0.0
        assert len(record["cases"]) == 5
        assert record["cases"][0]["result"]["monitor"] == "integrated_information"

    def test_benchmark_rejects_too_short_series(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            benchmark_integrated_information_approximations(n_samples=31)
