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
    _cross_partition_mean,
    _mean_off_diagonal,
    _minimum_bipartition,
    _mutual_information,
    _normalise_phi,
    benchmark_integrated_information_approximations,
    integrated_information,
)
from tests.typing_contracts import assert_precise_ndarray_hint

TWO_PI = 2.0 * np.pi


class TestIntegratedInformationContracts:
    def test_public_array_contract_is_parameterised(self) -> None:
        hints = get_type_hints(integrated_information)

        assert_precise_ndarray_hint(hints["phase_series"])
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

    def test_mixed_boolean_alias_phase_series_is_rejected(self) -> None:
        phases = np.array([[0.0, True], [1.0, 2.0]], dtype=object)

        with pytest.raises(ValueError, match="phase_series"):
            integrated_information(phases)

    @pytest.mark.parametrize("n_bins", [False, 1, 1.5])
    def test_invalid_bin_count_is_rejected(self, n_bins: Any) -> None:
        phases = np.zeros((2, 8), dtype=np.float64)

        with pytest.raises(ValueError, match="n_bins"):
            integrated_information(phases, n_bins=n_bins)

    def test_numpy_integer_bin_count_is_accepted(self) -> None:
        phases = np.zeros((2, 8), dtype=np.float64)

        result = integrated_information(phases, n_bins=np.int64(4))

        assert result.n_bins == 4

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "phi": -0.1,
                "normalised_phi": 0.0,
                "total_integration": 0.0,
                "minimum_partition": ((0,), (1,)),
                "pairwise_mi": [[0.0, 0.0], [0.0, 0.0]],
                "n_bins": 4,
            },
            {
                "phi": 0.0,
                "normalised_phi": 1.1,
                "total_integration": 0.0,
                "minimum_partition": ((0,), (1,)),
                "pairwise_mi": [[0.0, 0.0], [0.0, 0.0]],
                "n_bins": 4,
            },
            {
                "phi": 0.5,
                "normalised_phi": 0.1,
                "total_integration": 0.1,
                "minimum_partition": ((0,), (1,)),
                "pairwise_mi": [[0.0, 0.1], [0.1, 0.0]],
                "n_bins": 4,
            },
            {
                "phi": 0.0,
                "normalised_phi": 0.0,
                "total_integration": 0.0,
                "minimum_partition": ((0,), (0,)),
                "pairwise_mi": [[0.0, 0.0], [0.0, 0.0]],
                "n_bins": 4,
            },
            {
                "phi": 0.0,
                "normalised_phi": 0.0,
                "total_integration": 0.0,
                "minimum_partition": ((0,), (1,)),
                "pairwise_mi": [[0.0, 0.2], [0.1, 0.0]],
                "n_bins": 4,
            },
            {
                "phi": 0.0,
                "normalised_phi": 0.0,
                "total_integration": 0.0,
                "minimum_partition": ((0,), (1,)),
                "pairwise_mi": [[0.0, np.nan], [np.nan, 0.0]],
                "n_bins": 4,
            },
        ],
    )
    def test_result_record_rejects_invalid_invariants(
        self,
        payload: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError):
            IntegratedInformationResult(**payload)


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


class TestIntegratedInformationResidualPaths:
    def test_wrapping_is_applied_before_binning(self) -> None:
        phases = np.array(
            [
                [0.0, np.pi / 3, np.pi, 4 * np.pi / 3, 5 * np.pi / 3],
                [np.pi / 2, 4 * np.pi / 3, 5 * np.pi / 3, 2 * np.pi, 7 * np.pi / 3],
            ],
            dtype=np.float64,
        )
        shifted = phases + 4.0 * np.pi

        base_result = integrated_information(phases, n_bins=12)
        shifted_result = integrated_information(shifted, n_bins=12)

        np.testing.assert_allclose(
            base_result.pairwise_mi, shifted_result.pairwise_mi, atol=1e-12
        )
        assert base_result.phi == pytest.approx(shifted_result.phi, rel=0.0, abs=1e-12)
        assert base_result.total_integration == pytest.approx(
            shifted_result.total_integration, rel=0.0, abs=1e-12
        )

    def test_mutual_information_empty_streams_return_zero(self) -> None:
        empty = np.array([], dtype=np.float64)
        assert _mutual_information(empty, empty, 8) == 0.0

    def test_mean_off_diagonal_handles_degenerate_and_standard_shapes(self) -> None:
        singleton = np.array([[0.0]], dtype=np.float64)
        assert _mean_off_diagonal(singleton) == 0.0

        matrix = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )
        assert _mean_off_diagonal(matrix) == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)

    def test_minimum_bipartition_and_guards(self) -> None:
        matrix = np.array(
            [[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]],
            dtype=np.float64,
        )
        minimum_partition, phi = _minimum_bipartition(matrix)

        assert minimum_partition == ((0,), (1, 2))
        assert phi == pytest.approx(3.0)

        with pytest.raises(ValueError, match="at least two oscillators"):
            _minimum_bipartition(np.zeros((1, 1), dtype=np.float64))

    def test_cross_partition_mean_and_empty_partition_fallback(self) -> None:
        matrix = np.array(
            [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]],
            dtype=np.float64,
        )
        assert _cross_partition_mean(matrix, (0,), (1, 2)) == pytest.approx(1.5)
        assert _cross_partition_mean(matrix, (0, 1), ()) == 0.0

    def test_normalise_phi_handles_non_positive_scale(self) -> None:
        assert _normalise_phi(2.5, 0) == 0.0
        assert _normalise_phi(2.5, 1) == 0.0
