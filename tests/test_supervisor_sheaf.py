# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for supervisor sheaf coherence

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

import scpn_phase_orchestrator.supervisor.sheaf as sheaf_module
from scpn_phase_orchestrator.supervisor import (
    SheafCoherenceResult,
    SheafCoherenceSupervisor,
    SheafObstructionSummary,
    build_sheaf_obstruction_summary,
    sheaf_coherence,
    sheaf_laplacian,
)
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine
from tests.typing_contracts import assert_precise_ndarray_hint


def _identity_maps(n_nodes: int, n_channels: int) -> np.ndarray:
    maps = np.zeros((n_nodes, n_nodes, n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)
    for target in range(n_nodes):
        for source in range(n_nodes):
            if target != source:
                maps[target, source] = identity
    return maps


class TestSheafCoherenceContracts:
    def test_public_array_contracts_are_parameterised(self) -> None:
        hints = get_type_hints(sheaf_coherence)

        assert sheaf_module.sheaf_coherence is sheaf_coherence
        assert_precise_ndarray_hint(hints["node_states"])
        assert "float64" in str(hints["node_states"])
        assert_precise_ndarray_hint(hints["restriction_maps"])
        assert hints["return"] is SheafCoherenceResult

    def test_invalid_node_state_shape_is_rejected(self) -> None:
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="node_states"):
            sheaf_coherence(np.array([1.0, 2.0]), maps)

    def test_invalid_restriction_shape_is_rejected(self) -> None:
        states = np.zeros((2, 2), dtype=np.float64)

        with pytest.raises(ValueError, match="restriction_maps"):
            sheaf_coherence(states, np.zeros((2, 2), dtype=np.float64))

    def test_laplacian_rejects_invalid_restriction_shape(self) -> None:
        with pytest.raises(ValueError, match="restriction_maps"):
            sheaf_laplacian(np.zeros((2, 2, 1), dtype=np.float64))

    def test_empty_node_state_matrix_is_rejected(self) -> None:
        maps = np.zeros((0, 0, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="at least one node"):
            sheaf_coherence(np.zeros((0, 1), dtype=np.float64), maps)

    def test_non_finite_inputs_are_rejected(self) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        states[0, 0] = np.inf
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="finite"):
            sheaf_coherence(states, maps)

    def test_boolean_alias_node_states_are_rejected(self) -> None:
        states = np.array([[0.0], [np.bool_(True)]], dtype=object)
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="node_states must not contain boolean"):
            sheaf_coherence(states, maps)

    def test_non_finite_restrictions_are_rejected_by_coherence_and_laplacian(
        self,
    ) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)
        maps[0, 1, 0, 0] = np.nan

        with pytest.raises(ValueError, match="finite"):
            sheaf_coherence(states, maps)
        with pytest.raises(ValueError, match="finite"):
            sheaf_laplacian(maps)

    def test_boolean_alias_restriction_maps_are_rejected(self) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        maps = np.zeros((2, 2, 1, 1), dtype=object)
        maps[0, 1, 0, 0] = True

        with pytest.raises(
            ValueError, match="restriction_maps must not contain boolean"
        ):
            sheaf_coherence(states, maps)
        with pytest.raises(
            ValueError, match="restriction_maps must not contain boolean"
        ):
            sheaf_laplacian(maps)

    def test_negative_tolerance_is_rejected(self) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="tolerance"):
            sheaf_coherence(states, maps, tolerance=-1.0)

    @pytest.mark.parametrize("tolerance", [True, np.bool_(True), "1e-8", object()])
    def test_malformed_tolerance_is_rejected(self, tolerance: object) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="tolerance"):
            sheaf_coherence(states, maps, tolerance=tolerance)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="tolerance"):
            sheaf_laplacian(maps, tolerance=tolerance)  # type: ignore[arg-type]

    def test_non_square_restriction_maps_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="restriction_maps"):
            sheaf_laplacian(np.zeros((2, 3, 1, 1), dtype=np.float64))

    def test_supervisor_rejects_invalid_tolerance(self) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            SheafCoherenceSupervisor(tolerance=-1e-3)


class TestSheafCoherenceBehaviour:
    def test_consistent_global_section_has_zero_obstruction(self) -> None:
        states = np.tile(np.array([0.25, -0.5], dtype=np.float64), (3, 1))
        maps = _identity_maps(n_nodes=3, n_channels=2)

        result = sheaf_coherence(states, maps)

        assert result.edge_count == 6
        assert result.obstruction_score == pytest.approx(0.0)
        assert result.consistency_energy == pytest.approx(0.0)
        assert result.obstruction_dimension == 0
        assert result.kernel_dimension == 2
        assert np.allclose(result.residuals, 0.0)

    def test_inconsistent_state_reports_obstruction(self) -> None:
        states = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )
        maps = _identity_maps(n_nodes=3, n_channels=2)

        result = sheaf_coherence(states, maps)

        assert result.edge_count == 6
        assert result.obstruction_score > 0.0
        assert result.consistency_energy > 0.0
        assert result.obstruction_dimension > 0

    def test_subthreshold_restrictions_do_not_create_edges_or_residuals(self) -> None:
        states = np.array([[1.0], [2.0]], dtype=np.float64)
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)
        maps[0, 1, 0, 0] = 1e-10
        maps[1, 0, 0, 0] = 1e-10

        result = sheaf_coherence(states, maps, tolerance=1e-8)
        laplacian = sheaf_laplacian(maps, tolerance=1e-8)

        assert result.edge_count == 0
        assert result.obstruction_score == pytest.approx(0.0)
        assert result.obstruction_dimension == 0
        assert np.allclose(result.residuals, 0.0)
        assert np.allclose(laplacian, 0.0)

    def test_laplacian_is_symmetric_positive_semidefinite(self) -> None:
        maps = _identity_maps(n_nodes=4, n_channels=2)

        laplacian = sheaf_laplacian(maps)
        eigenvalues = np.linalg.eigvalsh(laplacian)

        assert laplacian.shape == (8, 8)
        assert np.allclose(laplacian, laplacian.T)
        assert np.min(eigenvalues) >= -1e-9

    def test_transform_restriction_can_be_consistent(self) -> None:
        states = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
        maps = np.zeros((2, 2, 2, 2), dtype=np.float64)
        swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        maps[0, 1] = swap
        maps[1, 0] = swap

        result = sheaf_coherence(states, maps)

        assert result.obstruction_score == pytest.approx(0.0)
        assert result.kernel_dimension == 2

    def test_single_node_case_preserves_channel_kernel_dim(self) -> None:
        states = np.array([[0.25, -0.5]], dtype=np.float64)
        maps = np.zeros((1, 1, 2, 2), dtype=np.float64)
        result = sheaf_coherence(states, maps)

        assert result.edge_count == 0
        assert result.obstruction_dimension == 0
        assert result.kernel_dimension == states.shape[1]
        assert result.laplacian.shape == (2, 2)
        assert np.allclose(result.laplacian, 0.0)

    def test_supervisor_uses_configured_tolerance_for_edge_detection(self) -> None:
        states = np.array([[0.0], [1e-9]], dtype=np.float64)
        maps = _identity_maps(n_nodes=2, n_channels=1)

        default = SheafCoherenceSupervisor().assess(states, maps)
        configured = SheafCoherenceSupervisor(tolerance=1e-6).assess(states, maps)
        strict = SheafCoherenceSupervisor(tolerance=1e-12).assess(states, maps)

        assert default.obstruction_dimension == 0
        assert configured.edge_count == 2
        assert configured.obstruction_dimension == 0
        assert strict.obstruction_dimension == 2
        assert strict.edge_count == 2

    def test_audit_record_is_compact_and_serialisable(self) -> None:
        states = np.zeros((2, 2), dtype=np.float64)
        maps = _identity_maps(n_nodes=2, n_channels=2)

        record = sheaf_coherence(states, maps).to_audit_record()

        assert record["method"] == "directed_cellular_sheaf_laplacian"
        assert record["laplacian_shape"] == [4, 4]
        assert record["residual_shape"] == [2, 2, 2]
        assert record["edge_count"] == 2

    def test_obstruction_summary_reports_severity_and_top_residuals(self) -> None:
        states = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )
        result = sheaf_coherence(states, _identity_maps(n_nodes=3, n_channels=2))

        summary = build_sheaf_obstruction_summary(
            result,
            warning_threshold=0.1,
            critical_threshold=0.5,
            top_k=2,
        )

        assert isinstance(summary, SheafObstructionSummary)
        assert summary.severity == "critical"
        assert len(summary.top_residual_edges) == 2
        assert summary.top_residual_edges[0][2] >= summary.top_residual_edges[1][2]
        record = summary.to_audit_record()
        assert record["severity"] == "critical"
        assert (
            record["top_residual_edges"][0]["norm"]
            >= (record["top_residual_edges"][1]["norm"])
        )

    def test_summary_top_k_is_bounded_by_edge_count_and_stable(self) -> None:
        states = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
                [0.0, -1.0],
            ],
            dtype=np.float64,
        )
        result = sheaf_coherence(states, _identity_maps(n_nodes=3, n_channels=2))
        summary = build_sheaf_obstruction_summary(
            result, warning_threshold=0.01, top_k=99
        )

        residual_norms = np.linalg.norm(result.residuals.reshape(-1, 2), axis=1)
        expected = np.sort(residual_norms[residual_norms > 0.0])[::-1]
        reported = np.array(
            [edge[2] for edge in summary.top_residual_edges], dtype=np.float64
        )
        assert len(summary.top_residual_edges) == result.edge_count
        assert np.all(reported <= expected[: len(reported)] + 1e-12)
        assert np.all(np.diff(reported) <= 1e-12)

    def test_obstruction_severity_uses_threshold_boundaries(self) -> None:
        synthetic = SheafCoherenceResult(
            laplacian=np.zeros((2, 2), dtype=np.float64),
            residuals=np.zeros((2, 2, 1), dtype=np.float64),
            obstruction_score=0.25,
            consistency_energy=0.0,
            kernel_dimension=2,
            obstruction_dimension=0,
            edge_count=0,
            tolerance=1e-8,
        )
        exact_warning = build_sheaf_obstruction_summary(
            synthetic,
            warning_threshold=0.25,
            critical_threshold=0.5,
        )
        exact_critical = build_sheaf_obstruction_summary(
            synthetic,
            warning_threshold=0.0,
            critical_threshold=0.25,
        )
        below_threshold = build_sheaf_obstruction_summary(
            synthetic,
            warning_threshold=0.5,
            critical_threshold=1.0,
        )

        assert exact_warning.severity == "warning"
        assert exact_critical.severity == "critical"
        assert below_threshold.severity == "nominal"

    def test_kernel_dimension_is_tolerant_to_small_numerical_noise(self) -> None:
        laplacian = np.array(
            [
                [1.0, -1.0 + 1e-12],
                [-1.0, 1.0 + 1e-12],
            ],
            dtype=np.float64,
        )
        assert sheaf_module._kernel_dimension(laplacian, tolerance=1e-8) == 1
        assert sheaf_module._kernel_dimension(laplacian, tolerance=1e-14) == 0

    def test_obstruction_summary_classifies_nominal_and_warning_states(self) -> None:
        nominal = sheaf_coherence(
            np.zeros((2, 2), dtype=np.float64),
            _identity_maps(n_nodes=2, n_channels=2),
        )
        warning = sheaf_coherence(
            np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64),
            _identity_maps(n_nodes=2, n_channels=2),
        )

        assert build_sheaf_obstruction_summary(nominal).severity == "nominal"
        assert (
            build_sheaf_obstruction_summary(
                warning,
                warning_threshold=0.05,
                critical_threshold=0.5,
            ).severity
            == "warning"
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"warning_threshold": -0.1}, "tolerance"),
            ({"warning_threshold": True}, "tolerance"),
            ({"warning_threshold": np.bool_(True)}, "tolerance"),
            ({"warning_threshold": 0.5, "critical_threshold": 0.1}, "critical"),
            ({"top_k": -1}, "top_k"),
            ({"top_k": True}, "top_k"),
            ({"top_k": np.bool_(True)}, "top_k"),
            ({"top_k": 1.5}, "top_k"),
        ],
    )
    def test_obstruction_summary_rejects_invalid_inputs(
        self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        result = sheaf_coherence(
            np.zeros((2, 2), dtype=np.float64),
            _identity_maps(n_nodes=2, n_channels=2),
        )

        with pytest.raises(ValueError, match=message):
            build_sheaf_obstruction_summary(result, **kwargs)

    @pytest.mark.parametrize("result", [None, object(), "result"])
    def test_obstruction_summary_rejects_invalid_result(self, result: object) -> None:
        with pytest.raises(ValueError, match="result"):
            build_sheaf_obstruction_summary(result)  # type: ignore[arg-type]

    def test_zero_top_k_yields_no_reported_residual_edges(self) -> None:
        states = np.array([[0.0, 0.0], [1.0, -1.0]], dtype=np.float64)
        result = sheaf_coherence(states, _identity_maps(n_nodes=2, n_channels=2))

        summary = build_sheaf_obstruction_summary(result, top_k=0)

        assert summary.top_residual_edges == ()
        assert summary.to_audit_record()["top_residual_edges"] == []
        assert summary.obstruction_score == result.obstruction_score

    def test_top_residual_edges_tie_sort_is_deterministic(self) -> None:
        states = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
        result = sheaf_coherence(states, _identity_maps(n_nodes=3, n_channels=1))
        summary = build_sheaf_obstruction_summary(result, top_k=4)

        assert summary.top_residual_edges == (
            (0, 2, 2.0, (-2.0,)),
            (2, 0, 2.0, (2.0,)),
            (0, 1, 1.0, (-1.0,)),
            (1, 0, 1.0, (1.0,)),
        )


class TestSheafCoherencePipelineWiring:
    def test_supervisor_assesses_sheaf_engine_state(self) -> None:
        n_nodes = 3
        n_channels = 2
        engine = SheafUPDEEngine(n_nodes, n_channels, dt=0.01, method="euler")
        states = np.array(
            [
                [0.1, 0.2],
                [0.12, 0.22],
                [0.15, 0.25],
            ],
            dtype=np.float64,
        )
        omegas = np.zeros_like(states)
        maps = _identity_maps(n_nodes, n_channels) * 0.2
        psi = np.zeros(n_channels, dtype=np.float64)

        next_states = engine.step(states, omegas, maps, zeta=0.0, psi=psi)
        result = SheafCoherenceSupervisor().assess(next_states, maps)

        assert result.laplacian.shape == (n_nodes * n_channels, n_nodes * n_channels)
        assert np.isfinite(result.obstruction_score)
        assert result.edge_count == n_nodes * (n_nodes - 1)


def test_empty_laplacian_kernel_dimension_guard() -> None:
    assert sheaf_module._kernel_dimension(np.zeros((0, 0), dtype=np.float64), 1e-8) == 0
