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

from scpn_phase_orchestrator.supervisor import (
    SheafCoherenceResult,
    SheafCoherenceSupervisor,
    sheaf_coherence,
    sheaf_laplacian,
)
from scpn_phase_orchestrator.upde.sheaf_engine import SheafUPDEEngine


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

        assert "numpy.ndarray" in str(hints["node_states"])
        assert "float64" in str(hints["node_states"])
        assert "numpy.ndarray" in str(hints["restriction_maps"])
        assert hints["return"] is SheafCoherenceResult

    def test_invalid_node_state_shape_is_rejected(self) -> None:
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="node_states"):
            sheaf_coherence(np.array([1.0, 2.0]), maps)

    def test_invalid_restriction_shape_is_rejected(self) -> None:
        states = np.zeros((2, 2), dtype=np.float64)

        with pytest.raises(ValueError, match="restriction_maps"):
            sheaf_coherence(states, np.zeros((2, 2), dtype=np.float64))

    def test_non_finite_inputs_are_rejected(self) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        states[0, 0] = np.inf
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="finite"):
            sheaf_coherence(states, maps)

    def test_negative_tolerance_is_rejected(self) -> None:
        states = np.zeros((2, 1), dtype=np.float64)
        maps = np.zeros((2, 2, 1, 1), dtype=np.float64)

        with pytest.raises(ValueError, match="tolerance"):
            sheaf_coherence(states, maps, tolerance=-1.0)


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

    def test_audit_record_is_compact_and_serialisable(self) -> None:
        states = np.zeros((2, 2), dtype=np.float64)
        maps = _identity_maps(n_nodes=2, n_channels=2)

        record = sheaf_coherence(states, maps).to_audit_record()

        assert record["method"] == "directed_cellular_sheaf_laplacian"
        assert record["laplacian_shape"] == [4, 4]
        assert record["residual_shape"] == [2, 2, 2]
        assert record["edge_count"] == 2


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
