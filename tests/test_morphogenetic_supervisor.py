# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Morphogenetic topology supervisor tests

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.supervisor.morphogenetic as morphogenetic_module
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticFieldPolicy,
    MorphogeneticFieldResult,
    MorphogeneticFieldSnapshot,
    MorphogeneticFieldState,
    MorphogeneticTopologySupervisor,
    build_morphogenetic_field_snapshot,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def _zero_knm(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


class TestMorphogeneticPolicyValidation:
    def test_rejects_invalid_policy_values(self) -> None:
        assert morphogenetic_module.MorphogeneticFieldPolicy is MorphogeneticFieldPolicy
        with pytest.raises(ValueError, match="growth_rate"):
            MorphogeneticFieldPolicy(growth_rate=1.2)
        with pytest.raises(ValueError, match="coherence_target"):
            MorphogeneticFieldPolicy(coherence_target=-0.1)
        with pytest.raises(ValueError, match="max_delta"):
            MorphogeneticFieldPolicy(max_delta=-0.1)


class TestMorphogeneticTopologySupervisor:
    def test_coherent_pair_grows_edge_from_empty_field(self) -> None:
        phases = np.array([0.0, 0.02, np.pi])
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(
                growth_rate=0.5,
                shrink_rate=0.1,
                diffusion_rate=0.0,
                coherence_target=0.8,
                max_delta=0.2,
                max_coupling=1.0,
            )
        )

        result = supervisor.step(phases, _zero_knm(3))

        assert isinstance(result, MorphogeneticFieldResult)
        assert result.knm[0, 1] > 0.0
        assert result.knm[1, 0] > 0.0
        assert result.grown_edges
        assert result.shrunk_edges == ()
        assert 0.0 <= result.global_coherence <= 1.0
        assert supervisor.last_result is result

    def test_incoherent_pair_shrinks_existing_field_and_coupling(self) -> None:
        phases = np.array([0.0, np.pi, 0.01])
        knm = np.full((3, 3), 0.4, dtype=np.float64)
        np.fill_diagonal(knm, 0.0)
        field = np.full((3, 3), 0.8, dtype=np.float64)
        np.fill_diagonal(field, 0.0)
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(
                growth_rate=0.0,
                shrink_rate=0.5,
                diffusion_rate=0.0,
                coherence_target=0.9,
                max_delta=0.2,
                max_coupling=1.0,
            )
        )

        result = supervisor.step(phases, knm, MorphogeneticFieldState(field))

        assert result.knm[0, 1] < knm[0, 1]
        assert result.knm[1, 0] < knm[1, 0]
        assert result.shrunk_edges
        assert np.all(result.knm >= 0.0)
        np.testing.assert_allclose(np.diag(result.knm), 0.0)

    def test_diffusion_spreads_incident_field(self) -> None:
        phases = np.array([0.0, 0.1, 0.2, 0.3])
        field = np.zeros((4, 4), dtype=np.float64)
        field[0, 1] = 1.0
        field[1, 0] = 1.0
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(
                growth_rate=0.0,
                shrink_rate=0.0,
                diffusion_rate=0.5,
                max_delta=0.1,
                max_coupling=1.0,
            )
        )

        result = supervisor.step(phases, _zero_knm(4), MorphogeneticFieldState(field))

        assert result.field_state.field[0, 2] > 0.0
        assert result.field_state.field[2, 0] > 0.0
        assert result.delta_norm > 0.0

    def test_audit_record_has_snapshot_and_edge_deltas(self) -> None:
        phases = np.array([0.0, 0.01, np.pi])
        result = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(max_delta=0.1)
        ).step(phases, _zero_knm(3))

        record = result.to_audit_record()

        assert set(record) == {
            "global_coherence",
            "delta_norm",
            "grown_edges",
            "shrunk_edges",
            "field",
        }
        assert record["field"]["shape"] == [3, 3]
        assert record["grown_edges"]

    def test_field_snapshot_builds_heatmap_and_top_edges(self) -> None:
        field = np.array(
            [
                [0.0, 1.0, 0.25],
                [0.5, 0.0, 0.75],
                [0.125, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

        snapshot = build_morphogenetic_field_snapshot(
            MorphogeneticFieldState(field),
            top_k=3,
            palette=" .#",
        )

        assert isinstance(snapshot, MorphogeneticFieldSnapshot)
        assert snapshot.shape == (3, 3)
        assert snapshot.mean == pytest.approx(float(np.mean(field)))
        assert snapshot.minimum == pytest.approx(0.0)
        assert snapshot.maximum == pytest.approx(1.0)
        assert snapshot.l2_norm == pytest.approx(float(np.linalg.norm(field)))
        assert snapshot.heatmap_rows == (" # ", ". #", "   ")
        assert snapshot.top_edges == (
            (0, 1, pytest.approx(1.0)),
            (1, 2, pytest.approx(0.75)),
            (1, 0, pytest.approx(0.5)),
        )
        assert snapshot.to_audit_record()["top_edges"] == [
            {"source": 0, "target": 1, "weight": pytest.approx(1.0)},
            {"source": 1, "target": 2, "weight": pytest.approx(0.75)},
            {"source": 1, "target": 0, "weight": pytest.approx(0.5)},
        ]

    def test_field_snapshot_accepts_step_results(self) -> None:
        result = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(growth_rate=0.5, max_delta=0.1),
        ).step(np.array([0.0, 0.01, np.pi]), _zero_knm(3))

        snapshot = build_morphogenetic_field_snapshot(result, top_k=1)

        assert snapshot.shape == (3, 3)
        assert len(snapshot.heatmap_rows) == 3
        assert len(snapshot.top_edges) == 1
        assert snapshot.to_audit_record()["shape"] == [3, 3]

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"top_k": -1}, "top_k must be non-negative"),
            ({"palette": ""}, "palette must be a non-empty string"),
        ],
    )
    def test_field_snapshot_rejects_invalid_snapshot_inputs(
        self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            build_morphogenetic_field_snapshot(
                MorphogeneticFieldState(np.zeros((2, 2), dtype=np.float64)),
                **kwargs,
            )

    def test_rejects_invalid_inputs(self) -> None:
        supervisor = MorphogeneticTopologySupervisor()
        with pytest.raises(ValueError, match="one-dimensional"):
            supervisor.step(np.zeros((2, 2)), _zero_knm(2))
        with pytest.raises(ValueError, match="shape"):
            supervisor.step(np.zeros(3), _zero_knm(2))
        with pytest.raises(ValueError, match="field must have shape"):
            supervisor.step(
                np.zeros(3),
                _zero_knm(3),
                MorphogeneticFieldState(np.zeros((2, 2))),
            )

    def test_result_feeds_upde_engine(self) -> None:
        phases = np.array([0.0, 0.02, 0.04, np.pi, np.pi + 0.01, np.pi + 0.03])
        omegas = np.ones(6, dtype=np.float64)
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(
                growth_rate=0.4,
                shrink_rate=0.1,
                diffusion_rate=0.1,
                coherence_target=0.8,
                max_delta=0.1,
                max_coupling=1.0,
            )
        )

        result = supervisor.step(phases, _zero_knm(6))
        engine = UPDEEngine(6, dt=0.01)
        updated = engine.step(
            phases,
            omegas,
            result.knm,
            0.0,
            0.0,
            np.zeros((6, 6), dtype=np.float64),
        )
        r_updated, _ = compute_order_parameter(updated)

        assert updated.shape == phases.shape
        assert np.all(np.isfinite(updated))
        assert 0.0 <= r_updated <= 1.0

    def test_exported_from_supervisor_package(self) -> None:
        import scpn_phase_orchestrator.supervisor as supervisor

        assert supervisor.MorphogeneticTopologySupervisor is (
            MorphogeneticTopologySupervisor
        )
        assert supervisor.MorphogeneticFieldPolicy is MorphogeneticFieldPolicy
