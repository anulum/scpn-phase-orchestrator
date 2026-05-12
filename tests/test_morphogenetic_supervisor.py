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
    MorphogeneticFieldSVG,
    MorphogeneticTopologySupervisor,
    build_morphogenetic_field_snapshot,
    render_morphogenetic_field_svg,
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

    def test_reset_clears_cached_result_without_mutating_field_snapshot(self) -> None:
        phases = np.array([0.0, 0.01, np.pi])
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(growth_rate=0.5, max_delta=0.1),
        )
        result = supervisor.step(phases, _zero_knm(3))
        cached_field = result.field_state.field.copy()

        supervisor.reset()

        assert supervisor.last_result is None
        np.testing.assert_allclose(result.field_state.field, cached_field)

    def test_zero_coupling_policy_keeps_single_oscillator_field_bounded(self) -> None:
        phases = np.array([0.75])
        knm = np.array([[2.0]], dtype=np.float64)
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(
                growth_rate=1.0,
                shrink_rate=1.0,
                diffusion_rate=1.0,
                coherence_target=0.0,
                max_delta=1.0,
                max_coupling=0.0,
            )
        )

        result = supervisor.step(phases, knm)

        np.testing.assert_allclose(result.knm, [[0.0]])
        np.testing.assert_allclose(result.field_state.field, [[0.0]])
        assert result.delta_norm == pytest.approx(2.0)
        assert result.grown_edges == ()
        assert result.shrunk_edges == ()
        assert result.global_coherence == pytest.approx(1.0)

    def test_diagonal_field_deltas_do_not_become_control_actions(self) -> None:
        phases = np.array([0.0, np.pi])
        knm = np.array(
            [
                [0.25, 0.5],
                [0.5, 0.25],
            ],
            dtype=np.float64,
        )
        field = np.array(
            [
                [0.9, 0.8],
                [0.8, 0.9],
            ],
            dtype=np.float64,
        )
        supervisor = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(
                growth_rate=0.0,
                shrink_rate=0.5,
                diffusion_rate=0.0,
                coherence_target=1.0,
                max_delta=0.2,
                max_coupling=1.0,
            )
        )

        result = supervisor.step(phases, knm, MorphogeneticFieldState(field))

        assert result.shrunk_edges == (
            (0, 1, pytest.approx(-0.1)),
            (1, 0, pytest.approx(-0.1)),
        )
        assert result.grown_edges == ()
        assert all(src != dst for src, dst, _ in result.shrunk_edges)
        np.testing.assert_allclose(np.diag(result.field_state.field), 0.0)

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

    def test_snapshot_one_symbol_palette_and_diagonal_edges_are_audit_safe(
        self,
    ) -> None:
        field = np.array(
            [
                [1.0, 0.2, 0.0],
                [0.0, 0.9, 0.8],
                [0.4, 0.0, 0.7],
            ],
            dtype=np.float64,
        )

        snapshot = build_morphogenetic_field_snapshot(
            MorphogeneticFieldState(field),
            top_k=5,
            palette="x",
        )

        assert snapshot.heatmap_rows == ("xxx", "xxx", "xxx")
        assert snapshot.top_edges == (
            (1, 2, pytest.approx(0.8)),
            (2, 0, pytest.approx(0.4)),
            (0, 1, pytest.approx(0.2)),
        )
        assert all(src != dst for src, dst, _ in snapshot.top_edges)

    def test_field_snapshot_accepts_step_results(self) -> None:
        result = MorphogeneticTopologySupervisor(
            MorphogeneticFieldPolicy(growth_rate=0.5, max_delta=0.1),
        ).step(np.array([0.0, 0.01, np.pi]), _zero_knm(3))

        snapshot = build_morphogenetic_field_snapshot(result, top_k=1)

        assert snapshot.shape == (3, 3)
        assert len(snapshot.heatmap_rows) == 3
        assert len(snapshot.top_edges) == 1
        assert snapshot.to_audit_record()["shape"] == [3, 3]

    def test_field_svg_renderer_emits_reviewable_svg_artifact(self) -> None:
        field = np.array(
            [
                [0.0, 0.9, 0.2],
                [0.4, 0.0, 0.7],
                [0.1, 0.3, 0.0],
            ],
            dtype=np.float64,
        )

        svg = render_morphogenetic_field_svg(
            MorphogeneticFieldState(field),
            top_k=2,
            cell_size=16,
            title="field <review>",
        )

        assert isinstance(svg, MorphogeneticFieldSVG)
        assert svg.width == 48
        assert svg.height == 156
        assert svg.snapshot.top_edges == (
            (0, 1, pytest.approx(0.9)),
            (1, 2, pytest.approx(0.7)),
        )
        assert "<svg " in svg.svg
        assert "field &lt;review&gt;" in svg.svg
        assert "0->1:0.90" in svg.svg
        assert "1->2:0.70" in svg.svg
        assert svg.to_audit_record()["snapshot"]["shape"] == [3, 3]

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

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"cell_size": 7}, "cell_size must be at least 8"),
            ({"title": ""}, "title must be a non-empty string"),
            ({"top_k": -1}, "top_k must be non-negative"),
        ],
    )
    def test_field_svg_renderer_rejects_invalid_inputs(
        self,
        kwargs: dict[str, object],
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            render_morphogenetic_field_svg(
                MorphogeneticFieldState(np.zeros((2, 2), dtype=np.float64)),
                **kwargs,
            )

    @pytest.mark.parametrize(
        ("phases", "knm", "field_state", "message"),
        [
            (
                np.array([], dtype=np.float64),
                np.zeros((0, 0), dtype=np.float64),
                None,
                "at least one oscillator",
            ),
            (
                np.array([0.0, np.nan], dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                None,
                "phases must be finite",
            ),
            (
                np.zeros(2, dtype=np.float64),
                np.array([[0.0, np.inf], [0.0, 0.0]], dtype=np.float64),
                None,
                "knm must be finite",
            ),
            (
                np.zeros(2, dtype=np.float64),
                np.array([[0.0, -0.1], [0.0, 0.0]], dtype=np.float64),
                None,
                "knm must be non-negative",
            ),
            (
                np.zeros(2, dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                MorphogeneticFieldState(
                    np.array([[0.0, np.nan], [0.0, 0.0]], dtype=np.float64)
                ),
                "field must be finite",
            ),
            (
                np.zeros(2, dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                MorphogeneticFieldState(
                    np.array([[0.0, 1.1], [0.0, 0.0]], dtype=np.float64)
                ),
                r"field values must be in \[0, 1\]",
            ),
        ],
    )
    def test_rejects_invalid_supervisor_state_domains(
        self,
        phases: np.ndarray,
        knm: np.ndarray,
        field_state: MorphogeneticFieldState | None,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            MorphogeneticTopologySupervisor().step(phases, knm, field_state)

    def test_rejects_invalid_input_geometry(self) -> None:
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

    @pytest.mark.parametrize(
        ("field", "message"),
        [
            (np.zeros((2, 3), dtype=np.float64), "field must be a square matrix"),
            (
                np.array([[0.0, np.inf], [0.0, 0.0]], dtype=np.float64),
                "field must be finite",
            ),
            (
                np.array([[0.0, -0.1], [0.0, 0.0]], dtype=np.float64),
                r"field values must be in \[0, 1\]",
            ),
        ],
    )
    def test_snapshot_rejects_invalid_audit_field_domains(
        self,
        field: np.ndarray,
        message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            build_morphogenetic_field_snapshot(MorphogeneticFieldState(field))

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
        assert supervisor.MorphogeneticFieldSVG is MorphogeneticFieldSVG
        assert supervisor.render_morphogenetic_field_svg is (
            render_morphogenetic_field_svg
        )
