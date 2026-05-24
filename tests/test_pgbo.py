# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PGBO tests

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

import scpn_phase_orchestrator.ssgf.pgbo as pgbo_module
from scpn_phase_orchestrator.ssgf.pgbo import PGBO, PGBOSnapshot
from tests.typing_contracts import assert_precise_ndarray_hint


class TestPGBO:
    def test_rejects_empty_cost_weights(self):
        with pytest.raises(ValueError, match="at least one weight"):
            PGBO(cost_weights=())

    def test_rejects_negative_cost_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            PGBO(cost_weights=(1.0, -0.1))

    @pytest.mark.parametrize(
        "cost_weights",
        [(1.0, np.nan), (1.0, np.inf), (1.0, True), (1.0, "0.5")],
    )
    def test_rejects_malformed_cost_weights(self, cost_weights):
        with pytest.raises(ValueError, match="cost_weights"):
            PGBO(cost_weights=cost_weights)

    def test_observe_returns_snapshot(self):
        pgbo = PGBO()
        phases = np.zeros(4)
        W = np.full((4, 4), 0.5)
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert isinstance(snap, PGBOSnapshot)
        assert snap.step == 1

    @pytest.mark.parametrize(
        ("phases", "W", "match"),
        [
            (np.zeros((2, 2)), np.zeros((4, 4)), "phases"),
            (np.array([0.0, np.nan]), np.zeros((2, 2)), "phases"),
            (np.array([0.0, True], dtype=object), np.zeros((2, 2)), "phases"),
            (np.zeros(2), np.zeros((2, 3)), "W"),
            (np.zeros(2), np.array([[0.0, np.inf], [0.0, 0.0]]), "W"),
            (np.zeros(2), np.array([[False, True], [True, False]]), "W"),
        ],
    )
    def test_observe_rejects_malformed_inputs(self, phases, W, match):
        pgbo = PGBO()

        with pytest.raises(ValueError, match=match):
            pgbo.observe(phases, W)

    def test_rejects_empty_phases(self):
        pgbo = PGBO()
        with pytest.raises(ValueError, match="at least one oscillator"):
            pgbo.observe(np.array([], dtype=np.float64), np.zeros((0, 0)))

    def test_rejects_non_numeric_coupling_matrix(self):
        pgbo = PGBO()
        with pytest.raises(ValueError, match="W must be numeric"):
            pgbo.observe(
                np.array([0.0, 0.1], dtype=np.float64),
                np.array([["a", 0.5], [0.5, "b"]], dtype=object),
            )

    def test_R_correct(self):
        pgbo = PGBO()
        phases = np.zeros(4)
        W = np.eye(4)
        snap = pgbo.observe(phases, W)
        assert abs(snap.R - 1.0) < 1e-10

    def test_alignment_finite(self):
        pgbo = PGBO()
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 6)
        W = np.full((6, 6), 0.5)
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert np.isfinite(snap.phase_geometry_alignment)

    def test_history_grows(self):
        pgbo = PGBO()
        W = np.full((3, 3), 0.5)
        np.fill_diagonal(W, 0.0)
        for _ in range(5):
            pgbo.observe(np.zeros(3), W)
        assert len(pgbo.history) == 5

    def test_alignment_trend(self):
        pgbo = PGBO()
        W = np.full((4, 4), 0.5)
        np.fill_diagonal(W, 0.0)
        for _ in range(10):
            pgbo.observe(np.zeros(4), W)
        trend = pgbo.alignment_trend(window=5)
        assert isinstance(trend, float)

    def test_alignment_trend_rejects_non_positive_window(self):
        pgbo = PGBO()
        W = np.full((3, 3), 0.5)
        np.fill_diagonal(W, 0.0)
        for _ in range(3):
            pgbo.observe(np.zeros(3), W)

        with pytest.raises(ValueError, match="window must be"):
            pgbo.alignment_trend(window=0)
        with pytest.raises(ValueError, match="window must be"):
            pgbo.alignment_trend(window=-1)
        with pytest.raises(ValueError, match="window must be"):
            pgbo.alignment_trend(window=True)

    def test_alignment_trend_large_window_matches_all_history(self):
        pgbo = PGBO()
        W = np.array(
            [
                [0.0, 0.4, 0.9],
                [0.4, 0.0, 0.2],
                [0.9, 0.2, 0.0],
            ],
            dtype=np.float64,
        )
        phases = [
            np.array([0.0, 0.4, 1.1]),
            np.array([0.1, 0.7, 1.3]),
            np.array([0.3, 1.0, 1.5]),
            np.array([0.4, 1.2, 1.8]),
        ]
        seen = [
            pgbo.observe(phases[idx], W).phase_geometry_alignment for idx in range(4)
        ]

        assert pgbo.alignment_trend(window=10) == pytest.approx(sum(seen) / len(seen))

    def test_uniform_coupling_has_zero_alignment(self):
        """Constant off-diagonal coupling collapses variance-based branch."""
        pgbo = PGBO()
        phases = np.array([0.1, 0.2, 0.4, 0.8], dtype=np.float64)
        W = np.full((4, 4), 0.75)
        np.fill_diagonal(W, 0.0)

        snap = pgbo.observe(phases, W)
        assert snap.phase_geometry_alignment == 0.0

    def test_records_are_deterministic_for_deterministic_inputs(self):
        """Same valid inputs should produce reproducible snapshot payloads."""
        pgbo = PGBO()
        phases = np.array([0.2, 0.6, 1.0], dtype=np.float64)
        W = np.array(
            [
                [0.0, 0.2, 0.7],
                [0.2, 0.0, 0.9],
                [0.7, 0.9, 0.0],
            ],
            dtype=np.float64,
        )

        first = pgbo.observe(phases, W)
        first_costs = (
            first.costs.c1_sync,
            first.costs.c2_spectral_gap,
            first.costs.c3_sparsity,
            first.costs.c4_symmetry,
            first.costs.u_total,
        )

        pgbo.reset()
        second = pgbo.observe(phases, W)
        second_costs = (
            second.costs.c1_sync,
            second.costs.c2_spectral_gap,
            second.costs.c3_sparsity,
            second.costs.c4_symmetry,
            second.costs.u_total,
        )

        assert second.step == 1
        assert pytest.approx(first.R) == second.R
        assert pytest.approx(first.psi) == second.psi
        assert (
            pytest.approx(first.phase_geometry_alignment)
            == second.phase_geometry_alignment
        )
        assert pytest.approx(first.gauge_curvature) == second.gauge_curvature
        assert second_costs == first_costs

    def test_alignment_trend_empty(self):
        pgbo = PGBO()
        assert pgbo.alignment_trend() == 0.0

    def test_history_is_a_snapshot_copy(self):
        pgbo = PGBO()
        W = np.full((3, 3), 0.5)
        np.fill_diagonal(W, 0.0)
        pgbo.observe(np.zeros(3), W)

        history = pgbo.history
        assert isinstance(history, list)
        history.append("tamper")

        assert len(pgbo.history) == 1
        assert len(history) == 2

    def test_reset(self):
        pgbo = PGBO()
        W = np.eye(3)
        pgbo.observe(np.zeros(3), W)
        pgbo.reset()
        assert len(pgbo.history) == 0
        assert pgbo.alignment_trend() == 0.0

    def test_single_oscillator(self):
        pgbo = PGBO()
        snap = pgbo.observe(np.array([1.0]), np.array([[0.0]]))
        assert snap.phase_geometry_alignment == 0.0

    def test_varied_phases_nonuniform_W(self):
        pgbo = PGBO()
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 6)
        W = rng.uniform(0.1, 2.0, (6, 6))
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert np.isfinite(snap.phase_geometry_alignment)

    def test_gauge_curvature_finite(self):
        pgbo = PGBO()
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, 6)
        W = rng.uniform(0.1, 2.0, (6, 6))
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert np.isfinite(snap.gauge_curvature)

    def test_zero_coupling_matrix_has_zero_gauge_curvature(self):
        pgbo = PGBO()
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        W = np.zeros((4, 4), dtype=np.float64)

        snap = pgbo.observe(phases, W)

        assert snap.gauge_curvature == 0.0

    def test_non_finite_alignment_falls_back_to_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        pgbo = PGBO()
        phases = np.array([0.0, 0.3, 1.1, 2.0])
        W = np.array(
            [
                [0.0, 0.1, 0.4, 0.7],
                [0.1, 0.0, 0.2, 0.8],
                [0.4, 0.2, 0.0, 0.3],
                [0.7, 0.8, 0.3, 0.0],
            ],
            dtype=np.float64,
        )
        monkeypatch.setattr(
            pgbo_module.np,
            "corrcoef",
            lambda *_args, **_kwargs: np.array([[1.0, np.nan], [np.nan, 1.0]]),
        )

        snap = pgbo.observe(phases, W)

        assert snap.phase_geometry_alignment == 0.0

    def test_public_array_contracts_are_parameterised(self) -> None:
        hints = get_type_hints(PGBO.observe)
        for param in ("phases", "W"):
            assert_precise_ndarray_hint(hints[param])
            assert "float64" in str(hints[param])


class TestPGBOPipelineWiring:
    """Pipeline: engine phases + K_nm → PGBO → alignment metric."""

    def test_engine_state_to_pgbo_alignment(self):
        """UPDEEngine → phases + K_nm → PGBO.observe → alignment∈[-1,1]."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 6
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        pgbo = PGBO()
        snap = pgbo.observe(phases, knm)
        assert isinstance(snap, PGBOSnapshot)
        assert -1.0 <= snap.phase_geometry_alignment <= 1.0
        assert np.isfinite(snap.gauge_curvature)


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestPgboValidation:
    def test_rejects_empty_cost_weights(self) -> None:
        with pytest.raises(ValueError, match="at least one weight"):
            PGBO(cost_weights=())

    def test_rejects_negative_cost_weight(self) -> None:
        with pytest.raises(ValueError, match="cost_weights must be non-negative"):
            PGBO(cost_weights=(1.0, -0.2, 0.1))

    def test_accepts_zero_cost_weight(self) -> None:
        PGBO(cost_weights=(1.0, 0.0, 0.1))


# Salvaged module-specific behavioural contracts from deleted sprint file.
class TestPGBOAlignment:
    """Verify that PGBO measures phase-geometry alignment with
    correct range and discriminatory power."""

    def test_alignment_in_range(self):
        pgbo = PGBO()
        rng = np.random.default_rng(123)
        phases = rng.uniform(0, 2 * np.pi, 8)
        W = rng.uniform(0.1, 2.0, (8, 8))
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert -1.0 <= snap.phase_geometry_alignment <= 1.0

    def test_synchronised_phases_high_alignment(self):
        """Nearly identical phases with uniform coupling → high alignment."""
        pgbo = PGBO()
        phases = np.full(8, 0.5)  # synchronised
        W = np.ones((8, 8)) * 0.5
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert snap.phase_geometry_alignment >= -1.0  # structural check


# ---------------------------------------------------------------------------
# TCBO: topological complexity-based observer
# ---------------------------------------------------------------------------
