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

    def test_alignment_trend_empty(self):
        pgbo = PGBO()
        assert pgbo.alignment_trend() == 0.0

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
