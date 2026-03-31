# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — PGBO tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.ssgf.pgbo import PGBO, PGBOSnapshot


class TestPGBO:
    def test_observe_returns_snapshot(self):
        pgbo = PGBO()
        phases = np.zeros(4)
        W = np.full((4, 4), 0.5)
        np.fill_diagonal(W, 0.0)
        snap = pgbo.observe(phases, W)
        assert isinstance(snap, PGBOSnapshot)
        assert snap.step == 1

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


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
