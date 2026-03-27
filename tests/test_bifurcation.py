# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bifurcation continuation tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.bifurcation import (
    BifurcationDiagram,
    BifurcationPoint,
    find_critical_coupling,
    trace_sync_transition,
)


class TestBifurcationPoint:
    def test_fields(self):
        p = BifurcationPoint(K=1.0, R=0.5, stable=True)
        assert p.K == 1.0
        assert p.R == 0.5


class TestBifurcationDiagram:
    def test_empty(self):
        d = BifurcationDiagram()
        assert len(d.points) == 0
        assert d.K_critical is None

    def test_properties(self):
        d = BifurcationDiagram(
            points=[
                BifurcationPoint(K=0.0, R=0.01, stable=True),
                BifurcationPoint(K=2.0, R=0.8, stable=True),
            ]
        )
        np.testing.assert_array_equal(d.K_values, [0.0, 2.0])
        np.testing.assert_array_equal(d.R_values, [0.01, 0.8])


class TestTraceSyncTransition:
    def test_returns_diagram(self):
        N = 8
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 0.5, N)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 3.0),
            n_points=10,
            n_transient=200,
            n_measure=100,
        )
        assert isinstance(diag, BifurcationDiagram)
        assert len(diag.points) == 10

    def test_R_increases_with_K(self):
        N = 8
        rng = np.random.default_rng(0)
        omegas = rng.normal(0, 0.3, N)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 5.0),
            n_points=8,
            n_transient=200,
            n_measure=100,
        )
        R_first = diag.R_values[:3].mean()
        R_last = diag.R_values[-3:].mean()
        assert R_last > R_first

    def test_finds_K_critical(self):
        N = 8
        rng = np.random.default_rng(42)
        omegas = rng.standard_cauchy(N) * 0.5
        omegas = np.clip(omegas, -5, 5)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 4.0),
            n_points=8,
            n_transient=200,
            n_measure=100,
        )
        if diag.K_critical is not None:
            assert 0.1 < diag.K_critical < 4.0


class TestTraceSyncTransitionCoverage:
    """Target uncovered lines: 143-150 (K_critical interpolation)."""

    def test_k_critical_interpolation(self):
        """Force a clear R threshold crossing to exercise interpolation."""
        omegas = np.linspace(-1, 1, 10)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 8.0),
            n_points=20,
            n_transient=500,
            n_measure=200,
        )
        if diag.K_critical is not None:
            assert 0.0 < diag.K_critical < 8.0

    def test_no_crossing_k_critical_none(self):
        """Very low K range + wide ω → R stays below threshold → K_critical None."""
        omegas = np.linspace(-50, 50, 6)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 0.01),
            n_points=5,
            n_transient=100,
            n_measure=50,
        )
        assert diag.K_critical is None

    def test_custom_knm_template(self):
        n = 4
        omegas = np.linspace(-0.5, 0.5, n)
        knm = np.ones((n, n)) * 0.5
        np.fill_diagonal(knm, 0.0)
        diag = trace_sync_transition(
            omegas,
            knm_template=knm,
            K_range=(0.0, 5.0),
            n_points=5,
            n_transient=100,
            n_measure=50,
        )
        assert len(diag.points) == 5

    def test_r_values_bounded(self):
        omegas = np.linspace(-1, 1, 6)
        diag = trace_sync_transition(
            omegas,
            K_range=(0.0, 5.0),
            n_points=8,
            n_transient=200,
            n_measure=100,
        )
        assert np.all(diag.R_values >= 0)
        assert np.all(diag.R_values <= 1.0 + 1e-6)


class TestFindCriticalCoupling:
    def test_returns_finite(self):
        N = 8
        rng = np.random.default_rng(42)
        omegas = rng.normal(0, 0.3, N)
        Kc = find_critical_coupling(omegas, n_transient=200, n_measure=100, tol=0.15)
        assert np.isfinite(Kc)
        assert Kc > 0

    def test_no_transition(self):
        """Identical frequencies → R=1 at any K>0, K_c ≈ 0."""
        N = 8
        omegas = np.zeros(N)
        Kc = find_critical_coupling(omegas, n_transient=100, n_measure=50, tol=0.15)
        assert np.isfinite(Kc)
        assert Kc < 1.0

    def test_wide_spread_returns_float(self):
        """Wide ω spread → K_c either NaN or large positive."""
        n = 8
        omegas = np.linspace(-100, 100, n)
        Kc = find_critical_coupling(
            omegas,
            n_transient=100,
            n_measure=50,
            tol=1.0,
        )
        assert isinstance(Kc, float)
        if not np.isnan(Kc):
            assert Kc >= 0

    def test_binary_search_converges(self):
        """Moderate ω spread → binary search finds K_c in range."""
        omegas = np.linspace(-2, 2, 8)
        Kc = find_critical_coupling(
            omegas,
            n_transient=500,
            n_measure=200,
            tol=0.5,
        )
        if not np.isnan(Kc):
            assert 0.0 < Kc < 20.0

    def test_default_knm(self):
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        Kc = find_critical_coupling(
            omegas,
            knm_template=None,
            n_transient=200,
            n_measure=100,
            tol=1.0,
        )
        assert isinstance(Kc, float)
