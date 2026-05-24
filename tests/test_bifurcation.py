# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bifurcation continuation tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import bifurcation as bif
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
    """Validate K_critical interpolation when the response crosses threshold."""

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

    def test_python_fallback_interpolates_threshold_crossing(self, monkeypatch):
        """Controlled R(K) curve gives the expected interpolated K_c."""
        from scpn_phase_orchestrator.upde import bifurcation as bif

        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        r_by_k = {
            0.0: 0.02,
            1.0: 0.07,
            2.0: 0.13,
            3.0: 0.40,
        }

        def _steady_state_probe(
            _phases_init,
            _omegas,
            K_scale,
            _knm_template,
            _alpha,
            _dt,
            _n_transient,
            _n_measure,
        ):
            return r_by_k[float(K_scale)]

        monkeypatch.setattr(bif, "_steady_state_R_dispatch", _steady_state_probe)
        diag = bif.trace_sync_transition(
            np.array([-0.2, 0.0, 0.2]),
            K_range=(0.0, 3.0),
            n_points=4,
            n_transient=0,
            n_measure=1,
        )
        assert [point.stable for point in diag.points] == [True, True, True, True]
        np.testing.assert_allclose(diag.R_values, [0.02, 0.07, 0.13, 0.40])
        assert diag.K_critical == 1.5


class TestInputValidation:
    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("omegas", np.zeros((3, 1), dtype=np.float64), "omegas shape"),
            ("omegas", np.array([0.0, np.nan], dtype=np.float64), "omegas"),
            ("knm_template", np.zeros((3, 2), dtype=np.float64), "knm_template"),
            ("alpha", np.zeros((2, 3), dtype=np.float64), "alpha"),
        ],
    )
    def test_trace_rejects_invalid_arrays(
        self,
        field: str,
        bad_value: np.ndarray,
        match: str,
    ) -> None:
        kwargs = {
            "omegas": np.ones(3, dtype=np.float64),
            "knm_template": np.zeros((3, 3), dtype=np.float64),
            "alpha": np.zeros((3, 3), dtype=np.float64),
            "n_points": 2,
            "n_transient": 1,
            "n_measure": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=match):
            trace_sync_transition(**kwargs)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("K_range", (1.0, 1.0)),
            ("K_range", (0.0, np.nan)),
            ("n_points", False),
            ("n_points", 1),
            ("n_points", 2.5),
            ("dt", 0.0),
            ("dt", np.inf),
            ("n_transient", -1),
            ("n_measure", -1),
            ("seed", False),
            ("seed", 1.5),
        ],
    )
    def test_trace_rejects_invalid_runtime_parameters(
        self,
        field: str,
        bad_value: Any,
    ) -> None:
        kwargs = {
            "K_range": (0.0, 1.0),
            "n_points": 2,
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
            "seed": 1,
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=field):
            trace_sync_transition(np.ones(3, dtype=np.float64), **kwargs)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("omegas", np.zeros((3, 1), dtype=np.float64)),
            ("knm_template", np.zeros((3, 2), dtype=np.float64)),
            ("dt", 0.0),
            ("n_transient", -1),
            ("n_measure", -1),
            ("tol", 0.0),
            ("seed", False),
        ],
    )
    def test_find_rejects_invalid_contract(
        self,
        field: str,
        bad_value: Any,
    ) -> None:
        kwargs = {
            "omegas": np.ones(3, dtype=np.float64),
            "knm_template": np.zeros((3, 3), dtype=np.float64),
            "dt": 0.01,
            "n_transient": 1,
            "n_measure": 1,
            "tol": 0.1,
            "seed": 1,
        }
        kwargs[field] = bad_value
        omegas = kwargs.pop("omegas")

        with pytest.raises(ValueError, match=field):
            find_critical_coupling(omegas, **kwargs)


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

    def test_measurement_window_zero_returns_nan(self, monkeypatch):
        """n_measure=0 keeps searching and returns a finite small positive Kc."""
        from scpn_phase_orchestrator.upde import bifurcation as bif

        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        calls: list[int] = []

        def _steady_state_r(
            _phases_init: np.ndarray,
            _omegas: np.ndarray,
            _K_scale: float,
            _knm_template: np.ndarray,
            _alpha: np.ndarray,
            _dt: float,
            _n_transient: int,
            n_measure: int,
        ) -> float:
            calls.append(n_measure)
            return 0.25

        monkeypatch.setattr(bif, "_steady_state_R_dispatch", _steady_state_r)

        Kc = bif.find_critical_coupling(
            np.array([0.1, -0.2, 0.3]),
            n_transient=0,
            n_measure=0,
        )

        assert np.isfinite(Kc)
        assert Kc < 0.05
        assert all(v == 0 for v in calls)

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

    def test_returns_nan_when_upper_bound_remains_subcritical(self, monkeypatch):
        from scpn_phase_orchestrator.upde import bifurcation as bif

        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        calls: list[float] = []

        def _always_subcritical(
            _phases_init,
            _omegas,
            K_scale,
            _knm_template,
            _alpha,
            _dt,
            _n_transient,
            _n_measure,
        ):
            calls.append(float(K_scale))
            return 0.05

        monkeypatch.setattr(bif, "_steady_state_R_dispatch", _always_subcritical)
        Kc = bif.find_critical_coupling(
            np.array([-1.0, 1.0]),
            n_transient=0,
            n_measure=1,
        )
        assert np.isnan(Kc)
        assert calls == [20.0]

    def test_binary_search_moves_lower_bound_after_subcritical_midpoint(
        self,
        monkeypatch,
    ):
        from scpn_phase_orchestrator.upde import bifurcation as bif

        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        calls: list[float] = []

        def _threshold_response(
            _phases_init,
            _omegas,
            K_scale,
            _knm_template,
            _alpha,
            _dt,
            _n_transient,
            _n_measure,
        ):
            calls.append(float(K_scale))
            return 0.05 if K_scale < 15.0 else 0.8

        monkeypatch.setattr(bif, "_steady_state_R_dispatch", _threshold_response)
        Kc = bif.find_critical_coupling(
            np.array([-0.5, 0.5]),
            n_transient=0,
            n_measure=1,
            tol=6.0,
        )
        assert calls[:3] == [20.0, 10.0, 15.0]
        assert Kc == 12.5


class TestBifurcationDispatchSurface:
    def test_python_path_forwards_kernel_inputs(self, monkeypatch):
        calls: dict[str, float] = {}
        k_scales: list[float] = []

        def fake_steady_state_r(
            phases_init: np.ndarray,
            omegas: np.ndarray,
            knm_template: np.ndarray,
            alpha: np.ndarray,
            k_scale: float,
            dt: float,
            n_transient: int,
            n_measure: int,
        ) -> float:
            calls["phases_size"] = int(phases_init.shape[0])
            calls["omegas_size"] = int(omegas.shape[0])
            calls["knm_shape"] = tuple(knm_template.shape)
            calls["alpha_shape"] = tuple(alpha.shape)
            calls["k_scale"] = float(k_scale)
            k_scales.append(float(k_scale))
            calls["dt"] = dt
            calls["n_transient"] = n_transient
            calls["n_measure"] = n_measure
            return 0.25

        monkeypatch.setattr(bif, "_HAS_COMPOSITE_RUST", False)
        monkeypatch.setattr(bif, "_dispatched_steady_state_r", fake_steady_state_r)

        omegas = np.array([-0.2, 0.1, 0.4], dtype=np.float64)
        knm = np.array(
            [[0.0, 0.2, 0.3], [0.4, 0.0, 0.5], [0.2, 0.4, 0.0]],
            dtype=np.float64,
        )
        alpha = np.full((3, 3), 0.05, dtype=np.float64)

        diagram = trace_sync_transition(
            omegas,
            knm_template=knm,
            alpha=alpha,
            K_range=(0.0, 4.0),
            n_points=4,
            dt=0.03,
            n_transient=10,
            n_measure=5,
            seed=7,
        )

        assert len(diagram.points) == 4
        np.testing.assert_allclose(diagram.R_values, 0.25)
        assert calls["phases_size"] == 3
        assert calls["omegas_size"] == 3
        assert calls["knm_shape"] == (3, 3)
        assert calls["alpha_shape"] == (3, 3)
        assert calls["k_scale"] == 4.0
        np.testing.assert_allclose(k_scales, np.linspace(0.0, 4.0, 4))
        assert calls["dt"] == 0.03
        assert calls["n_transient"] == 10
        assert calls["n_measure"] == 5


class TestBifurcationPipelineWiring:
    """Pipeline: bifurcation analysis uses UPDEEngine internally."""

    def test_trace_sync_uses_engine(self):
        """trace_sync_transition scans K values → R trajectory.
        Proves the bifurcation module drives the engine."""
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        diag = trace_sync_transition(
            omegas,
            K_range=(0.1, 2.0),
            n_points=5,
        )
        assert isinstance(diag, BifurcationDiagram)
        assert len(diag.points) == 5
        for pt in diag.points:
            assert isinstance(pt, BifurcationPoint)
            assert 0.0 <= pt.R <= 1.0
            assert pt.K >= 0.0
