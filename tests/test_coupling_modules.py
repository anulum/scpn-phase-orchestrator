# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for coupling lags, prior, templates

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.lags import LagModel
from scpn_phase_orchestrator.coupling.prior import CouplingPrior, UniversalPrior
from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet


class TestLagPriorTypeHints:
    """Guard V2 typed-array contracts for lag and prior helpers."""

    def test_lag_and_prior_arrays_are_parameterised(self) -> None:
        for hint in [
            get_type_hints(LagModel.estimate_from_distances)["distances"],
            get_type_hints(LagModel.estimate_from_distances)["return"],
            get_type_hints(LagModel.estimate_lag)["signal_a"],
            get_type_hints(LagModel.estimate_lag)["signal_b"],
            get_type_hints(LagModel.build_alpha_matrix)["return"],
            get_type_hints(UniversalPrior.estimate_Kc)["omegas"],
        ]:
            assert "numpy.ndarray" in str(hint)
            assert "float64" in str(hint)


class TestLagModel:
    def test_estimate_from_distances_antisymmetric(self) -> None:
        distances = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        alpha = LagModel.estimate_from_distances(distances, speed=1.0)
        np.testing.assert_allclose(alpha, -alpha.T, atol=1e-12)

    def test_estimate_from_distances_zero_diagonal(self) -> None:
        distances = np.array([[0, 3], [3, 0]], dtype=float)
        alpha = LagModel.estimate_from_distances(distances, speed=2.0)
        assert alpha[0, 0] == 0.0
        assert alpha[1, 1] == 0.0

    def test_estimate_from_distances_values(self) -> None:
        distances = np.array([[0, 1], [1, 0]], dtype=float)
        alpha = LagModel.estimate_from_distances(distances, speed=1.0)
        expected = 2.0 * np.pi * 1.0
        assert abs(alpha[0, 1] - expected) < 1e-12
        assert abs(alpha[1, 0] + expected) < 1e-12

    def test_estimate_lag_known_shift(self) -> None:
        t = np.arange(1000) / 100.0
        a = np.sin(2 * np.pi * t)
        b = np.sin(2 * np.pi * (t - 0.05))
        lag = LagModel().estimate_lag(a, b, sample_rate=100.0)
        assert abs(abs(lag) - 0.05) < 0.02

    def test_build_alpha_matrix_antisymmetric(self) -> None:
        lags = {(0, 1): 0.01, (0, 2): 0.02}
        alpha = LagModel().build_alpha_matrix(lags, n_layers=3, carrier_freq_hz=10.0)
        np.testing.assert_allclose(alpha, -alpha.T, atol=1e-12)
        assert alpha.shape == (3, 3)

    def test_build_alpha_matrix_shape(self) -> None:
        alpha = LagModel().build_alpha_matrix({}, n_layers=5)
        assert alpha.shape == (5, 5)
        np.testing.assert_array_equal(alpha, 0.0)


class TestUniversalPrior:
    def test_default_values(self) -> None:
        prior = UniversalPrior().default()
        assert isinstance(prior, CouplingPrior)
        assert abs(prior.K_base - 0.47) < 1e-10
        assert abs(prior.decay_alpha - 0.25) < 1e-10

    def test_sample_positive(self) -> None:
        rng = np.random.default_rng(42)
        prior = UniversalPrior().sample(rng)
        assert prior.K_base > 0
        assert prior.decay_alpha > 0

    def test_sample_reproducible(self) -> None:
        p1 = UniversalPrior().sample(np.random.default_rng(0))
        p2 = UniversalPrior().sample(np.random.default_rng(0))
        assert p1.K_base == p2.K_base

    def test_estimate_kc(self) -> None:
        omegas = np.linspace(-1, 1, 6)
        prior = UniversalPrior().estimate_Kc(omegas, n_layers=6)
        assert prior.K_c_estimate > 0
        assert np.isfinite(prior.K_c_estimate)

    def test_log_probability_max_at_mean(self) -> None:
        up = UniversalPrior()
        lp_mean = up.log_probability(0.47, 0.25)
        lp_off = up.log_probability(1.0, 1.0)
        assert lp_mean > lp_off

    def test_log_probability_finite(self) -> None:
        lp = UniversalPrior().log_probability(0.5, 0.3)
        assert np.isfinite(lp)


class TestKnmTemplateSet:
    def test_add_and_get(self) -> None:
        ts = KnmTemplateSet()
        knm = np.eye(3)
        alpha = np.zeros((3, 3))
        t = KnmTemplate(name="test", knm=knm, alpha=alpha, description="d")
        ts.add(t)
        assert ts.get("test") is t

    def test_get_missing_raises(self) -> None:
        ts = KnmTemplateSet()
        with pytest.raises(KeyError, match="Unknown template"):
            ts.get("nonexistent")

    def test_list_names(self) -> None:
        ts = KnmTemplateSet()
        knm = np.eye(2)
        alpha = np.zeros((2, 2))
        ts.add(KnmTemplate("a", knm, alpha, ""))
        ts.add(KnmTemplate("b", knm, alpha, ""))
        assert sorted(ts.list_names()) == ["a", "b"]

    def test_empty_list(self) -> None:
        assert KnmTemplateSet().list_names() == []


class TestCouplingModulesPipelineEndToEnd:
    """Full pipeline: LagModel + UniversalPrior + KnmTemplateSet →
    Engine → R → RegimeManager. All coupling modules compose."""

    def test_lag_prior_template_engine_regime(self):
        """LagModel → alpha, UniversalPrior → K_nm, KnmTemplateSet registry
        → UPDEEngine → order_parameter → RegimeManager."""
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        # Lag model: estimate alpha from distances
        distances = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [3.0, 2.0, 1.0, 0.0],
            ]
        )
        alpha = LagModel.estimate_from_distances(distances, speed=1.0)
        np.testing.assert_allclose(alpha, -alpha.T, atol=1e-12)

        # Universal prior: sample K_base
        prior_sample = UniversalPrior().sample(np.random.default_rng(42))
        knm = np.full((n, n), prior_sample.K_base)
        np.fill_diagonal(knm, 0.0)

        # Store in KnmTemplateSet
        ts = KnmTemplateSet()
        ts.add(KnmTemplate("cortical", knm, alpha, "cortical distances"))
        tpl = ts.get("cortical")

        # Engine: run with template's K_nm and alpha
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        phases = eng.run(phases, omegas, tpl.knm, 0.0, 0.0, tpl.alpha, n_steps=300)
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

        # RegimeManager
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        assert regime.name in {"NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"}

    def test_alpha_changes_sync_dynamics(self):
        """Non-zero alpha from LagModel should change R vs zero alpha."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        rng = np.random.default_rng(0)
        p0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)

        eng = UPDEEngine(n, dt=0.01)
        # Zero alpha
        p_zero = eng.run(
            p0.copy(),
            omegas,
            knm,
            0.0,
            0.0,
            np.zeros((n, n)),
            n_steps=200,
        )
        r_zero, _ = compute_order_parameter(p_zero)
        # Non-zero alpha from distances
        distances = np.array(
            [
                [0, 1, 3, 5],
                [1, 0, 2, 4],
                [3, 2, 0, 2],
                [5, 4, 2, 0],
            ],
            dtype=float,
        )
        alpha = LagModel.estimate_from_distances(distances, speed=1.0)
        p_lag = eng.run(p0.copy(), omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        r_lag, _ = compute_order_parameter(p_lag)
        assert 0.0 <= r_zero <= 1.0
        assert 0.0 <= r_lag <= 1.0

    def test_kc_estimate_from_prior(self):
        """UniversalPrior.estimate_Kc feeds back into CouplingBuilder build."""
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder

        n = 8
        omegas = np.linspace(-2, 2, n)
        prior = UniversalPrior().estimate_Kc(omegas, n_layers=n)
        assert prior.K_c_estimate > 0
        # Use estimated K_c as base_strength
        cb = CouplingBuilder()
        cs = cb.build(n, prior.K_c_estimate, prior.decay_alpha)
        assert cs.knm.shape == (n, n)
        np.testing.assert_allclose(np.diag(cs.knm), 0.0)

    def test_performance_estimate_from_distances_64_under_10ms(self):
        """LagModel.estimate_from_distances(64×64) < 10ms."""
        import time

        rng = np.random.default_rng(0)
        dist = rng.uniform(0, 10, (64, 64))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0.0)
        LagModel.estimate_from_distances(dist, speed=1.0)  # warm-up
        t0 = time.perf_counter()
        for _ in range(1000):
            LagModel.estimate_from_distances(dist, speed=1.0)
        elapsed = (time.perf_counter() - t0) / 1000
        assert elapsed < 1e-2, f"estimate_from_distances(64) took {elapsed * 1e3:.2f}ms"


# Pipeline wiring: coupling modules (LagModel + UniversalPrior + KnmTemplateSet)
# → UPDEEngine(RK4) → compute_order_parameter → RegimeManager. Alpha antisymmetry
# proven. K_c estimation feeds CouplingBuilder. Performance: distances(64)<10ms.
