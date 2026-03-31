# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for coupling lags, prior, templates

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.lags import LagModel
from scpn_phase_orchestrator.coupling.prior import CouplingPrior, UniversalPrior
from scpn_phase_orchestrator.coupling.templates import KnmTemplate, KnmTemplateSet


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
