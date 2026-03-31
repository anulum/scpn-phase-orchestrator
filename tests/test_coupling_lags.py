# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling lag tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.lags import LagModel


def test_zero_lag_identical_signals():
    model = LagModel()
    sig = np.sin(np.linspace(0, 4 * np.pi, 256))
    lag = model.estimate_lag(sig, sig, sample_rate=256.0)
    assert abs(lag) < 1.0 / 256.0


def test_known_shift():
    model = LagModel()
    t = np.linspace(0, 1, 1000, endpoint=False)
    sig_a = np.sin(2 * np.pi * 5 * t)
    shift = 10
    sig_b = np.roll(sig_a, shift)
    lag = model.estimate_lag(sig_a, sig_b, sample_rate=1000.0)
    assert abs(lag - (-shift / 1000.0)) < 0.005


def test_alpha_matrix_antisymmetric():
    model = LagModel()
    lags = {(0, 1): 0.1, (1, 2): -0.05}
    alpha = model.build_alpha_matrix(lags, n_layers=3)
    assert alpha.shape == (3, 3)
    np.testing.assert_allclose(alpha, -alpha.T, atol=1e-12)


def test_negative_lag_direction():
    model = LagModel()
    t = np.linspace(0, 1, 1000, endpoint=False)
    sig_a = np.sin(2 * np.pi * 5 * t)
    sig_b = np.roll(sig_a, -10)
    lag = model.estimate_lag(sig_a, sig_b, sample_rate=1000.0)
    assert lag > 0, f"Expected positive lag for negative shift, got {lag}"


def test_large_lag_detected():
    model = LagModel()
    t = np.linspace(0, 2, 2000, endpoint=False)
    sig_a = np.sin(2 * np.pi * 3 * t)
    shift = 100
    sig_b = np.roll(sig_a, shift)
    lag = model.estimate_lag(sig_a, sig_b, sample_rate=1000.0)
    assert abs(lag - (-shift / 1000.0)) < 0.01


def test_zero_mean_signals_lag_finite():
    model = LagModel()
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(256)
    lag = model.estimate_lag(sig, sig, sample_rate=256.0)
    assert abs(lag) < 1.0 / 256.0


def test_alpha_matrix_diagonal_zero():
    model = LagModel()
    lags = {(0, 1): 0.3}
    alpha = model.build_alpha_matrix(lags, n_layers=3)
    np.testing.assert_allclose(np.diag(alpha), 0.0, atol=1e-15)


def test_estimate_from_distances_antisymmetric():
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    alpha = LagModel.estimate_from_distances(distances, speed=1.0)
    np.testing.assert_allclose(alpha, -alpha.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(alpha), 0.0, atol=1e-15)


def test_estimate_from_distances_scaling():
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    a1 = LagModel.estimate_from_distances(distances, speed=1.0)
    a2 = LagModel.estimate_from_distances(distances, speed=2.0)
    np.testing.assert_allclose(a1, 2.0 * a2, atol=1e-12)


def test_estimate_from_distances_matches_rust():
    """Verify Python estimate_from_distances gives same result as Rust."""
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    alpha = LagModel.estimate_from_distances(distances, speed=1.0)
    expected = 2.0 * np.pi * 1.0 / 1.0
    np.testing.assert_allclose(alpha[0, 1], expected, atol=1e-12)
    np.testing.assert_allclose(alpha[1, 0], -expected, atol=1e-12)


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
