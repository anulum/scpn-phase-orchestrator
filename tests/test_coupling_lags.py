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
