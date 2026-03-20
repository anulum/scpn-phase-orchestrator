# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adjoint gradient tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.adjoint import cost_R, gradient_knm_fd
from scpn_phase_orchestrator.upde.engine import UPDEEngine


class TestCostR:
    def test_synced_zero_cost(self):
        phases = np.zeros(4)
        assert cost_R(phases) < 1e-10

    def test_spread_positive_cost(self):
        phases = np.linspace(0, 2 * np.pi, 4, endpoint=False)
        assert cost_R(phases) > 0.5

    def test_range(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 8)
            c = cost_R(phases)
            assert 0.0 <= c <= 1.0


class TestGradientFD:
    def test_returns_correct_shape(self):
        n = 4
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.3)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=20)
        assert grad.shape == (n, n)

    def test_diagonal_is_zero(self):
        n = 3
        engine = UPDEEngine(n, dt=0.01)
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=20)
        np.testing.assert_array_equal(np.diag(grad), 0.0)

    def test_gradient_sign(self):
        # Increasing coupling should decrease cost (increase R)
        # So gradient should be negative (cost decreases with more K)
        n = 4
        engine = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.3)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=50)
        # Most off-diagonal entries should be negative
        off_diag = grad[~np.eye(n, dtype=bool)]
        assert np.mean(off_diag < 0) > 0.5

    def test_finite_values(self):
        n = 3
        engine = UPDEEngine(n, dt=0.01)
        phases = np.array([0.0, 0.5, 1.0])
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=10)
        assert np.all(np.isfinite(grad))
