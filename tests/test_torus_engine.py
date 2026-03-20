# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus engine tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


class TestTorusEngine:
    def test_output_on_torus(self):
        n = 6
        eng = TorusEngine(n, dt=0.01)
        phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)

    def test_pure_rotation_exact(self):
        n = 4
        dt = 0.01
        eng = TorusEngine(n, dt=dt)
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (phases + dt * omegas) % (2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_wrapping_smooth(self):
        n = 2
        eng = TorusEngine(n, dt=0.1)
        # Phase near 2π should wrap smoothly
        phases = np.array([2 * np.pi - 0.05, 0.05])
        omegas = np.array([1.0, -1.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)

    def test_synchronization(self):
        n = 8
        eng = TorusEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = np.full((n, n), 1.0)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        R, _ = compute_order_parameter(result)
        assert R > 0.9

    def test_run_returns_correct_shape(self):
        n = 5
        eng = TorusEngine(n, dt=0.01)
        result = eng.run(
            np.zeros(n), np.ones(n),
            np.zeros((n, n)), 0.0, 0.0, np.zeros((n, n)),
            n_steps=10,
        )
        assert result.shape == (n,)

    def test_preserves_sync(self):
        n = 4
        eng = TorusEngine(n, dt=0.01)
        phases = np.full(n, 2.0)
        omegas = np.ones(n)
        knm = np.full((n, n), 0.5)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        R, _ = compute_order_parameter(result)
        assert R > 0.99
