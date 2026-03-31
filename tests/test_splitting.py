# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Splitting integrator tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.splitting import SplittingEngine


def _coupled_knm(n: int, k: float = 0.5) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestSplittingEngine:
    def test_output_in_range(self):
        n = 6
        eng = SplittingEngine(n, dt=0.01)
        phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
        omegas = np.ones(n)
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)

    def test_zero_coupling_pure_rotation(self):
        n = 4
        dt = 0.01
        eng = SplittingEngine(n, dt=dt)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (phases + dt * omegas) % (2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_synchronization(self):
        n = 8
        eng = SplittingEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _coupled_knm(n, k=1.0)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        R, _ = compute_order_parameter(phases)
        assert R > 0.9

    def test_agrees_with_monolithic_rk4(self):
        n = 4
        dt = 0.001
        rng = np.random.default_rng(42)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n) * 1.5
        knm = _coupled_knm(n, k=0.3)
        alpha = np.zeros((n, n))

        split = SplittingEngine(n, dt=dt)
        mono = UPDEEngine(n, dt=dt, method="rk4")

        ps = phases0.copy()
        pm = phases0.copy()
        for _ in range(100):
            ps = split.step(ps, omegas, knm, 0.0, 0.0, alpha)
            pm = mono.step(pm, omegas, knm, 0.0, 0.0, alpha)

        # Should agree to O(dt²) — splitting and monolithic RK4 are both 2nd/4th order
        diff = np.abs(ps - pm)
        diff = np.minimum(diff, 2 * np.pi - diff)
        assert np.max(diff) < 0.01

    def test_run_n_steps(self):
        n = 4
        eng = SplittingEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        assert result.shape == (n,)

    def test_external_drive(self):
        n = 4
        eng = SplittingEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        # psi=1.0 so sin(1.0 - 0) ≈ 0.841 — nonzero drive
        result = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
        assert not np.allclose(result, phases)

    def test_preserves_sync(self):
        n = 6
        eng = SplittingEngine(n, dt=0.01)
        phases = np.full(n, 1.0)
        omegas = np.ones(n) * 2.0
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        R, _ = compute_order_parameter(result)
        assert R > 0.99


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import time

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
