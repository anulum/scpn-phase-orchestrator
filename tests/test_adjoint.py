# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adjoint gradient tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.adjoint import cost_R, gradient_knm_fd
from scpn_phase_orchestrator.upde.engine import UPDEEngine


@pytest.fixture()
def small_system():
    N = 4
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, N)
    omegas = rng.normal(0, 0.5, N)
    knm = rng.uniform(0.1, 0.5, (N, N))
    np.fill_diagonal(knm, 0)
    alpha = np.zeros((N, N))
    engine = UPDEEngine(N, dt=0.01)
    return engine, phases, omegas, knm, alpha


class TestCostR:
    def test_synchronized_low_cost(self):
        phases = np.zeros(8)
        assert cost_R(phases) == pytest.approx(0.0, abs=1e-10)

    def test_uniform_high_cost(self):
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        assert cost_R(phases) > 0.9

    def test_range(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 16)
            c = cost_R(phases)
            assert 0.0 <= c <= 1.0


class TestGradientFD:
    def test_shape(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=10)
        assert grad.shape == knm.shape

    def test_zero_diagonal(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=10)
        np.testing.assert_array_equal(np.diag(grad), 0.0)

    def test_gradient_not_all_zero(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=20)
        assert np.any(grad != 0.0)

    def test_gradient_direction(self, small_system):
        engine, phases, omegas, knm, alpha = small_system
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=50)
        knm_new = knm - 0.1 * grad
        np.fill_diagonal(knm_new, 0)
        p_old = engine.run(phases, omegas, knm, 0, 0, alpha, 50)
        p_new = engine.run(phases, omegas, knm_new, 0, 0, alpha, 50)
        assert cost_R(p_new) <= cost_R(p_old) + 0.1

    def test_zero_coupling(self):
        N = 3
        engine = UPDEEngine(N, dt=0.01)
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.zeros(N)
        knm = np.zeros((N, N))
        alpha = np.zeros((N, N))
        grad = gradient_knm_fd(engine, phases, omegas, knm, alpha, n_steps=5)
        assert grad.shape == (N, N)


class TestGradientJAX:
    def test_jax_import_error(self):
        from scpn_phase_orchestrator.upde.adjoint import gradient_knm_jax

        try:
            import jax  # noqa: F401

            pytest.skip("JAX is installed")
        except ImportError:
            with pytest.raises(ImportError, match="JAX is required"):
                gradient_knm_jax(
                    np.zeros(4), np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4))
                )
