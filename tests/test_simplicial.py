# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Simplicial coupling tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine


def _complete_knm(n: int, k: float = 0.5) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestSimplicialEngine:
    def test_sigma2_zero_is_standard_kuramoto(self):
        n = 4
        eng = SimplicialEngine(n, dt=0.01, sigma2=0.0)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _complete_knm(n)
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (n,)
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)

    def test_three_body_changes_dynamics(self):
        n = 6
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _complete_knm(n)
        alpha = np.zeros((n, n))

        eng0 = SimplicialEngine(n, dt=0.01, sigma2=0.0)
        eng1 = SimplicialEngine(n, dt=0.01, sigma2=0.5)
        r0 = eng0.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r1 = eng1.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert not np.allclose(r0, r1)

    def test_synchronization_with_three_body(self):
        n = 8
        eng = SimplicialEngine(n, dt=0.01, sigma2=0.3)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _complete_knm(n, k=1.0)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        R, _ = compute_order_parameter(result)
        assert R > 0.8

    def test_three_body_brute_force_parity(self):
        """Verify vectorized 3-body matches naive triple loop."""
        n = 4
        rng = np.random.default_rng(42)
        theta = rng.uniform(0, 2 * np.pi, n)
        sigma2 = 0.5

        # Brute force
        three_body_bf = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(n):
                for k in range(n):
                    s += np.sin(theta[j] + theta[k] - 2 * theta[i])
            three_body_bf[i] = sigma2 * s / (n * n)

        # Vectorized (from engine)
        three_body_vec = np.zeros(n)
        diff_ji = theta[np.newaxis, :] - theta[:, np.newaxis]
        for i in range(n):
            d = diff_ji[i, :]
            S = np.sum(np.sin(d))
            C = np.sum(np.cos(d))
            three_body_vec[i] = sigma2 * 2 * S * C / (n * n)

        np.testing.assert_allclose(three_body_vec, three_body_bf, atol=1e-10)

    def test_sigma2_setter(self):
        eng = SimplicialEngine(4, dt=0.01, sigma2=0.0)
        eng.sigma2 = 1.0
        assert eng.sigma2 == 1.0

    def test_small_n_no_three_body(self):
        eng = SimplicialEngine(2, dt=0.01, sigma2=1.0)
        phases = np.array([0.0, 1.0])
        omegas = np.ones(2)
        knm = np.array([[0.0, 0.5], [0.5, 0.0]])
        alpha = np.zeros((2, 2))
        # n < 3, three-body term should be skipped
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert result.shape == (2,)

    def test_run_n_steps(self):
        n = 5
        eng = SimplicialEngine(n, dt=0.01, sigma2=0.2)
        result = eng.run(
            np.zeros(n),
            np.ones(n),
            _complete_knm(n),
            0.0,
            0.0,
            np.zeros((n, n)),
            n_steps=50,
        )
        assert result.shape == (n,)
