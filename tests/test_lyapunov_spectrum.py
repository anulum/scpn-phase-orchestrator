# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov spectrum tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.lyapunov import lyapunov_spectrum


class TestLyapunovSpectrum:
    def test_returns_n_exponents(self):
        N = 6
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, N)
        omegas = rng.normal(0, 0.5, N)
        knm = np.ones((N, N)) * 0.5
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((N, N))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=200)
        assert spec.shape == (N,)

    def test_sorted_descending(self):
        N = 8
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, N)
        omegas = rng.normal(0, 1.0, N)
        knm = rng.uniform(0, 0.3, (N, N))
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((N, N))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=300)
        for i in range(len(spec) - 1):
            assert spec[i] >= spec[i + 1] - 1e-10

    def test_synchronized_negative(self):
        """Strongly coupled → all exponents should be non-positive (stable)."""
        N = 4
        phases = np.zeros(N)
        omegas = np.zeros(N)
        knm = np.ones((N, N)) * 2.0
        np.fill_diagonal(knm, 0)
        alpha = np.zeros((N, N))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=500)
        assert spec[0] <= 0.1  # max exponent near zero or negative

    def test_uncoupled_zero(self):
        """Zero coupling → all exponents should be ~0 (neutral stability)."""
        N = 4
        rng = np.random.default_rng(99)
        phases = rng.uniform(0, 2 * np.pi, N)
        omegas = rng.normal(0, 0.5, N)
        knm = np.zeros((N, N))
        alpha = np.zeros((N, N))
        spec = lyapunov_spectrum(phases, omegas, knm, alpha, n_steps=200)
        assert np.all(np.abs(spec) < 0.5)

    def test_small_system(self):
        spec = lyapunov_spectrum(
            np.array([0.0, 1.0]),
            np.array([0.0, 0.0]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.zeros((2, 2)),
            n_steps=100,
        )
        assert spec.shape == (2,)
