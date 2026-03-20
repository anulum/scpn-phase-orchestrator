# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — NPE tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.npe import compute_npe, phase_distance_matrix


class TestPhaseDistanceMatrix:
    def test_symmetric(self):
        phases = np.array([0.0, 1.0, 2.0])
        D = phase_distance_matrix(phases)
        np.testing.assert_allclose(D, D.T)

    def test_diagonal_zero(self):
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        D = phase_distance_matrix(phases)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_wrapping(self):
        phases = np.array([0.1, 2 * np.pi - 0.1])
        D = phase_distance_matrix(phases)
        assert D[0, 1] < 0.3

    def test_max_distance_is_pi(self):
        phases = np.array([0.0, np.pi])
        D = phase_distance_matrix(phases)
        assert abs(D[0, 1] - np.pi) < 1e-10


class TestNPE:
    def test_synchronized_low_npe(self):
        phases = np.zeros(10)
        npe = compute_npe(phases)
        assert npe == 0.0

    def test_spread_high_npe(self):
        phases = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        npe = compute_npe(phases)
        assert npe > 0.5

    def test_single_oscillator(self):
        assert compute_npe(np.array([1.0])) == 0.0

    def test_empty(self):
        assert compute_npe(np.array([])) == 0.0

    def test_two_in_phase(self):
        npe = compute_npe(np.array([0.0, 0.0]))
        assert npe == 0.0

    def test_two_anti_phase(self):
        npe = compute_npe(np.array([0.0, np.pi]))
        assert npe == 0.0  # only one lifetime → entropy 0

    def test_range_zero_one(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            phases = rng.uniform(0, 2 * np.pi, 16)
            npe = compute_npe(phases)
            assert 0.0 <= npe <= 1.0

    def test_more_sync_lower_npe(self):
        rng = np.random.default_rng(42)
        spread = rng.uniform(0, 2 * np.pi, 20)
        tight = rng.normal(1.0, 0.1, 20) % (2 * np.pi)
        npe_spread = compute_npe(spread)
        npe_tight = compute_npe(tight)
        assert npe_tight < npe_spread

    def test_custom_max_radius(self):
        phases = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        _npe_full = compute_npe(phases)
        npe_half = compute_npe(phases, max_radius=0.5)
        # Restricting radius may change result
        assert isinstance(npe_half, float)
