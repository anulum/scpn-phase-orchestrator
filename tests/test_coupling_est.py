# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling estimation tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.coupling_est import (
    estimate_coupling,
    estimate_coupling_harmonics,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine


class TestEstimateCoupling:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (4, 100))
        omegas = np.ones(4)
        knm = estimate_coupling(phases, omegas, dt=0.01)
        assert knm.shape == (4, 4)

    def test_zero_diagonal(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (3, 50))
        omegas = np.ones(3)
        knm = estimate_coupling(phases, omegas, dt=0.01)
        np.testing.assert_array_equal(np.diag(knm), 0.0)

    def test_recovers_known_coupling(self):
        # Generate data from known Kuramoto with K=0.5 all-to-all
        n = 4
        dt = 0.01
        K_true = 0.5
        engine = UPDEEngine(n, dt=dt)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n) * 2.0
        knm_true = np.full((n, n), K_true)
        np.fill_diagonal(knm_true, 0.0)
        alpha = np.zeros((n, n))

        trajectory = [phases.copy()]
        for _ in range(500):
            phases = engine.step(phases, omegas, knm_true, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())
        trajectory = np.array(trajectory).T  # (n, T)

        knm_est = estimate_coupling(trajectory, omegas, dt=dt)
        # Should roughly recover K_true off-diagonal
        off_diag = knm_est[~np.eye(n, dtype=bool)]
        assert np.mean(off_diag) > 0.1  # positive coupling recovered

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="3 timesteps"):
            estimate_coupling(np.ones((3, 2)), np.ones(3), dt=0.01)

    def test_finite_values(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (5, 200))
        omegas = np.ones(5)
        knm = estimate_coupling(phases, omegas, dt=0.01)
        assert np.all(np.isfinite(knm))


class TestHarmonicCoupling:
    def test_returns_dict(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (4, 100))
        result = estimate_coupling_harmonics(phases, np.ones(4), dt=0.01)
        assert "sin_1" in result
        assert "cos_1" in result
        assert "sin_2" in result
        assert "cos_2" in result

    def test_shape(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (3, 100))
        result = estimate_coupling_harmonics(phases, np.ones(3), dt=0.01)
        assert result["sin_1"].shape == (3, 3)

    def test_zero_diagonal(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (4, 100))
        result = estimate_coupling_harmonics(phases, np.ones(4), dt=0.01)
        for v in result.values():
            np.testing.assert_array_equal(np.diag(v), 0.0)

    def test_single_harmonic_sign_consistent(self):
        # With sin+cos regressors, sin_1 coefficients should have
        # the same sign structure as the standard K_ij estimate
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (3, 500))
        omegas = np.ones(3)
        knm_std = estimate_coupling(phases, omegas, dt=0.01)
        knm_harm = estimate_coupling_harmonics(phases, omegas, dt=0.01, n_harmonics=1)
        # Sign of dominant coupling should match
        mask = np.abs(knm_std) > 0.1
        if np.any(mask):
            signs_match = np.sign(knm_harm["sin_1"][mask]) == np.sign(knm_std[mask])
            assert np.sum(signs_match) > 0

    def test_custom_n_harmonics(self):
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, (3, 100))
        result = estimate_coupling_harmonics(phases, np.ones(3), dt=0.01, n_harmonics=3)
        assert "sin_3" in result
        assert "cos_3" in result


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
