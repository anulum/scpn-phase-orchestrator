# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Three-factor plasticity tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)


def test_eligibility_synchronised_phases():
    """Identical phases → cos(0) = 1 off-diagonal, 0 on diagonal."""
    phases = np.zeros(5)
    elig = compute_eligibility(phases)
    expected = np.ones((5, 5))
    np.fill_diagonal(expected, 0.0)
    np.testing.assert_allclose(elig, expected, atol=1e-12)


def test_eligibility_antiphase():
    """Two oscillators at 0 and pi → cos(pi) = -1."""
    phases = np.array([0.0, np.pi])
    elig = compute_eligibility(phases)
    assert elig[0, 0] == 0.0
    assert elig[1, 1] == 0.0
    np.testing.assert_allclose(elig[0, 1], -1.0, atol=1e-12)
    np.testing.assert_allclose(elig[1, 0], -1.0, atol=1e-12)


def test_eligibility_diagonal_zero():
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, 20)
    elig = compute_eligibility(phases)
    np.testing.assert_allclose(np.diag(elig), 0.0, atol=1e-15)


def test_three_factor_gate_off_no_update():
    knm = np.ones((3, 3))
    elig = np.ones((3, 3))
    result = three_factor_update(knm, elig, modulator=1.0, phase_gate=False, lr=0.1)
    np.testing.assert_array_equal(result, knm)


def test_three_factor_positive_modulator_increases_coupling():
    knm = np.zeros((4, 4))
    elig = np.ones((4, 4))
    np.fill_diagonal(elig, 0.0)
    result = three_factor_update(knm, elig, modulator=1.0, phase_gate=True, lr=0.05)
    expected = 0.05 * elig
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_three_factor_negative_modulator_decreases_coupling():
    knm = np.ones((3, 3))
    elig = np.ones((3, 3))
    result = three_factor_update(knm, elig, modulator=-1.0, phase_gate=True, lr=0.1)
    assert np.all(result < knm)


def test_three_factor_does_not_mutate_input():
    knm = np.ones((3, 3))
    original = knm.copy()
    elig = np.ones((3, 3))
    three_factor_update(knm, elig, modulator=1.0, phase_gate=True, lr=0.1)
    np.testing.assert_array_equal(knm, original)


def test_three_factor_zero_modulator_no_change():
    knm = np.ones((5, 5)) * 0.5
    elig = np.ones((5, 5))
    result = three_factor_update(knm, elig, modulator=0.0, phase_gate=True, lr=0.1)
    np.testing.assert_allclose(result, knm, atol=1e-15)


def test_eligibility_symmetry():
    """cos(θ_j - θ_i) = cos(θ_i - θ_j) → eligibility is symmetric."""
    rng = np.random.default_rng(123)
    phases = rng.uniform(0, 2 * np.pi, 10)
    elig = compute_eligibility(phases)
    np.testing.assert_allclose(elig, elig.T, atol=1e-12)


class TestPlasticityPipelineWiring:
    """Pipeline: engine phases → eligibility → three-factor → updated K_nm."""

    def test_plasticity_loop_changes_coupling(self):
        """UPDEEngine → phases → eligibility → three_factor_update →
        engine uses updated K_nm. Proves plasticity isn't decorative."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        # Run 50 steps, then apply plasticity, then run 50 more
        for _ in range(50):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        elig = compute_eligibility(phases)
        knm_updated = three_factor_update(
            knm,
            elig,
            modulator=1.0,
            phase_gate=True,
            lr=0.01,
        )
        assert not np.allclose(knm, knm_updated), "Plasticity must change K_nm"

        for _ in range(50):
            phases = eng.step(phases, omegas, knm_updated, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
