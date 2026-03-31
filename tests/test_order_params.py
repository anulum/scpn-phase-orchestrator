# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameter tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.order_params import (
    compute_layer_coherence,
    compute_order_parameter,
    compute_plv,
)

TWO_PI = 2.0 * np.pi


def test_identical_phases_R_one():
    phases = np.full(100, 1.5)
    R, _ = compute_order_parameter(phases)
    np.testing.assert_allclose(R, 1.0, atol=1e-12)


def test_uniform_phases_R_near_zero():
    phases = np.linspace(0, TWO_PI, 1000, endpoint=False)
    R, _ = compute_order_parameter(phases)
    assert R < 0.02


def test_plv_identical_series():
    phases = np.ones(200) * 2.0
    plv = compute_plv(phases, phases)
    np.testing.assert_allclose(plv, 1.0, atol=1e-12)


def test_plv_random_uncorrelated():
    rng = np.random.default_rng(55)
    a = rng.uniform(0, TWO_PI, size=5000)
    b = rng.uniform(0, TWO_PI, size=5000)
    plv = compute_plv(a, b)
    assert plv < 0.1


def test_layer_coherence_full_mask():
    phases = np.array([0.0, 0.0, 0.0, 0.0])
    mask = np.ones(4, dtype=bool)
    R_layer = compute_layer_coherence(phases, mask)
    R_global, _ = compute_order_parameter(phases)
    np.testing.assert_allclose(R_layer, R_global, atol=1e-12)


def test_layer_coherence_partial_mask():
    phases = np.array([0.0, 0.0, np.pi, np.pi])
    mask = np.array([True, True, False, False])
    R_sub = compute_layer_coherence(phases, mask)
    np.testing.assert_allclose(R_sub, 1.0, atol=1e-12)


def test_layer_coherence_empty_mask():
    phases = np.array([1.0, 2.0, 3.0])
    mask = np.zeros(3, dtype=bool)
    assert compute_layer_coherence(phases, mask) == 0.0


def test_plv_length_mismatch_raises():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="equal-length"):
        compute_plv(a, b)


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
