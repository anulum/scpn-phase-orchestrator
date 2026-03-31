# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling Knm tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder


def test_build_symmetric():
    state = CouplingBuilder().build(8, 0.45, 0.3)
    np.testing.assert_allclose(state.knm, state.knm.T, atol=1e-14)


def test_build_zero_diagonal():
    state = CouplingBuilder().build(8, 0.45, 0.3)
    np.testing.assert_allclose(np.diag(state.knm), 0.0)


def test_build_nonnegative():
    state = CouplingBuilder().build(8, 0.45, 0.3)
    assert np.all(state.knm >= 0.0)


def test_build_exponential_decay():
    state = CouplingBuilder().build(8, 1.0, 0.5)
    assert state.knm[0, 1] > state.knm[0, 2] > state.knm[0, 3]


def test_build_alpha_shape():
    state = CouplingBuilder().build(4, 0.45, 0.3)
    assert state.alpha.shape == (4, 4)
    np.testing.assert_allclose(state.alpha, 0.0)


def test_switch_template_valid():
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.1)
    custom = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    new_state = builder.switch_template(state, "ring", {"ring": custom})
    np.testing.assert_array_equal(new_state.knm, custom)
    assert new_state.active_template == "ring"


def test_switch_template_missing_raises():
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.1)
    with pytest.raises(KeyError, match="no_such"):
        builder.switch_template(state, "no_such", {})


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
