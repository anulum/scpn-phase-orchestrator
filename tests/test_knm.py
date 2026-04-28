# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Knm builder tests

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder, CouplingState


def test_symmetric():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=8, base_strength=0.5, decay_alpha=0.3)
    np.testing.assert_allclose(cs.knm, cs.knm.T, atol=1e-14)


def test_zero_diagonal():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=6, base_strength=1.0, decay_alpha=0.1)
    np.testing.assert_allclose(np.diag(cs.knm), 0.0, atol=1e-15)


def test_coupling_decays_with_distance():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=8, base_strength=0.5, decay_alpha=0.3)
    # K(0,1) > K(0,3) > K(0,7)
    assert cs.knm[0, 1] > cs.knm[0, 3]
    assert cs.knm[0, 3] > cs.knm[0, 7]


def test_non_negative():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=10, base_strength=0.5, decay_alpha=0.5)
    assert np.all(cs.knm >= 0.0)


def test_default_template_name():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    assert cs.active_template == "default"


def test_switch_template():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    alt_knm = np.eye(4) * 0.0 + 0.1
    np.fill_diagonal(alt_knm, 0.0)
    templates = {"alt": alt_knm}
    cs2 = builder.switch_template(cs, "alt", templates)
    assert cs2.active_template == "alt"
    np.testing.assert_allclose(cs2.knm, alt_knm)


def test_switch_to_missing_template_raises():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=4, base_strength=0.1, decay_alpha=0.1)
    with pytest.raises(KeyError, match="notfound"):
        builder.switch_template(cs, "notfound", {})


def test_alpha_initialized_to_zero():
    builder = CouplingBuilder()
    cs = builder.build(n_layers=5, base_strength=0.3, decay_alpha=0.2)
    np.testing.assert_allclose(cs.alpha, 0.0)


def test_coupling_state_frozen():
    cs = CouplingState(knm=np.eye(3), alpha=np.zeros((3, 3)), active_template="default")
    with pytest.raises(dataclasses.FrozenInstanceError):
        cs.active_template = "other"


class TestKnmPipelineWiring:
    """Pipeline: CouplingBuilder → K_nm → engine → R."""

    def test_built_knm_drives_engine(self):
        """CouplingBuilder.build → K_nm → engine → R∈[0,1].
        Proves builder output feeds the simulation core."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        cs = CouplingBuilder().build(n, 0.5, 0.3)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        for _ in range(200):
            phases = eng.step(
                phases,
                omegas,
                cs.knm,
                0.0,
                0.0,
                cs.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
