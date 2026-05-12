# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Rotating machinery domainpack tests

from __future__ import annotations

import numpy as np
import pytest

from domainpacks.rotating_machinery import run as rotating_machinery_run
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.coupling.knm import CouplingState


def test_rotating_machinery_policy_alpha_action_updates_blade_damper_lag():
    coupling = CouplingState(
        knm=np.ones((10, 10), dtype=float),
        alpha=np.zeros((10, 10), dtype=float),
        active_template="base",
        knm_r=None,
    )
    action = ControlAction(
        knob="alpha",
        scope="layer_1",
        value=0.4,
        ttl_s=5.0,
        justification="policy flutter suppression",
    )

    updated, zeta = rotating_machinery_run._apply_rotating_machinery_action(
        coupling,
        action,
        zeta=0.04,
        layer_map={0: [0, 1, 2], 1: [3, 4], 2: [5, 6, 7], 3: [8, 9]},
    )

    assert updated.alpha[3, 4] == pytest.approx(0.4)
    assert updated.alpha[4, 3] == pytest.approx(0.4)
    assert np.count_nonzero(updated.alpha[:3, :]) == 0
    assert np.count_nonzero(updated.alpha[:, :3]) == 0
    assert np.count_nonzero(updated.alpha[5:, :]) == 0
    assert np.count_nonzero(updated.alpha[:, 5:]) == 0
    assert np.array_equal(updated.knm, coupling.knm)
    assert zeta == pytest.approx(0.04)
