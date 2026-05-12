# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Laser array domainpack tests

from __future__ import annotations

import numpy as np
import pytest

from domainpacks.laser_array import run as laser_array_run
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.coupling.knm import CouplingState


def test_laser_array_policy_alpha_action_updates_single_laser_detuning_lag():
    coupling = CouplingState(
        knm=np.ones((8, 8), dtype=float),
        alpha=np.zeros((8, 8), dtype=float),
        active_template="base",
        knm_r=None,
    )
    action = ControlAction(
        knob="alpha",
        scope="layer_0",
        value=0.3,
        ttl_s=5.0,
        justification="policy feedback suppression",
    )

    updated, zeta, psi_target = laser_array_run._apply_laser_array_action(
        coupling,
        action,
        zeta=0.05,
        psi_target=0.0,
        layer_map={0: [0, 1, 2, 3], 1: [4, 5], 2: [6, 7]},
    )

    for i in [0, 1, 2, 3]:
        for j in [0, 1, 2, 3]:
            if i == j:
                assert updated.alpha[i, j] == pytest.approx(0.0)
            else:
                assert updated.alpha[i, j] == pytest.approx(0.3)
    assert np.count_nonzero(updated.alpha[4:, :]) == 0
    assert np.count_nonzero(updated.alpha[:, 4:]) == 0
    assert np.array_equal(updated.knm, coupling.knm)
    assert zeta == pytest.approx(0.05)
    assert psi_target == pytest.approx(0.0)
