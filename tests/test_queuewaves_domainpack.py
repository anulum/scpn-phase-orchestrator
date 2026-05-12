# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - QueueWaves domainpack tests

from __future__ import annotations

import numpy as np
import pytest

from domainpacks.queuewaves import run as queuewaves_run
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.coupling.knm import CouplingState


def test_queuewaves_policy_alpha_action_updates_micro_lag_matrix():
    coupling = CouplingState(
        knm=np.ones((6, 6), dtype=float),
        alpha=np.zeros((6, 6), dtype=float),
        active_template="base",
        knm_r=None,
    )
    action = ControlAction(
        knob="alpha",
        scope="layer_0",
        value=0.3,
        ttl_s=5.0,
        justification="policy retry-storm suppression",
    )

    updated, zeta, psi_target = queuewaves_run._apply_queuewaves_action(
        coupling,
        action,
        zeta=0.1,
        psi_target=0.0,
        layer_map={0: [0, 1, 2], 1: [3], 2: [4, 5]},
    )

    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i == j:
                assert updated.alpha[i, j] == pytest.approx(0.0)
            else:
                assert updated.alpha[i, j] == pytest.approx(0.3)
    assert np.count_nonzero(updated.alpha[3:, :]) == 0
    assert np.count_nonzero(updated.alpha[:, 3:]) == 0
    assert np.array_equal(updated.knm, coupling.knm)
    assert zeta == pytest.approx(0.1)
    assert psi_target == pytest.approx(0.0)
