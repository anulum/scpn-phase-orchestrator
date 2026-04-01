# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Active Inference Agent tests

from __future__ import annotations

import numpy as np
import pytest

from spo_kernel import PyActiveInferenceAgent

class TestActiveInferenceAgent:
    def test_agent_adjusts_zeta(self):
        # Target R = 0.5 (metastability)
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=2.0)
        
        # Case 1: R too low (0.1 < 0.5) -> should encourage sync
        zeta, psi = agent.control(r_obs=0.1, psi_obs=0.0, dt=0.01)
        # error = 0.1 - 0.5 = -0.4
        # zeta should be > 0
        assert zeta > 0
        # psi should align with global phase
        assert psi == 0.0
        
        # Case 2: R too high (0.9 > 0.5) -> should suppress sync
        zeta_high, psi_high = agent.control(r_obs=0.9, psi_obs=0.0, dt=0.01)
        # error = 0.9 - 0.5 = 0.4
        assert zeta_high > 0
        # psi should be anti-phase (pi)
        assert abs(psi_high - np.pi) < 1e-10

    def test_target_r_property(self):
        agent = PyActiveInferenceAgent(target_r=0.8)
        assert agent.target_r == 0.8
        agent.target_r = 0.3
        assert agent.target_r == 0.3
