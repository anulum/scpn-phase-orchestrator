# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fault-injection supervisor hardening tests

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


class _FaultyPetriAdapter:
    def __init__(self, fail_at: int, nominal_return: Regime) -> None:
        self._fail_at = fail_at
        self._nominal_return = nominal_return
        self._calls = 0

    def step(self, _ctx: dict[str, float]):
        self._calls += 1
        if self._calls == self._fail_at:
            raise RuntimeError("fault injection")
        return self._nominal_return


def _state(r: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=r,
        regime_id="fault-injection",
    )


@given(fail_at=st.integers(min_value=1, max_value=25))
@settings(max_examples=50, deadline=None)
def test_policy_degrade_path_survives_fault_injection(fail_at: int) -> None:
    rm = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(
        rm,
        petri_adapter=_FaultyPetriAdapter(fail_at, Regime.DEGRADED),
    )

    for _ in range(25):
        actions = policy.decide(
            upde_state=_state(0.5),
            boundary_state=BoundaryState(),
            petri_ctx={"stability_proxy": 0.5},
        )
        assert isinstance(actions, list)
        assert actions
        assert actions[0].knob == "K"


@given(fail_at=st.integers(min_value=1, max_value=25))
@settings(max_examples=50, deadline=None)
def test_policy_critical_path_survives_fault_injection(fail_at: int) -> None:
    rm = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(
        rm,
        petri_adapter=_FaultyPetriAdapter(fail_at, Regime.CRITICAL),
    )

    for _ in range(25):
        actions = policy.decide(
            upde_state=_state(0.1),
            boundary_state=BoundaryState(),
            petri_ctx={"stability_proxy": 0.1},
        )
        assert isinstance(actions, list)
        knobs = {a.knob for a in actions}
        assert "zeta" in knobs
