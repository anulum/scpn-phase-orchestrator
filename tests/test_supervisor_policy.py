# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(rs: list[float]) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0) for r in rs],
        cross_layer_alignment=np.zeros((len(rs), len(rs))),
        stability_proxy=sum(rs) / max(len(rs), 1),
        regime_id="test",
    )


_NO_VIOLATIONS = BoundaryState()


def test_nominal_returns_no_actions():
    rm = RegimeManager()
    policy = SupervisorPolicy(rm)
    actions = policy.decide(_state([0.9, 0.95]), _NO_VIOLATIONS)
    assert actions == []


def test_degraded_boosts_coupling():
    rm = RegimeManager()
    policy = SupervisorPolicy(rm)
    actions = policy.decide(_state([0.5, 0.5]), _NO_VIOLATIONS)
    assert len(actions) == 1
    assert actions[0].knob == "K"
    assert actions[0].value > 0


def test_critical_increases_zeta_and_reduces_k():
    rm = RegimeManager()
    policy = SupervisorPolicy(rm)
    actions = policy.decide(_state([0.1, 0.2]), _NO_VIOLATIONS)
    knobs = {a.knob for a in actions}
    assert "zeta" in knobs
    assert "K" in knobs
    k_action = next(a for a in actions if a.knob == "K")
    assert k_action.value < 0  # reduction


def test_critical_worst_layer_targeted():
    rm = RegimeManager()
    policy = SupervisorPolicy(rm)
    actions = policy.decide(_state([0.1, 0.25]), _NO_VIOLATIONS)
    k_action = next(a for a in actions if a.knob == "K")
    assert k_action.scope == "layer_0"  # layer 0 has R=0.1 (worst)


def test_recovery_gradual_restore():
    rm = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(rm)
    # Force into critical first
    policy.decide(_state([0.1, 0.1]), _NO_VIOLATIONS)
    # Now mid-range R triggers recovery path (cooldown=0 allows immediate transition)
    actions = policy.decide(_state([0.5, 0.5]), _NO_VIOLATIONS)
    assert len(actions) == 1
    assert actions[0].knob == "K"
    assert actions[0].value > 0  # gradual restore


def test_empty_layers_triggers_critical():
    rm = RegimeManager()
    policy = SupervisorPolicy(rm)
    actions = policy.decide(_state([]), _NO_VIOLATIONS)
    knobs = {a.knob for a in actions}
    assert "zeta" in knobs


def test_hard_violation_forces_critical():
    rm = RegimeManager()
    policy = SupervisorPolicy(rm)
    bs = BoundaryState(hard_violations=["pressure_high"])
    actions = policy.decide(_state([0.9, 0.9]), bs)
    assert len(actions) >= 1
    assert any(a.knob == "zeta" for a in actions)
