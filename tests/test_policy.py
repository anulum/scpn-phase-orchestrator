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


def _make_upde(r_values):
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    n = len(r_values)
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(n) if n else np.empty((0, 0)),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id="nominal",
    )


def test_nominal_no_actions():
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    state = _make_upde([0.9, 0.85, 0.88])
    boundary = BoundaryState()
    actions = policy.decide(state, boundary)
    assert actions == []


def test_degraded_k_increase():
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    state = _make_upde([0.4, 0.45, 0.5])
    boundary = BoundaryState()
    actions = policy.decide(state, boundary)
    assert len(actions) >= 1
    assert actions[0].knob == "K"
    assert actions[0].value > 0


def test_critical_zeta_and_k_reduction():
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    state = _make_upde([0.1, 0.15, 0.2])
    boundary = BoundaryState()
    actions = policy.decide(state, boundary)
    knobs = [a.knob for a in actions]
    assert "zeta" in knobs
    k_actions = [a for a in actions if a.knob == "K"]
    assert any(a.value < 0 for a in k_actions)


def test_critical_via_hard_violation():
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    state = _make_upde([0.9, 0.85])
    boundary = BoundaryState(violations=["x"], hard_violations=["x"])
    actions = policy.decide(state, boundary)
    knobs = [a.knob for a in actions]
    assert "zeta" in knobs


def test_recovery_after_critical():
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    # First: drive into CRITICAL
    state_crit = _make_upde([0.1, 0.15])
    policy.decide(state_crit, BoundaryState())
    # R in [R_CRITICAL, R_DEGRADED) while current=CRITICAL → RECOVERY
    state_rec = _make_upde([0.45, 0.50])
    actions = policy.decide(state_rec, BoundaryState())
    assert len(actions) == 1
    assert actions[0].knob == "K"
    assert "recovery" in actions[0].justification


def test_critical_with_empty_layers():
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    state = _make_upde([])
    boundary = BoundaryState()
    actions = policy.decide(state, boundary)
    knobs = [a.knob for a in actions]
    assert "zeta" in knobs
