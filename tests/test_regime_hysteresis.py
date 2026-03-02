# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_state(r_values, regime_id="nominal"):
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(len(r_values)),
        stability_proxy=min(r_values) if r_values else 0.0,
        regime_id=regime_id,
    )


def _clean_boundary():
    return BoundaryState()


def _hard_violation():
    return BoundaryState(
        violations=["R below floor"],
        hard_violations=["R below floor"],
    )


def test_nominal_when_r_high():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.9, 0.8, 0.85])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.NOMINAL


def test_critical_on_hard_violation():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.9, 0.8])
    regime = mgr.evaluate(state, _hard_violation())
    assert regime == Regime.CRITICAL


def test_degraded_when_r_low():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.4, 0.45, 0.5])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.DEGRADED


def test_critical_when_r_very_low():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([0.1, 0.15, 0.2])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.CRITICAL


def test_cooldown_prevents_rapid_transition():
    mgr = RegimeManager(cooldown_steps=5)
    result1 = mgr.transition(Regime.DEGRADED)
    assert result1 == Regime.DEGRADED
    result2 = mgr.transition(Regime.NOMINAL)
    assert result2 == Regime.DEGRADED


def test_always_escalate_to_critical():
    mgr = RegimeManager(cooldown_steps=100)
    mgr.transition(Regime.DEGRADED)
    result = mgr.transition(Regime.CRITICAL)
    assert result == Regime.CRITICAL


def test_recovery_when_current_is_critical():
    mgr = RegimeManager(cooldown_steps=0)
    mgr._current = Regime.CRITICAL
    # R in [R_CRITICAL, R_DEGRADED) → Recovery
    state = _make_state([0.45, 0.50])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.RECOVERY


def test_critical_to_nominal_when_r_exceeds_band():
    mgr = RegimeManager(cooldown_steps=0)
    mgr._current = Regime.CRITICAL
    # R well above R_DEGRADED + hysteresis → Nominal
    state = _make_state([0.7, 0.75])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.NOMINAL


def test_mean_r_empty_layers():
    mgr = RegimeManager(cooldown_steps=0)
    state = _make_state([])
    regime = mgr.evaluate(state, _clean_boundary())
    assert regime == Regime.CRITICAL
