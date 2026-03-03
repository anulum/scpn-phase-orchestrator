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


def _state(r: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=r,
        regime_id="test",
    )


_NO_VIOLATIONS = BoundaryState()


def test_high_r_evaluates_nominal():
    rm = RegimeManager()
    assert rm.evaluate(_state(0.9), _NO_VIOLATIONS) == Regime.NOMINAL


def test_mid_r_evaluates_degraded():
    rm = RegimeManager()
    assert rm.evaluate(_state(0.5), _NO_VIOLATIONS) == Regime.DEGRADED


def test_low_r_evaluates_critical():
    rm = RegimeManager()
    assert rm.evaluate(_state(0.1), _NO_VIOLATIONS) == Regime.CRITICAL


def test_hard_violation_forces_critical():
    rm = RegimeManager()
    bs = BoundaryState(hard_violations=["temp_high: T=500 outside [None, 400]"])
    assert rm.evaluate(_state(0.9), bs) == Regime.CRITICAL


def test_hysteresis_prevents_premature_upgrade():
    rm = RegimeManager(hysteresis=0.05)
    rm.transition(Regime.DEGRADED)
    # Just above R_DEGRADED but within hysteresis band
    proposed = rm.evaluate(_state(0.62), _NO_VIOLATIONS)
    assert proposed == Regime.DEGRADED


def test_cooldown_blocks_non_critical():
    rm = RegimeManager(cooldown_steps=10)
    rm.transition(Regime.DEGRADED)
    result = rm.transition(Regime.NOMINAL)
    assert result == Regime.DEGRADED  # still in cooldown


def test_critical_bypasses_cooldown():
    rm = RegimeManager(cooldown_steps=100)
    rm.transition(Regime.DEGRADED)
    result = rm.transition(Regime.CRITICAL)
    assert result == Regime.CRITICAL


def test_recovery_path_from_critical():
    rm = RegimeManager()
    rm._current = Regime.CRITICAL
    proposed = rm.evaluate(_state(0.5), _NO_VIOLATIONS)
    assert proposed == Regime.RECOVERY


def test_empty_layers_evaluates_critical():
    rm = RegimeManager()
    empty = UPDEState(
        layers=[],
        cross_layer_alignment=np.zeros((0, 0)),
        stability_proxy=0.0,
        regime_id="test",
    )
    assert rm.evaluate(empty, _NO_VIOLATIONS) == Regime.CRITICAL
