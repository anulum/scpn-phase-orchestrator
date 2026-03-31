# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy amplitude tests

"""Tests for v0.4.1 amplitude metrics in the policy engine."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _amp_state(
    pac_max: float = 0.0,
    mean_amplitude: float = 0.0,
    subcritical_fraction: float = 0.0,
    layer_amps: list[tuple[float, float]] | None = None,
) -> UPDEState:
    layers = []
    if layer_amps:
        for ma, spread in layer_amps:
            layers.append(
                LayerState(R=0.5, psi=0.0, mean_amplitude=ma, amplitude_spread=spread)
            )
    else:
        layers.append(LayerState(R=0.5, psi=0.0))
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.zeros((len(layers), len(layers))),
        stability_proxy=0.5,
        regime_id="test",
        mean_amplitude=mean_amplitude,
        pac_max=pac_max,
        subcritical_fraction=subcritical_fraction,
    )


def _rule(metric: str, layer: int | None, op: str, threshold: float) -> PolicyRule:
    return PolicyRule(
        name=f"test_{metric}",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric=metric, layer=layer, op=op, threshold=threshold
        ),
        actions=[PolicyAction(knob="K", scope="global", value=1.0, ttl_s=5.0)],
    )


def test_pac_max_fires():
    engine = PolicyEngine([_rule("pac_max", None, ">", 0.4)])
    actions = engine.evaluate(Regime.NOMINAL, _amp_state(pac_max=0.6), [], [])
    assert len(actions) == 1


def test_pac_max_no_fire():
    engine = PolicyEngine([_rule("pac_max", None, ">", 0.4)])
    actions = engine.evaluate(Regime.NOMINAL, _amp_state(pac_max=0.2), [], [])
    assert actions == []


def test_mean_amplitude_fires():
    engine = PolicyEngine([_rule("mean_amplitude", None, ">", 0.5)])
    actions = engine.evaluate(Regime.NOMINAL, _amp_state(mean_amplitude=0.8), [], [])
    assert len(actions) == 1


def test_subcritical_fraction_fires():
    engine = PolicyEngine([_rule("subcritical_fraction", None, ">", 0.3)])
    actions = engine.evaluate(
        Regime.NOMINAL, _amp_state(subcritical_fraction=0.5), [], []
    )
    assert len(actions) == 1


def test_amplitude_spread_per_layer():
    state = _amp_state(layer_amps=[(1.0, 0.8), (0.5, 0.2)])
    engine = PolicyEngine([_rule("amplitude_spread", 0, ">", 0.5)])
    actions = engine.evaluate(Regime.NOMINAL, state, [], [])
    assert len(actions) == 1


def test_amplitude_spread_layer_out_of_range():
    state = _amp_state(layer_amps=[(1.0, 0.8)])
    engine = PolicyEngine([_rule("amplitude_spread", 5, ">", 0.0)])
    actions = engine.evaluate(Regime.NOMINAL, state, [], [])
    assert actions == []


def test_mean_amplitude_layer():
    state = _amp_state(layer_amps=[(1.5, 0.1), (0.3, 0.1)])
    engine = PolicyEngine([_rule("mean_amplitude_layer", 0, ">", 1.0)])
    actions = engine.evaluate(Regime.NOMINAL, state, [], [])
    assert len(actions) == 1


def test_compound_pac_and_subcritical():
    rule = PolicyRule(
        name="compound_amp",
        regimes=["NOMINAL"],
        condition=CompoundCondition(
            conditions=[
                PolicyCondition(metric="pac_max", layer=None, op=">", threshold=0.4),
                PolicyCondition(
                    metric="subcritical_fraction", layer=None, op=">", threshold=0.2
                ),
            ],
            logic="AND",
        ),
        actions=[PolicyAction(knob="K", scope="global", value=1.5, ttl_s=5.0)],
    )
    engine = PolicyEngine([rule])
    state = _amp_state(pac_max=0.6, subcritical_fraction=0.4)
    assert len(engine.evaluate(Regime.NOMINAL, state, [], [])) == 1
    state_low = _amp_state(pac_max=0.6, subcritical_fraction=0.1)
    assert engine.evaluate(Regime.NOMINAL, state_low, [], []) == []


# Pipeline wiring: tests above exercise PolicyEngine.evaluate with amplitude-
# specific metrics (pac_max, mean_amplitude, subcritical_fraction, amplitude_
# spread) from UPDEState — the SL amplitude policy pipeline.
