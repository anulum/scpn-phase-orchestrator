# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
    load_policy_rules,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(rs: list[float]) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0) for r in rs],
        cross_layer_alignment=np.zeros((len(rs), len(rs))),
        stability_proxy=sum(rs) / max(len(rs), 1),
        regime_id="test",
    )


def _rule(
    name: str = "r1",
    regimes: list[str] | None = None,
    metric: str = "R",
    layer: int = 0,
    op: str = ">",
    threshold: float = 0.5,
    knob: str = "K",
    scope: str = "global",
    value: float = 0.1,
) -> PolicyRule:
    return PolicyRule(
        name=name,
        regimes=regimes or ["NOMINAL"],
        condition=PolicyCondition(
            metric=metric,
            layer=layer,
            op=op,
            threshold=threshold,
        ),
        knob=knob,
        scope=scope,
        value=value,
        ttl_s=5.0,
    )


def test_fires_when_condition_met():
    engine = PolicyEngine([_rule(op=">", threshold=0.5)])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.8]), [0], [])
    assert len(actions) == 1
    assert actions[0].knob == "K"


def test_no_fire_when_condition_unmet():
    engine = PolicyEngine([_rule(op=">", threshold=0.9)])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.5]), [0], [])
    assert actions == []


def test_regime_filter():
    engine = PolicyEngine([_rule(regimes=["CRITICAL"])])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.8]), [0], [])
    assert actions == []
    actions = engine.evaluate(Regime.CRITICAL, _state([0.8]), [0], [])
    assert len(actions) == 1


def test_r_bad_metric():
    rule = _rule(metric="R_bad", layer=0, op=">", threshold=0.7)
    engine = PolicyEngine([rule])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.9, 0.5]), [], [0])
    assert len(actions) == 1


def test_r_good_metric():
    rule = _rule(metric="R_good", layer=0, op="<", threshold=0.4)
    engine = PolicyEngine([rule])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.3, 0.9]), [0], [])
    assert len(actions) == 1


def test_load_policy_rules_yaml(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        "rules:\n"
        "  - name: test_rule\n"
        "    regime: [NOMINAL]\n"
        "    condition:\n"
        "      metric: R\n"
        "      layer: 0\n"
        "      op: '>'\n"
        "      threshold: 0.5\n"
        "    action:\n"
        "      knob: K\n"
        "      scope: global\n"
        "      value: 0.1\n"
        "      ttl_s: 5.0\n",
        encoding="utf-8",
    )
    rules = load_policy_rules(p)
    assert len(rules) == 1
    assert rules[0].name == "test_rule"
    assert rules[0].condition.op == ">"


def test_load_empty_policy_yaml(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")
    assert load_policy_rules(p) == []


def test_out_of_range_layer_returns_none():
    rule = _rule(metric="R", layer=99, op=">", threshold=0.0)
    engine = PolicyEngine([rule])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.5]), [0], [])
    assert actions == []
