# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
    load_policy_rules,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(rs: list[float], stability: float | None = None) -> UPDEState:
    stab = stability if stability is not None else sum(rs) / max(len(rs), 1)
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0) for r in rs],
        cross_layer_alignment=np.zeros((len(rs), len(rs))),
        stability_proxy=stab,
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
            metric=metric, layer=layer, op=op, threshold=threshold
        ),
        actions=[PolicyAction(knob=knob, scope=scope, value=value, ttl_s=5.0)],
    )


# ── v0.1 backward compatibility ──────────────────────────────────────


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
    assert isinstance(rules[0].condition, PolicyCondition)
    assert rules[0].condition.op == ">"
    assert len(rules[0].actions) == 1
    assert rules[0].actions[0].knob == "K"


def test_load_empty_policy_yaml(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")
    assert load_policy_rules(p) == []


def test_out_of_range_layer_returns_none():
    rule = _rule(metric="R", layer=99, op=">", threshold=0.0)
    engine = PolicyEngine([rule])
    actions = engine.evaluate(Regime.NOMINAL, _state([0.5]), [0], [])
    assert actions == []


# ── v0.2 compound conditions ─────────────────────────────────────────


def test_compound_and_both_true():
    rule = PolicyRule(
        name="both",
        regimes=["NOMINAL"],
        condition=CompoundCondition(
            conditions=[
                PolicyCondition(metric="R", layer=0, op=">", threshold=0.5),
                PolicyCondition(metric="R", layer=1, op=">", threshold=0.5),
            ],
            logic="AND",
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.2, ttl_s=5.0)],
    )
    engine = PolicyEngine([rule])
    assert len(engine.evaluate(Regime.NOMINAL, _state([0.8, 0.9]), [0, 1], [])) == 1


def test_compound_and_one_false():
    rule = PolicyRule(
        name="partial",
        regimes=["NOMINAL"],
        condition=CompoundCondition(
            conditions=[
                PolicyCondition(metric="R", layer=0, op=">", threshold=0.5),
                PolicyCondition(metric="R", layer=1, op=">", threshold=0.5),
            ],
            logic="AND",
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.2, ttl_s=5.0)],
    )
    engine = PolicyEngine([rule])
    assert engine.evaluate(Regime.NOMINAL, _state([0.8, 0.2]), [0, 1], []) == []


def test_compound_or_one_true():
    rule = PolicyRule(
        name="either",
        regimes=["NOMINAL"],
        condition=CompoundCondition(
            conditions=[
                PolicyCondition(metric="R", layer=0, op=">", threshold=0.9),
                PolicyCondition(metric="R", layer=1, op=">", threshold=0.5),
            ],
            logic="OR",
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=3.0)],
    )
    engine = PolicyEngine([rule])
    assert len(engine.evaluate(Regime.NOMINAL, _state([0.3, 0.8]), [0, 1], [])) == 1


def test_compound_or_both_false():
    rule = PolicyRule(
        name="neither",
        regimes=["NOMINAL"],
        condition=CompoundCondition(
            conditions=[
                PolicyCondition(metric="R", layer=0, op=">", threshold=0.9),
                PolicyCondition(metric="R", layer=1, op=">", threshold=0.9),
            ],
            logic="OR",
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=3.0)],
    )
    engine = PolicyEngine([rule])
    assert engine.evaluate(Regime.NOMINAL, _state([0.3, 0.3]), [0, 1], []) == []


def test_stability_proxy_metric():
    rule = PolicyRule(
        name="stab",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric="stability_proxy", layer=None, op="<", threshold=0.5
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.3, ttl_s=5.0)],
    )
    engine = PolicyEngine([rule])
    low = engine.evaluate(Regime.NOMINAL, _state([0.2], stability=0.3), [0], [])
    assert len(low) == 1
    high = engine.evaluate(Regime.NOMINAL, _state([0.8], stability=0.9), [0], [])
    assert high == []


# ── v0.2 action chains ───────────────────────────────────────────────


def test_action_chain_multiple_actions():
    rule = PolicyRule(
        name="chain",
        regimes=["DEGRADED"],
        condition=PolicyCondition(metric="R", layer=0, op="<", threshold=0.4),
        actions=[
            PolicyAction(knob="K", scope="global", value=0.5, ttl_s=5.0),
            PolicyAction(knob="alpha", scope="layer_0", value=0.1, ttl_s=3.0),
            PolicyAction(knob="zeta", scope="global", value=0.2, ttl_s=10.0),
        ],
    )
    engine = PolicyEngine([rule])
    actions = engine.evaluate(Regime.DEGRADED, _state([0.2]), [0], [])
    assert len(actions) == 3
    assert [a.knob for a in actions] == ["K", "alpha", "zeta"]
    assert actions[1].scope == "layer_0"


# ── v0.2 cooldown and max_fires ──────────────────────────────────────


def test_cooldown_blocks_rapid_refire():
    rule = PolicyRule(
        name="cd",
        regimes=["NOMINAL"],
        condition=PolicyCondition(metric="R", layer=0, op=">", threshold=0.0),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        cooldown_s=10.0,
    )
    engine = PolicyEngine([rule])
    s = _state([0.5])
    assert len(engine.evaluate(Regime.NOMINAL, s, [0], [])) == 1
    assert engine.evaluate(Regime.NOMINAL, s, [0], []) == []
    engine.advance_clock(11.0)
    assert len(engine.evaluate(Regime.NOMINAL, s, [0], [])) == 1


def test_max_fires_cap():
    rule = PolicyRule(
        name="cap",
        regimes=["NOMINAL"],
        condition=PolicyCondition(metric="R", layer=0, op=">", threshold=0.0),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        max_fires=2,
    )
    engine = PolicyEngine([rule])
    s = _state([0.5])
    assert len(engine.evaluate(Regime.NOMINAL, s, [0], [])) == 1
    assert len(engine.evaluate(Regime.NOMINAL, s, [0], [])) == 1
    assert engine.evaluate(Regime.NOMINAL, s, [0], []) == []


# ── v0.2 YAML compound loading ───────────────────────────────────────


def test_load_compound_yaml(tmp_path):
    p = tmp_path / "compound.yaml"
    p.write_text(
        "rules:\n"
        "  - name: compound_rule\n"
        "    regime: [DEGRADED]\n"
        "    logic: AND\n"
        "    conditions:\n"
        "      - metric: R\n"
        "        layer: 0\n"
        "        op: '<'\n"
        "        threshold: 0.3\n"
        "      - metric: stability_proxy\n"
        "        op: '<'\n"
        "        threshold: 0.5\n"
        "    actions:\n"
        "      - knob: K\n"
        "        scope: global\n"
        "        value: 0.5\n"
        "        ttl_s: 5.0\n"
        "      - knob: alpha\n"
        "        scope: layer_0\n"
        "        value: 0.1\n"
        "        ttl_s: 3.0\n"
        "    cooldown_s: 10.0\n"
        "    max_fires: 5\n",
        encoding="utf-8",
    )
    rules = load_policy_rules(p)
    assert len(rules) == 1
    r = rules[0]
    assert isinstance(r.condition, CompoundCondition)
    assert len(r.condition.conditions) == 2
    assert r.condition.logic == "AND"
    assert len(r.actions) == 2
    assert r.actions[0].knob == "K"
    assert r.actions[1].knob == "alpha"
    assert r.cooldown_s == 10.0
    assert r.max_fires == 5


def test_load_mixed_v1_v2_yaml(tmp_path):
    p = tmp_path / "mixed.yaml"
    p.write_text(
        "rules:\n"
        "  - name: v1_rule\n"
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
        "      ttl_s: 5.0\n"
        "  - name: v2_rule\n"
        "    regime: [DEGRADED]\n"
        "    logic: OR\n"
        "    conditions:\n"
        "      - metric: R\n"
        "        layer: 0\n"
        "        op: '<'\n"
        "        threshold: 0.3\n"
        "      - metric: R\n"
        "        layer: 1\n"
        "        op: '<'\n"
        "        threshold: 0.3\n"
        "    actions:\n"
        "      - knob: K\n"
        "        scope: global\n"
        "        value: 0.5\n"
        "        ttl_s: 8.0\n",
        encoding="utf-8",
    )
    rules = load_policy_rules(p)
    assert len(rules) == 2
    assert isinstance(rules[0].condition, PolicyCondition)
    assert isinstance(rules[1].condition, CompoundCondition)
    assert rules[1].condition.logic == "OR"
