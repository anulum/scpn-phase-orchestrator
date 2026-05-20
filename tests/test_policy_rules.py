# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyRule,
    PolicySTLResult,
    PolicySTLSpec,
    evaluate_policy_stl_specs,
    load_policy_rules,
    load_policy_stl_specs,
    synthesise_policy_stl_automata,
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


def _rich_state() -> UPDEState:
    return UPDEState(
        layers=[
            LayerState(R=0.21, psi=0.0, mean_amplitude=0.44, amplitude_spread=0.12),
            LayerState(R=0.82, psi=0.0, mean_amplitude=0.91, amplitude_spread=0.34),
        ],
        cross_layer_alignment=np.zeros((2, 2)),
        stability_proxy=0.62,
        regime_id="test",
        mean_amplitude=0.71,
        pac_max=0.23,
        subcritical_fraction=0.15,
        boundary_violation_count=3,
        imprint_mean=0.56,
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


@pytest.mark.parametrize("rules", [None, [], [object()]])
def test_policy_engine_rejects_invalid_rules(rules):
    with pytest.raises(ValueError, match="rules"):
        PolicyEngine(rules)


@pytest.mark.parametrize("dt", [True, -0.1, float("nan"), float("inf"), "1.0"])
def test_policy_engine_rejects_invalid_clock_increment(dt):
    engine = PolicyEngine([_rule()])
    with pytest.raises(ValueError, match="dt"):
        engine.advance_clock(dt)


def test_policy_engine_evaluates_compound_or_and_branching() -> None:
    engine = PolicyEngine(
        [
            PolicyRule(
                name="or_rule",
                regimes=["NOMINAL"],
                condition=CompoundCondition(
                    [
                        PolicyCondition("R", layer=0, op=">", threshold=0.8),
                        PolicyCondition(
                            "stability_proxy", layer=None, op="<", threshold=0.9
                        ),
                    ],
                    logic="OR",
                ),
                actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
            ),
            PolicyRule(
                name="and_rule",
                regimes=["NOMINAL"],
                condition=CompoundCondition(
                    [
                        PolicyCondition("R", layer=0, op=">", threshold=0.8),
                        PolicyCondition(
                            "stability_proxy", layer=None, op="<", threshold=0.9
                        ),
                    ],
                    logic="AND",
                ),
                actions=[
                    PolicyAction(knob="alpha", scope="global", value=-0.2, ttl_s=2.0)
                ],
            ),
        ]
    )

    actions = engine.evaluate(
        Regime.NOMINAL,
        _state([0.95]),
        [0],
        [],
    )

    assert [action.knob for action in actions] == ["K"]
    assert [action.justification for action in actions] == ["policy rule: or_rule"]


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


def test_policy_actions_preserve_metadata_and_rule_order():
    first = _rule(
        name="raise_k",
        threshold=0.7,
        knob="K",
        scope="global",
        value=0.2,
    )
    second = _rule(
        name="trim_alpha",
        threshold=0.7,
        knob="alpha",
        scope="layer_1",
        value=-0.05,
    )
    engine = PolicyEngine([first, second])

    actions = engine.evaluate(Regime.NOMINAL, _state([0.8]), [0], [])

    assert [action.knob for action in actions] == ["K", "alpha"]
    assert [action.scope for action in actions] == ["global", "layer_1"]
    assert [action.value for action in actions] == [0.2, -0.05]
    assert [action.ttl_s for action in actions] == [5.0, 5.0]
    assert [action.justification for action in actions] == [
        "policy rule: raise_k",
        "policy rule: trim_alpha",
    ]


def test_missing_metric_and_unsupported_operator_do_not_fire():
    rules = [
        PolicyRule(
            name="unknown_metric",
            regimes=["NOMINAL"],
            condition=PolicyCondition(
                metric="not_a_metric", layer=None, op=">", threshold=0.0
            ),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
        ),
        PolicyRule(
            name="unknown_operator",
            regimes=["NOMINAL"],
            condition=PolicyCondition(metric="R", layer=0, op="!=", threshold=0.0),
            actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
        ),
    ]

    assert PolicyEngine(rules).evaluate(Regime.NOMINAL, _state([0.8]), [0], []) == []


@pytest.mark.parametrize(
    ("metric", "layer", "threshold"),
    [
        ("R_good", 1, 0.0),
        ("R_bad", 1, 0.0),
        ("amplitude_spread", 5, 0.0),
        ("mean_amplitude_layer", 5, 0.0),
    ],
)
def test_missing_layer_mapped_metrics_do_not_fire(
    metric: str, layer: int, threshold: float
):
    rule = _rule(metric=metric, layer=layer, op=">", threshold=threshold)

    actions = PolicyEngine([rule]).evaluate(Regime.NOMINAL, _rich_state(), [0], [1])

    assert actions == []


@pytest.mark.parametrize(
    ("metric", "layer", "op", "threshold"),
    [
        ("pac_max", None, "==", 0.23),
        ("mean_amplitude", None, ">=", 0.70),
        ("subcritical_fraction", None, "<=", 0.15),
        ("amplitude_spread", 1, "==", 0.34),
        ("mean_amplitude_layer", 0, "==", 0.44),
        ("boundary_violation_count", None, "==", 3.0),
        ("imprint_mean", None, "==", 0.56),
    ],
)
def test_policy_engine_evaluates_diagnostic_metrics(
    metric: str, layer: int | None, op: str, threshold: float
):
    rule = PolicyRule(
        name=f"diagnostic_{metric}",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric=metric,
            layer=layer,
            op=op,
            threshold=threshold,
        ),
        actions=[PolicyAction(knob="diagnostic", scope="global", value=1.0, ttl_s=2.0)],
    )

    actions = PolicyEngine([rule]).evaluate(Regime.NOMINAL, _rich_state(), [0], [1])

    assert [action.justification for action in actions] == [
        f"policy rule: diagnostic_{metric}"
    ]


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


def test_load_policy_rules_missing_rules_section_returns_empty(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("stl_monitors: []\n", encoding="utf-8")
    assert load_policy_rules(p) == []


def test_load_policy_stl_specs_yaml(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        "rules: []\n"
        "stl_monitors:\n"
        "  - name: keep_sync\n"
        "    spec: always (R >= 0.3)\n"
        "    severity: hard\n"
        "  - name: eventual_recovery\n"
        "    spec: eventually (R >= 0.8)\n",
        encoding="utf-8",
    )

    specs = load_policy_stl_specs(p)

    assert specs == [
        PolicySTLSpec(
            name="keep_sync",
            spec="always (R >= 0.3)",
            severity="hard",
        ),
        PolicySTLSpec(
            name="eventual_recovery",
            spec="eventually (R >= 0.8)",
            severity="soft",
        ),
    ]


def test_load_policy_stl_specs_missing_section_returns_empty(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")

    assert load_policy_stl_specs(p) == []


def test_load_policy_stl_specs_rejects_bad_severity(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        "stl_monitors:\n"
        "  - name: bad\n"
        "    spec: always (R >= 0.3)\n"
        "    severity: emergency\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="severity"):
        load_policy_stl_specs(p)


def test_evaluate_policy_stl_specs_returns_audit_records():
    specs = [
        PolicySTLSpec(
            name="keep_sync",
            spec="always (R >= 0.3)",
            severity="hard",
        ),
        PolicySTLSpec(
            name="recover",
            spec="eventually (R >= 0.8)",
        ),
    ]

    results = evaluate_policy_stl_specs(specs, {"R": [0.2, 0.4, 0.9]})
    audit_records = [result.to_audit_record() for result in results]

    assert all(isinstance(result, PolicySTLResult) for result in results)
    assert audit_records[0]["name"] == "keep_sync"
    assert audit_records[0]["severity"] == "hard"
    assert audit_records[0]["satisfied"] is False
    assert audit_records[1]["satisfied"] is True


def test_synthesise_policy_stl_automata_returns_named_audit_records():
    specs = [
        PolicySTLSpec(
            name="recover",
            spec="eventually (R >= 0.8)",
            severity="hard",
        )
    ]

    automata = synthesise_policy_stl_automata(specs, {"R": [0.2, 0.9]})
    audit_record = automata[0].to_audit_record()

    assert audit_record["name"] == "recover"
    assert audit_record["severity"] == "hard"
    assert audit_record["spec"] == "eventually (R >= 0.8)"
    assert "states" in audit_record


def test_load_policy_rules_recursion_error_is_parse_error(tmp_path, monkeypatch):
    import yaml

    p = tmp_path / "policy.yaml"
    p.write_text("rules: []\n", encoding="utf-8")

    def raise_recursion(_: str) -> object:
        raise RecursionError("nested YAML")

    monkeypatch.setattr(yaml, "safe_load", raise_recursion)
    with pytest.raises(ValueError, match="policy rules YAML parse error"):
        load_policy_rules(p)


def test_load_policy_rules_fuzzer_unicode_overflow_is_parse_error(tmp_path):
    """Malformed YAML Unicode escapes from fuzzing must not escape the loader."""
    p = tmp_path / "policy.yaml"
    p.write_text('"\\\\\\U' + ("e" * 57) + "\\\\~", encoding="utf-8")

    with pytest.raises(ValueError, match="policy rules YAML parse error"):
        load_policy_rules(p)


def test_load_policy_stl_specs_fuzzer_unicode_overflow_is_parse_error(tmp_path):
    """The STL policy loader shares the same YAML parser hardening contract."""
    p = tmp_path / "policy.yaml"
    p.write_text('"\\\\\\U' + ("e" * 57) + "\\\\~", encoding="utf-8")

    with pytest.raises(ValueError, match="policy rules YAML parse error"):
        load_policy_stl_specs(p)


def test_policy_loaders_wrap_unreadable_paths(tmp_path):
    with pytest.raises(ValueError, match="cannot read policy rules"):
        load_policy_rules(tmp_path)
    with pytest.raises(ValueError, match="cannot read policy rules"):
        load_policy_stl_specs(tmp_path)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("rules: not-a-list\n", "rules must be a list"),
        ("rules:\n  - 5\n", "rule must be a mapping"),
        (
            "rules:\n"
            "  - regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "rule missing required key 'name'",
        ),
        (
            "rules:\n"
            "  - name: ''\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "rule.name must be a non-empty string",
        ),
        (
            "rules:\n"
            "  - name: bad_threshold\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: not-finite\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "condition.threshold must be a finite number",
        ),
        (
            "rules:\n"
            "  - name: infinite_value\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: .inf\n"
            "      ttl_s: 1.0\n",
            "action.value must be a finite number",
        ),
        (
            "rules:\n"
            "  - name: negative_ttl\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: -1.0\n",
            "action.ttl_s must be non-negative",
        ),
        (
            "rules:\n"
            "  - name: bool_layer\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: true\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "condition.layer must be a non-negative integer",
        ),
        (
            "rules:\n"
            "  - name: text_layer\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: first\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "condition.layer must be a non-negative integer",
        ),
        (
            "rules:\n"
            "  - name: negative_fires\n"
            "    regime: [NOMINAL]\n"
            "    max_fires: -1\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "rule.max_fires must be non-negative",
        ),
        (
            "rules:\n"
            "  - name: bad_op\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '!='\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "condition.op is unsupported",
        ),
        (
            "rules:\n"
            "  - name: empty_conditions\n"
            "    regime: [NOMINAL]\n"
            "    conditions: []\n"
            "    actions:\n"
            "      - knob: K\n"
            "        scope: global\n"
            "        value: 0.1\n"
            "        ttl_s: 1.0\n",
            "rule.conditions must not be empty",
        ),
        (
            "rules:\n"
            "  - name: bad_logic_type\n"
            "    regime: [NOMINAL]\n"
            "    logic: 5\n"
            "    conditions:\n"
            "      - metric: R\n"
            "        layer: 0\n"
            "        op: '>'\n"
            "        threshold: 0.1\n"
            "    actions:\n"
            "      - knob: K\n"
            "        scope: global\n"
            "        value: 0.1\n"
            "        ttl_s: 1.0\n",
            "rule.logic must be a string",
        ),
        (
            "rules:\n"
            "  - name: bad_logic\n"
            "    regime: [NOMINAL]\n"
            "    logic: XOR\n"
            "    conditions:\n"
            "      - metric: R\n"
            "        layer: 0\n"
            "        op: '>'\n"
            "        threshold: 0.1\n"
            "    actions:\n"
            "      - knob: K\n"
            "        scope: global\n"
            "        value: 0.1\n"
            "        ttl_s: 1.0\n",
            "rule.logic must be AND or OR",
        ),
        (
            "rules:\n"
            "  - name: empty_actions\n"
            "    regime: [NOMINAL]\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    actions: []\n",
            "rule.actions must not be empty",
        ),
        (
            "rules:\n"
            "  - name: bad_regime\n"
            "    regime: [NOMINAL, '']\n"
            "    condition:\n"
            "      metric: R\n"
            "      layer: 0\n"
            "      op: '>'\n"
            "      threshold: 0.1\n"
            "    action:\n"
            "      knob: K\n"
            "      scope: global\n"
            "      value: 0.1\n"
            "      ttl_s: 1.0\n",
            "rule.regime must be a list of non-empty strings",
        ),
    ],
)
def test_load_policy_rules_rejects_bad_schema(tmp_path, body: str, match: str):
    p = tmp_path / "policy.yaml"
    p.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_policy_rules(p)


def test_load_policy_rules_rejects_rule_and_action_caps(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("rules:\n" + ("  - {}\n" * 1001), encoding="utf-8")
    with pytest.raises(ValueError, match="too many rules"):
        load_policy_rules(p)

    too_many_conditions = "\n".join(
        "      - {metric: R, layer: 0, op: '>', threshold: 0.1}" for _ in range(33)
    )
    p.write_text(
        "rules:\n"
        "  - name: too_many_conditions\n"
        "    regime: [NOMINAL]\n"
        "    conditions:\n"
        f"{too_many_conditions}\n"
        "    actions:\n"
        "      - {knob: K, scope: global, value: 0.1, ttl_s: 1.0}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="too many rule conditions"):
        load_policy_rules(p)

    too_many_actions = "\n".join(
        "      - {knob: K, scope: global, value: 0.1, ttl_s: 1.0}" for _ in range(33)
    )
    p.write_text(
        "rules:\n"
        "  - name: too_many_actions\n"
        "    regime: [NOMINAL]\n"
        "    condition: {metric: R, layer: 0, op: '>', threshold: 0.1}\n"
        "    actions:\n"
        f"{too_many_actions}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="too many rule actions"):
        load_policy_rules(p)


def test_load_policy_stl_specs_rejects_bad_schema_and_cap(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("stl_monitors: not-a-list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stl_monitors must be a list"):
        load_policy_stl_specs(p)

    p.write_text("stl_monitors:\n  - 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stl_monitor must be a mapping"):
        load_policy_stl_specs(p)

    p.write_text("stl_monitors:\n" + ("  - {}\n" * 1001), encoding="utf-8")
    with pytest.raises(ValueError, match="too many stl monitors"):
        load_policy_stl_specs(p)


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


def test_cooldown_refires_at_exact_boundary_and_preserves_rule_caps():
    rule = PolicyRule(
        name="boundary_cd",
        regimes=["NOMINAL"],
        condition=PolicyCondition(metric="R", layer=0, op=">", threshold=0.0),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        cooldown_s=10.0,
        max_fires=2,
    )
    engine = PolicyEngine([rule])
    state = _state([0.5])

    assert len(engine.evaluate(Regime.NOMINAL, state, [0], [])) == 1
    engine.advance_clock(9.999)
    assert engine.evaluate(Regime.NOMINAL, state, [0], []) == []
    engine.advance_clock(0.001)
    assert len(engine.evaluate(Regime.NOMINAL, state, [0], [])) == 1
    engine.advance_clock(10.0)
    assert engine.evaluate(Regime.NOMINAL, state, [0], []) == []


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


def test_rejects_unknown_metric_and_unsupported_operator():
    state = UPDEState(
        layers=[LayerState(R=0.2, psi=0.0)],
        cross_layer_alignment=np.array([[1.0]]),
        stability_proxy=0.1,
        regime_id="NOMINAL",
    )

    unknown_metric = PolicyRule(
        name="unknown_metric",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric="does_not_exist", layer=None, op=">", threshold=0.0
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )
    unsupported_op = PolicyRule(
        name="unsupported_op",
        regimes=["NOMINAL"],
        condition=PolicyCondition(metric="R", layer=None, op="!=", threshold=0.0),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )

    assert PolicyEngine([unknown_metric]).evaluate(Regime.NOMINAL, state, [], []) == []
    assert PolicyEngine([unsupported_op]).evaluate(Regime.NOMINAL, state, [], []) == []


@pytest.mark.parametrize(
    "metric,layer", [("amplitude_spread", 3), ("mean_amplitude_layer", 3)]
)
def test_metric_layer_lookup_is_none_out_of_range(metric, layer):
    state = UPDEState(
        layers=[LayerState(R=0.4, psi=0.0)],
        cross_layer_alignment=np.array([[1.0]]),
        stability_proxy=0.1,
        regime_id="NOMINAL",
    )

    rule = PolicyRule(
        name="range_lookup",
        regimes=["NOMINAL"],
        condition=PolicyCondition(metric=metric, layer=layer, op=">", threshold=0.0),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )
    assert PolicyEngine([rule]).evaluate(Regime.NOMINAL, state, [], []) == []


def test_rule_metrics_support_boundary_violation_count_and_imprint_mean():
    state = UPDEState(
        layers=[LayerState(R=0.4, psi=0.0)],
        cross_layer_alignment=np.array([[1.0]]),
        stability_proxy=0.1,
        regime_id="NOMINAL",
        boundary_violation_count=4,
        imprint_mean=0.12,
    )

    boundary = PolicyRule(
        name="boundary_violation",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric="boundary_violation_count", layer=None, op=">=", threshold=3
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )
    imprint = PolicyRule(
        name="imprint_metric",
        regimes=["NOMINAL"],
        condition=PolicyCondition(
            metric="imprint_mean", layer=None, op="<", threshold=0.5
        ),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
    )

    assert PolicyEngine([boundary]).evaluate(Regime.NOMINAL, state, [], []) != []
    assert PolicyEngine([imprint]).evaluate(Regime.NOMINAL, state, [], []) != []


def test_load_policy_rules_rejects_rule_mapping_type_errors(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text("rules:\n  - 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rule must be a mapping"):
        load_policy_rules(p)

    p.write_text(
        "rules:\n"
        "  - name: missing_threshold\n"
        "    regime: [NOMINAL]\n"
        "    condition:\n"
        "      metric: R\n"
        "      layer: 0\n"
        "      op: '>'\n"
        "    action:\n"
        "      knob: K\n"
        "      scope: global\n"
        "      value: 0.1\n"
        "      ttl_s: 1.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="condition missing required key 'threshold'"):
        load_policy_rules(p)


def test_load_policy_stl_specs_defaults_and_audit_records(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        "stl_monitors:\n  - name: sync_guard\n    spec: always (R >= 0.3)\n",
        encoding="utf-8",
    )
    specs = load_policy_stl_specs(p)
    assert len(specs) == 1
    assert specs[0].name == "sync_guard"
    assert specs[0].severity == "soft"

    results = evaluate_policy_stl_specs(specs, {"R": [0.5]})
    audit = results[0].to_audit_record()
    assert audit["name"] == "sync_guard"
    assert audit["severity"] == "soft"
    assert "robustness" in audit
    assert "satisfied" in audit

    automata = synthesise_policy_stl_automata((specs[0],), {"R": [0.2, 0.5]})
    automaton_record = automata[0].to_audit_record()
    assert automaton_record["name"] == "sync_guard"
    assert automaton_record["severity"] == "soft"
    assert "satisfied" in automaton_record
    assert "states" in automaton_record


def test_load_policy_stl_specs_rejects_empty_name(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        "stl_monitors:\n  - name: ''\n    spec: always (R >= 0.3)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stl_monitor.name must be a non-empty string"):
        load_policy_stl_specs(p)


def test_cooldown_then_max_fires_prevents_post_boundary_refire():
    rule = PolicyRule(
        name="combo",
        regimes=["NOMINAL"],
        condition=PolicyCondition(metric="R", layer=0, op=">", threshold=0.0),
        actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=5.0)],
        cooldown_s=5.0,
        max_fires=1,
    )
    engine = PolicyEngine([rule])
    state = UPDEState(
        layers=[LayerState(R=0.5, psi=0.0)],
        cross_layer_alignment=np.array([[1.0]]),
        stability_proxy=0.1,
        regime_id="NOMINAL",
    )

    assert len(engine.evaluate(Regime.NOMINAL, state, [0], [])) == 1
    engine.advance_clock(10.0)
    assert engine.evaluate(Regime.NOMINAL, state, [0], []) == []


class TestPolicyRulesPipelineWiring:
    """Pipeline: engine R → UPDEState → PolicyEngine → actions."""

    def test_engine_r_triggers_policy_rule(self):
        """UPDEEngine → R → UPDEState → PolicyEngine.evaluate → actions.
        Proves policy rules consume engine state and produce control."""
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.normal(1.0, 0.5, n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)

        rule = PolicyRule(
            name="engine_test",
            regimes=["NOMINAL", "DEGRADED", "CRITICAL"],
            condition=PolicyCondition(
                metric="stability_proxy",
                layer=None,
                op="<",
                threshold=0.99,
            ),
            actions=[
                PolicyAction(
                    knob="K",
                    scope="global",
                    value=0.1,
                    ttl_s=5.0,
                ),
            ],
        )
        engine = PolicyEngine([rule])
        state = _state([r], stability=r)
        actions = engine.evaluate(Regime.NOMINAL, state, [0], [])
        assert len(actions) >= 1, "Rule should fire for R < 0.99"
