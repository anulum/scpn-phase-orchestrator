# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal exporter residual boundary tests

from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
    PolicySTLSpec,
    export_policy_rules_prism,
    export_policy_rules_tla,
    export_stl_specs_prism,
)


def _action(knob: str = "K") -> PolicyAction:
    return PolicyAction(knob=knob, scope="global", value=0.1, ttl_s=5.0)


def _rule(
    name: str,
    *,
    condition: PolicyCondition | CompoundCondition | None = None,
    regimes: list[str] | None = None,
    actions: list[PolicyAction] | None = None,
) -> PolicyRule:
    return PolicyRule(
        name=name,
        regimes=regimes or ["nominal", "CRITICAL"],
        condition=condition
        or PolicyCondition(metric="R_bad", layer=0, op="<=", threshold=0.25),
        actions=actions or [_action()],
        max_fires=2,
    )


def test_policy_rule_exports_are_deterministic_and_sorted_at_public_boundary() -> None:
    rules = [
        _rule(
            "rule-with-spaces",
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition("z.metric", None, ">=", 0.4),
                    PolicyCondition("a.metric", 2, "==", 0.7),
                ],
                logic="OR",
            ),
            actions=[_action("K-local"), _action("K local")],
        ),
        _rule(
            "rule with spaces",
            condition=PolicyCondition("R_good", 0, ">", 0.6),
        ),
    ]

    prism_a = export_policy_rules_prism(rules, module_name="policy exporter")
    prism_b = export_policy_rules_prism(rules, module_name="policy exporter")
    tla_a = export_policy_rules_tla(rules, module_name="9 policy exporter")
    tla_b = export_policy_rules_tla(rules, module_name="9 policy exporter")

    assert prism_a == prism_b
    assert tla_a == tla_b
    assert prism_a.metric_names == {
        "R_good.0": "R_good_0",
        "a.metric.2": "a_metric_2",
        "z.metric": "z_metric",
    }
    assert prism_a.rule_names == {
        "rule-with-spaces": "rule_with_spaces",
        "rule with spaces": "rule_with_spaces_2",
    }
    assert "//   CRITICAL -> 0" in prism_a.model
    assert "//   NOMINAL -> 1" in prism_a.model
    assert "(z_metric >= 0.40000000000000002 | a_metric_2 == 0.69999999999999996)" in (
        prism_a.model
    )
    assert "---- MODULE SpoModule_9_policy_exporter ----" in tla_a.module
    assert "(z_metric >= 0.40000000000000002 \\/ a_metric_2 = 0.69999999999999996)" in (
        tla_a.module
    )


@pytest.mark.parametrize(
    "exporter",
    [export_policy_rules_prism, export_policy_rules_tla],
)
def test_policy_rule_exports_reject_malformed_compound_logic_fail_closed(
    exporter,
) -> None:
    bad_rule = _rule(
        "bad-logic",
        condition=CompoundCondition(
            [PolicyCondition("R", 0, ">", 0.5)],
            logic="XOR",
        ),
    )

    with pytest.raises(PolicyError, match="logic must be AND or OR"):
        exporter([bad_rule])


@pytest.mark.parametrize(
    "bad_rule",
    [
        PolicyRule(
            name="no-regime",
            regimes=[],
            condition=PolicyCondition("R", 0, ">", 0.5),
            actions=[_action()],
        ),
        PolicyRule(
            name="no-action",
            regimes=["NOMINAL"],
            condition=PolicyCondition("R", 0, ">", 0.5),
            actions=[],
        ),
        PolicyRule(
            name="empty-metric",
            regimes=["NOMINAL"],
            condition=PolicyCondition("", 0, ">", 0.5),
            actions=[_action()],
        ),
        PolicyRule(
            name="nan-threshold",
            regimes=["NOMINAL"],
            condition=PolicyCondition("R", 0, ">", math.nan),
            actions=[_action()],
        ),
        PolicyRule(
            name="bad-op",
            regimes=["NOMINAL"],
            condition=PolicyCondition("R", 0, "!=", 0.5),
            actions=[_action()],
        ),
    ],
)
def test_policy_rule_exports_reject_malformed_rules_before_text_generation(
    bad_rule: PolicyRule,
) -> None:
    with pytest.raises(PolicyError):
        export_policy_rules_prism([bad_rule])
    with pytest.raises(PolicyError):
        export_policy_rules_tla([bad_rule])


def test_stl_export_rejects_bad_syntax_and_keeps_signals_safe() -> None:
    export = export_stl_specs_prism(
        [
            PolicySTLSpec("sync monitor", "always (R >= 0.5 and phase_error < 2.0)"),
            PolicySTLSpec("recovery", "eventually (R_good == 1.0)", severity="hard"),
        ],
        module_name="stl residual",
    )

    assert export.stl_names == {
        "sync monitor": "sync_monitor",
        "recovery": "recovery",
    }
    assert export.metric_names == {
        "R": "R",
        "R_good": "R_good",
        "phase_error": "phase_error",
    }
    assert 'label "stl_sync_monitor_satisfied"' in export.model
    assert "phase_error < 2" in export.model

    with pytest.raises(PolicyError, match="unsupported export syntax"):
        export_stl_specs_prism([PolicySTLSpec("until", "R >= 0.5 until R_good >= 1")])
    with pytest.raises(PolicyError, match="unsupported predicate syntax"):
        export_stl_specs_prism([PolicySTLSpec("bad predicate", "always (R != 0.5)")])
