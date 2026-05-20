# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule residual boundary tests

from __future__ import annotations

import textwrap

import numpy as np
import pytest

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


def _state() -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=0.9, psi=0.0), LayerState(R=0.2, psi=0.0)],
        cross_layer_alignment=np.eye(2),
        stability_proxy=0.55,
        regime_id="test",
    )


def _policy_yaml(body: str, tmp_path) -> str:
    path = tmp_path / "policy.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def _rule(condition: PolicyCondition | CompoundCondition) -> PolicyRule:
    return PolicyRule(
        name="boundary",
        regimes=["NOMINAL"],
        condition=condition,
        actions=[PolicyAction("K", "global", 0.1, ttl_s=1.0)],
    )


def test_policy_engine_rejects_bad_compound_logic() -> None:
    engine = PolicyEngine(
        [
            _rule(
                CompoundCondition(
                    [PolicyCondition("R", 0, ">", 0.5)],
                    logic="XOR",
                )
            )
        ]
    )

    with pytest.raises(ValueError, match="rule.logic must be AND or OR"):
        engine.evaluate(Regime.NOMINAL, _state(), [0], [1])


def test_policy_engine_supports_casefolded_compound_logic() -> None:
    engine = PolicyEngine(
        [
            _rule(
                CompoundCondition(
                    [
                        PolicyCondition("R", layer=0, op=">", threshold=0.5),
                        PolicyCondition("R_good", layer=99, op=">", threshold=0.5),
                    ],
                    logic="or",
                )
            )
        ]
    )

    actions = engine.evaluate(Regime.NOMINAL, _state(), [0], [1])

    assert len(actions) == 1
    assert actions[0].justification == "policy rule: boundary"


def test_policy_engine_compound_or_shortfall_fails_closed() -> None:
    engine = PolicyEngine(
        [
            _rule(
                CompoundCondition(
                    [
                        PolicyCondition("R_good", 4, ">", 0.5),
                        PolicyCondition("missing_metric", None, ">", 0.5),
                    ],
                    logic="OR",
                )
            )
        ]
    )

    assert engine.evaluate(Regime.NOMINAL, _state(), [0], [1]) == []


@pytest.mark.parametrize(
    ("yaml_body", "message"),
    [
        (
            """
            rules:
              - name: bad_logic
                regime: [NOMINAL]
                logic: XOR
                conditions:
                  - {metric: R, layer: 0, op: ">", threshold: 0.5}
                action: {knob: K, scope: global, value: 0.1, ttl_s: 1.0}
            """,
            "rule.logic must be AND or OR",
        ),
        (
            """
            rules:
              - name: bool_max
                regime: [NOMINAL]
                condition: {metric: R, layer: 0, op: ">", threshold: 0.5}
                action: {knob: K, scope: global, value: 0.1, ttl_s: 1.0}
                max_fires: true
            """,
            "rule.max_fires must be a non-negative integer",
        ),
        (
            """
            rules:
              - name: negative_ttl
                regime: [NOMINAL]
                condition: {metric: R, layer: 0, op: ">", threshold: 0.5}
                action: {knob: K, scope: global, value: 0.1, ttl_s: -1.0}
            """,
            "action.ttl_s must be non-negative",
        ),
    ],
)
def test_policy_rule_loader_rejects_malformed_public_contracts(
    tmp_path,
    yaml_body: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_policy_rules(_policy_yaml(yaml_body, tmp_path))


def test_loader_preserves_action_order_and_cooldown_contract(tmp_path) -> None:
    rules = load_policy_rules(
        _policy_yaml(
            """
            rules:
              - name: ordered_actions
                regime: [nominal]
                cooldown_s: 2.5
                max_fires: 1
                condition: {metric: stability_proxy, op: ">=", threshold: 0.5}
                actions:
                  - {knob: K, scope: global, value: 0.1, ttl_s: 1.0}
                  - {knob: damping, scope: layer_1, value: 0.2, ttl_s: 3.0}
            """,
            tmp_path,
        )
    )

    engine = PolicyEngine(rules)
    first = engine.evaluate(Regime.NOMINAL, _state(), [0], [1])
    second = engine.evaluate(Regime.NOMINAL, _state(), [0], [1])
    engine.advance_clock(3.0)
    third = engine.evaluate(Regime.NOMINAL, _state(), [0], [1])

    action_contract = [
        (action.knob, action.scope, action.value, action.ttl_s) for action in first
    ]
    assert action_contract == [
        ("K", "global", 0.1, 1.0),
        ("damping", "layer_1", 0.2, 3.0),
    ]
    assert second == []
    assert third == []
