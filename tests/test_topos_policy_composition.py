# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos policy composition validation tests

"""Tests for deterministic policy-composition validation."""

from __future__ import annotations

import hashlib
import json

import pytest

import scpn_phase_orchestrator.supervisor.topos_policy as topos_policy
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
)


def _rule(
    name: str,
    regimes: list[str],
    condition: PolicyCondition | CompoundCondition,
    *,
    actions: list[PolicyAction],
) -> PolicyRule:
    return PolicyRule(
        name=name,
        regimes=regimes,
        condition=condition,
        actions=actions,
    )


def _hash(record: dict) -> str:
    normalized = dict(record)
    normalized.pop("report_hash", None)
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_rule(
    name: str,
    regimes: list[str],
    metric: str = "R",
    layer: int | None = 0,
    op: str = ">",
    threshold: float = 0.5,
    ttl_s: float = 10.0,
    value: float = 0.1,
) -> PolicyRule:
    return _rule(
        name=name,
        regimes=regimes,
        condition=PolicyCondition(
            metric=metric,
            layer=layer,
            op=op,
            threshold=threshold,
        ),
        actions=[PolicyAction(knob="K", scope="global", value=value, ttl_s=ttl_s)],
    )


def test_validate_policy_composition_category_passes_for_valid_rules():
    rules = [
        _atomic_rule("alpha", ["NOMINAL", "CRITICAL"]),
        _rule(
            "beta",
            ["NOMINAL"],
            CompoundCondition(
                logic="AND",
                conditions=[
                    PolicyCondition(metric="R", layer=0, op=">", threshold=0.5),
                    PolicyCondition(metric="R", layer=0, op="<", threshold=1.0),
                ],
            ),
            actions=[
                PolicyAction(knob="A", scope="layer_0", value=1.0, ttl_s=5.0),
                PolicyAction(knob="B", scope="layer_1", value=2.0, ttl_s=7.0),
            ],
        ),
    ]

    report = topos_policy.validate_policy_composition_category(rules)
    payload = report.to_audit_record()

    assert report.passed is True
    assert report.non_actuating is True
    assert report.object_count == 2
    assert report.morphism_count == 4
    assert report.proof_boundary == (
        "categorical_validation_prototype_not_formal_topos_proof"
    )
    assert report.schema_name == "policy_composition_category"
    assert report.schema_version == "0.1.0"
    assert _hash(payload) == report.report_hash
    assert any(
        item["name"] == "rules_collection_valid"
        for item in payload["obligation_records"]
    )


def test_validate_policy_composition_category_deterministic_across_shapes():
    rule_a = _atomic_rule("alpha", ["NOMINAL"], ttl_s=1.0, value=0.1)
    rule_b = _atomic_rule("beta", ["NOMINAL", "CRITICAL"], ttl_s=2.0, value=0.2)
    ordered = [rule_a, rule_b]
    shuffled = [rule_b, rule_a]

    report_list = topos_policy.validate_policy_composition_category(ordered)
    report_tuple = topos_policy.validate_policy_composition_category(tuple(shuffled))

    assert report_list.report_hash == report_tuple.report_hash
    assert report_list.object_count == report_tuple.object_count == 2
    assert report_list.morphism_count == report_tuple.morphism_count == 3

    json_payload = json.dumps(report_list.to_audit_record())
    assert json_payload == json.dumps(json.loads(json_payload))


def test_invalid_rule_collections_fail_closed():
    with pytest.raises(ValueError, match="rules must be"):
        topos_policy.validate_policy_composition_category(None)

    with pytest.raises(ValueError, match="rules must contain only PolicyRule"):
        topos_policy.validate_policy_composition_category([object()])

    with pytest.raises(ValueError, match="non-empty"):
        topos_policy.validate_policy_composition_category([])


def test_duplicate_rule_names_or_invalid_compound_logic_cannot_pass():
    duplicate_report = topos_policy.validate_policy_composition_category(
        [
            _atomic_rule("dup", ["NOMINAL"]),
            _atomic_rule("dup", ["NOMINAL"]),
        ]
    )
    assert duplicate_report.passed is False
    assert any(
        item["name"] == "rule_names_unique" and item["status"] == "failed"
        for item in duplicate_report.to_audit_record()["obligation_records"]
    )

    invalid_logic_report = topos_policy.validate_policy_composition_category(
        [
            _rule(
                "broken",
                ["NOMINAL"],
                CompoundCondition(
                    logic="XOR",
                    conditions=[
                        PolicyCondition(metric="R", layer=0, op=">", threshold=0.1),
                    ],
                ),
                actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
            )
        ]
    )
    assert invalid_logic_report.passed is False
    assert any(
        item["name"] == "rule.broken.condition" and item["status"] == "failed"
        for item in invalid_logic_report.to_audit_record()["obligation_records"]
    )


@pytest.mark.parametrize(
    ("condition", "evidence"),
    [
        (
            PolicyCondition(metric="", layer=0, op=">", threshold=0.1),
            "metric",
        ),
        (
            PolicyCondition(metric="R", layer=True, op=">", threshold=0.1),
            "layer",
        ),
        (
            PolicyCondition(metric="R", layer=-1, op=">", threshold=0.1),
            "layer",
        ),
        (
            PolicyCondition(metric="R", layer=0, op="!=", threshold=0.1),
            "op",
        ),
        (
            PolicyCondition(metric="R", layer=0, op=">", threshold=True),
            "threshold",
        ),
        (
            PolicyCondition(metric="R", layer=0, op=">", threshold=float("nan")),
            "threshold",
        ),
    ],
)
def test_invalid_atomic_condition_contracts_cannot_pass(
    condition: PolicyCondition,
    evidence: str,
) -> None:
    report = topos_policy.validate_policy_composition_category(
        [
            _rule(
                "broken",
                ["NOMINAL"],
                condition,
                actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
            )
        ]
    )

    condition_obligation = next(
        item
        for item in report.to_audit_record()["obligation_records"]
        if item["name"] == "rule.broken.condition"
    )
    assert report.passed is False
    assert condition_obligation["status"] == "failed"
    assert evidence in str(condition_obligation["evidence"])


def test_invalid_compound_condition_member_contract_cannot_pass() -> None:
    report = topos_policy.validate_policy_composition_category(
        [
            _rule(
                "compound_broken",
                ["NOMINAL"],
                CompoundCondition(
                    logic="AND",
                    conditions=[
                        PolicyCondition(metric="R", layer=0, op=">", threshold=0.1),
                        PolicyCondition(metric="R", layer=True, op=">", threshold=0.2),
                    ],
                ),
                actions=[PolicyAction(knob="K", scope="global", value=0.1, ttl_s=1.0)],
            )
        ]
    )

    condition_obligation = next(
        item
        for item in report.to_audit_record()["obligation_records"]
        if item["name"] == "rule.compound_broken.condition"
    )
    assert report.passed is False
    assert condition_obligation["status"] == "failed"
    assert "layer" in str(condition_obligation["evidence"])


def test_invalid_action_member_contract_cannot_pass() -> None:
    report = topos_policy.validate_policy_composition_category(
        [
            _rule(
                "bad_action",
                ["NOMINAL"],
                PolicyCondition(metric="R", layer=0, op=">", threshold=0.1),
                actions=[object()],
            )
        ]
    )

    action_obligation = next(
        item
        for item in report.to_audit_record()["obligation_records"]
        if item["name"] == "rule.bad_action.actions"
    )
    assert report.passed is False
    assert action_obligation["status"] == "failed"
    assert "PolicyAction" in str(action_obligation["evidence"])


def test_validation_report_does_not_emit_control_actions(monkeypatch):
    class SpyControlAction:
        def __init__(self, *args, **kwargs):
            raise AssertionError("validation should not instantiate actuation actions")

    monkeypatch.setattr(topos_policy, "ControlAction", SpyControlAction, raising=False)
    report = topos_policy.validate_policy_composition_category(
        [_atomic_rule("alpha", ["NOMINAL"])]
    )
    assert report.passed is True
    assert report.non_actuating is True
