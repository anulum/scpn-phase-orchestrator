# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for value-alignment supervisor guard

from __future__ import annotations

from typing import get_type_hints

import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import (
    ValueAlignmentDecision,
    ValueAlignmentGuard,
    ValueAlignmentPolicy,
    ValueConstraint,
    value_alignment_policy_from_binding_spec,
    value_alignment_policy_from_template,
)


def _action(knob: str = "K", value: float = 0.05) -> ControlAction:
    return ControlAction(
        knob=knob,
        scope="global",
        value=value,
        ttl_s=5.0,
        justification="test proposal",
    )


class TestValueAlignmentContracts:
    def test_public_contracts_are_typed(self) -> None:
        hints = get_type_hints(ValueAlignmentGuard.evaluate)

        assert "ControlAction" in str(hints["actions"])
        assert hints["return"] is ValueAlignmentDecision

    def test_empty_constraint_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ValueAlignmentPolicy(constraints=())

    def test_bad_constraint_bounds_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="<="):
            ValueConstraint("bad", min_value=1.0, max_value=0.0)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"name": ""}, "non-empty"),
            ({"name": "bad-abs", "max_abs_value": -0.1}, "non-negative"),
            ({"name": "bad-weight", "weight": -0.1}, "weight"),
            ({"name": "nan-bound", "max_value": float("nan")}, "max_value"),
        ],
    )
    def test_constraint_validation_rejects_unsafe_policy_shapes(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            ValueConstraint(**kwargs)

    def test_bad_policy_score_is_rejected(self) -> None:
        constraint = ValueConstraint("limit", max_abs_value=1.0)

        with pytest.raises(ValueError, match="minimum_score"):
            ValueAlignmentPolicy((constraint,), minimum_score=1.5)


class TestValueAlignmentBehaviour:
    def test_compliant_action_is_approved(self) -> None:
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("limit-K", knob="K", max_abs_value=0.1),)
            )
        )

        decision = guard.evaluate([_action(value=0.05)])

        assert decision.satisfied
        assert decision.actions_to_apply == decision.approved_actions
        assert len(decision.approved_actions) == 1
        assert not decision.blocked_actions
        assert not decision.violations

    def test_violating_action_is_blocked_and_fallback_applies(self) -> None:
        fallback = ControlAction(
            knob="zeta",
            scope="global",
            value=0.0,
            ttl_s=1.0,
            justification="alignment fallback: hold",
        )
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("limit-K", knob="K", max_abs_value=0.1),),
                fallback_actions=(fallback,),
            )
        )

        decision = guard.evaluate([_action(value=0.2)])

        assert not decision.satisfied
        assert decision.actions_to_apply == (fallback,)
        assert decision.blocked_actions == (_action(value=0.2),)
        assert decision.violations[0].constraint == "limit-K"
        assert decision.violations[0].failed_bounds == ("max_abs_value",)

    def test_wildcard_constraint_applies_to_all_knobs(self) -> None:
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("small-actuation", max_abs_value=0.05),)
            )
        )

        decision = guard.evaluate([_action("zeta", 0.06)])

        assert not decision.satisfied
        assert decision.violations[0].knob == "zeta"

    def test_minimum_bound_violation_is_blocked_fail_closed(self) -> None:
        fallback = ControlAction(
            knob="K",
            scope="global",
            value=0.0,
            ttl_s=1.0,
            justification="alignment fallback: restore lower bound",
        )
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("floor-K", knob="K", min_value=0.1),),
                fallback_actions=(fallback,),
            )
        )

        decision = guard.evaluate([_action(value=0.05)])
        record = decision.to_audit_record()

        assert not decision.satisfied
        assert decision.blocked_actions == (_action(value=0.05),)
        assert decision.actions_to_apply == (fallback,)
        assert decision.violations[0].failed_bounds == ("min_value",)
        assert record["actions_to_apply"][0] == {
            "knob": "K",
            "scope": "global",
            "value": 0.0,
            "ttl_s": 1.0,
            "justification": "alignment fallback: restore lower bound",
        }

    def test_non_applicable_constraint_does_not_penalise_action(self) -> None:
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("limit-K", knob="K", max_abs_value=0.1),)
            )
        )

        decision = guard.evaluate([_action("zeta", 0.5)])

        assert decision.satisfied
        assert decision.alignment_score == pytest.approx(1.0)

    def test_score_uses_tightest_lower_and_upper_bound_margin(self) -> None:
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(
                    ValueConstraint(
                        "operating-window",
                        knob="K",
                        min_value=0.2,
                        max_value=0.8,
                    ),
                )
            )
        )

        decision = guard.evaluate([_action(value=0.5)])

        assert decision.satisfied
        assert decision.alignment_score == pytest.approx(0.3)
        assert decision.actions_to_apply == (_action(value=0.5),)

    def test_minimum_score_can_force_fallback_without_bound_violation(self) -> None:
        fallback = ControlAction(
            knob="zeta",
            scope="global",
            value=0.0,
            ttl_s=1.0,
            justification="alignment fallback: hold",
        )
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("prefer-small", max_abs_value=1.0),),
                fallback_actions=(fallback,),
                minimum_score=0.96,
            )
        )

        decision = guard.evaluate([_action(value=0.05)])

        assert not decision.satisfied
        assert not decision.violations
        assert decision.actions_to_apply == (fallback,)
        assert decision.score_counterfactuals[0].counterfactual == (
            "fallback_applied_because_alignment_score_below_policy_minimum"
        )

    def test_score_threshold_counterfactual_is_serialised_for_audit(self) -> None:
        fallback = ControlAction(
            knob="zeta",
            scope="global",
            value=0.0,
            ttl_s=1.0,
            justification="alignment fallback: hold",
        )
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("prefer-small", max_abs_value=1.0),),
                fallback_actions=(fallback,),
                minimum_score=0.96,
            )
        )

        record = guard.evaluate([_action(value=0.05)]).to_audit_record()

        assert record["satisfied"] is False
        assert record["score_counterfactuals"] == [
            {
                "observed_score": pytest.approx(0.95),
                "required_score": pytest.approx(0.96),
                "counterfactual": (
                    "fallback_applied_because_alignment_score_below_policy_minimum"
                ),
            }
        ]
        assert record["actions_to_apply"][0]["justification"] == (
            "alignment fallback: hold"
        )

    def test_audit_record_contains_counterfactual_violation(self) -> None:
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("limit-K", knob="K", max_value=0.1),)
            )
        )

        record = guard.evaluate([_action(value=0.2)]).to_audit_record()

        assert record["satisfied"] is False
        assert record["blocked_count"] == 1
        assert record["violations"][0]["counterfactual"] == (
            "blocked_action_prevents_constraint_violation"
        )
        assert record["score_counterfactuals"] == []
        assert record["actions_to_apply"] == []

    def test_template_loader_builds_policy_with_fallback_actions(self) -> None:
        template = {
            "minimum_score": 0.8,
            "constraints": [
                {
                    "name": "limit-coupling",
                    "knob": "K",
                    "scope": "global",
                    "max_abs_value": 0.1,
                    "weight": 2.0,
                }
            ],
            "fallback_actions": [
                {
                    "knob": "zeta",
                    "scope": "global",
                    "value": 0.0,
                    "ttl_s": 1.0,
                    "justification": "value guard safe hold",
                }
            ],
        }

        policy = value_alignment_policy_from_template(template)
        decision = ValueAlignmentGuard(policy).evaluate([_action(value=0.2)])

        assert policy.minimum_score == pytest.approx(0.8)
        assert policy.constraints[0].name == "limit-coupling"
        assert not decision.satisfied
        assert decision.actions_to_apply[0].justification == "value guard safe hold"

    def test_binding_spec_template_loader_returns_none_when_absent(self) -> None:
        class Spec:
            value_alignment: dict[str, object] = {}

        assert value_alignment_policy_from_binding_spec(Spec()) is None

    def test_binding_spec_template_loader_rejects_non_mapping_value_alignment(
        self,
    ) -> None:
        class Spec:
            value_alignment = ["not", "a", "mapping"]

        with pytest.raises(ValueError, match="must be a mapping"):
            value_alignment_policy_from_binding_spec(Spec())

    def test_binding_spec_template_loader_uses_value_alignment_mapping(self) -> None:
        class Spec:
            value_alignment = {
                "constraints": [
                    {"name": "limit-alpha", "knob": "alpha", "max_value": 0.2}
                ]
            }

        policy = value_alignment_policy_from_binding_spec(Spec())

        assert policy is not None
        assert policy.constraints[0].knob == "alpha"

    def test_template_loader_rejects_malformed_entries(self) -> None:
        with pytest.raises(ValueError, match="constraints"):
            value_alignment_policy_from_template({"constraints": ["not-a-map"]})

        with pytest.raises(ValueError, match="numeric"):
            value_alignment_policy_from_template(
                {"minimum_score": "high", "constraints": []}
            )

    @pytest.mark.parametrize(
        "template",
        [
            {"constraints": {"name": "limit-K"}},
            {"constraints": [{"name": ""}]},
            {"constraints": [{"name": "limit-K"}], "fallback_actions": "hold"},
            {
                "constraints": [{"name": "limit-K"}],
                "fallback_actions": [
                    {
                        "knob": "",
                        "scope": "global",
                        "value": 0.0,
                        "ttl_s": 1.0,
                        "justification": "value guard safe hold",
                    }
                ],
            },
        ],
    )
    def test_template_loader_rejects_invalid_collection_and_string_shapes(
        self, template: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError, match="value_alignment"):
            value_alignment_policy_from_template(template)
