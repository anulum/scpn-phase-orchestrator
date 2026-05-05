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

    def test_non_applicable_constraint_does_not_penalise_action(self) -> None:
        guard = ValueAlignmentGuard(
            ValueAlignmentPolicy(
                constraints=(ValueConstraint("limit-K", knob="K", max_abs_value=0.1),)
            )
        )

        decision = guard.evaluate([_action("zeta", 0.5)])

        assert decision.satisfied
        assert decision.alignment_score == pytest.approx(1.0)

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
        assert record["actions_to_apply"] == []
