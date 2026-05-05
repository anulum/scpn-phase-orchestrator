# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Value-alignment supervisor guard

"""Pareto-style value guard for supervisor actuation proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from scpn_phase_orchestrator.actuation.mapper import ControlAction

__all__ = [
    "ValueAlignmentDecision",
    "ValueAlignmentGuard",
    "ValueAlignmentPolicy",
    "ValueConstraint",
    "ValueViolation",
]

ActionTuple: TypeAlias = tuple[ControlAction, ...]


@dataclass(frozen=True)
class ValueConstraint:
    """A hard value constraint over a proposed control action.

    ``knob`` and ``scope`` accept ``"*"`` wildcards. Bounds are inclusive.
    ``weight`` controls how much this constraint contributes to the
    reported alignment score; a failed hard constraint always blocks the
    action regardless of weight.
    """

    name: str
    knob: str = "*"
    scope: str = "*"
    min_value: float | None = None
    max_value: float | None = None
    max_abs_value: float | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("constraint name must be non-empty")
        _validate_bound(self.min_value, "min_value")
        _validate_bound(self.max_value, "max_value")
        _validate_bound(self.max_abs_value, "max_abs_value")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be <= max_value")
        if self.max_abs_value is not None and self.max_abs_value < 0.0:
            raise ValueError("max_abs_value must be non-negative")
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("weight must be finite and non-negative")

    def applies_to(self, action: ControlAction) -> bool:
        """Return whether this constraint applies to ``action``."""
        knob_match = self.knob == "*" or self.knob == action.knob
        scope_match = self.scope == "*" or self.scope == action.scope
        return knob_match and scope_match

    def violations_for(self, action: ControlAction) -> tuple[str, ...]:
        """Return failed bound names for ``action``."""
        failures: list[str] = []
        if self.min_value is not None and action.value < self.min_value:
            failures.append("min_value")
        if self.max_value is not None and action.value > self.max_value:
            failures.append("max_value")
        if self.max_abs_value is not None and abs(action.value) > self.max_abs_value:
            failures.append("max_abs_value")
        return tuple(failures)


@dataclass(frozen=True)
class ValueViolation:
    """A blocked action and the value constraint it violated."""

    constraint: str
    knob: str
    scope: str
    proposed_value: float
    failed_bounds: tuple[str, ...]
    counterfactual: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable violation record."""
        return {
            "constraint": self.constraint,
            "knob": self.knob,
            "scope": self.scope,
            "proposed_value": self.proposed_value,
            "failed_bounds": list(self.failed_bounds),
            "counterfactual": self.counterfactual,
        }


@dataclass(frozen=True)
class ValueAlignmentPolicy:
    """Configured objective constraints and fallback actuation."""

    constraints: tuple[ValueConstraint, ...]
    fallback_actions: ActionTuple = ()
    minimum_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("at least one value constraint is required")
        if not np.isfinite(self.minimum_score) or not 0.0 <= self.minimum_score <= 1.0:
            raise ValueError("minimum_score must be finite and in [0, 1]")


@dataclass(frozen=True)
class ValueAlignmentDecision:
    """Result of applying value constraints to proposed actions."""

    approved_actions: ActionTuple
    blocked_actions: ActionTuple
    fallback_actions: ActionTuple
    violations: tuple[ValueViolation, ...]
    alignment_score: float
    minimum_score: float

    @property
    def satisfied(self) -> bool:
        """Return whether the proposed action set passed the guard."""
        return not self.violations and self.alignment_score >= self.minimum_score

    @property
    def actions_to_apply(self) -> ActionTuple:
        """Return approved actions or the forced safe fallback path."""
        if self.satisfied:
            return self.approved_actions
        return self.fallback_actions

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable guard decision."""
        return {
            "satisfied": self.satisfied,
            "alignment_score": self.alignment_score,
            "minimum_score": self.minimum_score,
            "approved_count": len(self.approved_actions),
            "blocked_count": len(self.blocked_actions),
            "fallback_count": len(self.fallback_actions),
            "violations": [
                violation.to_audit_record() for violation in self.violations
            ],
            "actions_to_apply": [
                _action_record(action) for action in self.actions_to_apply
            ],
        }


class ValueAlignmentGuard:
    """Block supervisor actions that violate configured value constraints."""

    def __init__(self, policy: ValueAlignmentPolicy) -> None:
        self.policy = policy

    def evaluate(
        self, actions: list[ControlAction] | ActionTuple
    ) -> ValueAlignmentDecision:
        """Evaluate proposed actions and return an auditable decision."""
        proposed = tuple(actions)
        approved: list[ControlAction] = []
        blocked: list[ControlAction] = []
        violations: list[ValueViolation] = []
        scores: list[float] = []

        for action in proposed:
            action_violations = self._violations_for_action(action)
            if action_violations:
                blocked.append(action)
                violations.extend(action_violations)
                scores.append(0.0)
            else:
                approved.append(action)
                scores.append(self._score_action(action))

        alignment_score = 1.0 if not scores else float(min(scores))
        return ValueAlignmentDecision(
            approved_actions=tuple(approved),
            blocked_actions=tuple(blocked),
            fallback_actions=self.policy.fallback_actions,
            violations=tuple(violations),
            alignment_score=alignment_score,
            minimum_score=self.policy.minimum_score,
        )

    def _violations_for_action(
        self, action: ControlAction
    ) -> tuple[ValueViolation, ...]:
        violations: list[ValueViolation] = []
        for constraint in self.policy.constraints:
            if not constraint.applies_to(action):
                continue
            failed = constraint.violations_for(action)
            if failed:
                violations.append(
                    ValueViolation(
                        constraint=constraint.name,
                        knob=action.knob,
                        scope=action.scope,
                        proposed_value=float(action.value),
                        failed_bounds=failed,
                        counterfactual=("blocked_action_prevents_constraint_violation"),
                    )
                )
        return tuple(violations)

    def _score_action(self, action: ControlAction) -> float:
        applicable = [
            constraint
            for constraint in self.policy.constraints
            if constraint.applies_to(action) and constraint.weight > 0.0
        ]
        if not applicable:
            return 1.0
        weighted_scores = [
            _constraint_margin(action, constraint) * constraint.weight
            for constraint in applicable
        ]
        total_weight = sum(constraint.weight for constraint in applicable)
        if total_weight <= 0.0:
            return 1.0
        return float(np.clip(sum(weighted_scores) / total_weight, 0.0, 1.0))


def _constraint_margin(action: ControlAction, constraint: ValueConstraint) -> float:
    margins: list[float] = []
    if constraint.min_value is not None:
        margins.append(
            _one_sided_margin(action.value, constraint.min_value, lower=True)
        )
    if constraint.max_value is not None:
        margins.append(
            _one_sided_margin(action.value, constraint.max_value, lower=False)
        )
    if constraint.max_abs_value is not None:
        margins.append(
            max(
                0.0,
                1.0 - abs(action.value) / max(constraint.max_abs_value, 1e-12),
            )
        )
    return 1.0 if not margins else float(min(margins))


def _one_sided_margin(value: float, bound: float, *, lower: bool) -> float:
    scale = max(abs(bound), 1.0)
    if lower:
        return float(np.clip((value - bound) / scale, 0.0, 1.0))
    return float(np.clip((bound - value) / scale, 0.0, 1.0))


def _action_record(action: ControlAction) -> dict[str, object]:
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": action.value,
        "ttl_s": action.ttl_s,
        "justification": action.justification,
    }


def _validate_bound(value: float | None, name: str) -> None:
    if value is not None and not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
