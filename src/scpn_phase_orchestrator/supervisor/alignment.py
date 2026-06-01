# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Value-alignment supervisor guard

"""Pareto-style value guard for supervisor actuation proposals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import TypeAlias

import numpy as np

from scpn_phase_orchestrator.actuation.mapper import ControlAction

__all__ = [
    "ValueAlignmentDecision",
    "ValueAlignmentGuard",
    "ValueAlignmentPolicy",
    "ValueScoreCounterfactual",
    "ValueParetoObjective",
    "ValueParetoViolation",
    "ValueConstraint",
    "ValueViolation",
    "calibrate_value_alignment_replay_evidence",
    "value_alignment_policy_from_binding_spec",
    "value_alignment_policy_from_template",
]

ActionTuple: TypeAlias = tuple[ControlAction, ...]
ObjectiveDeltas: TypeAlias = Mapping[str, float]


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
        _validate_non_negative_scalar(self.weight, "weight")

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
class ValueScoreCounterfactual:
    """Counterfactual record explaining a score-threshold fallback."""

    observed_score: float
    required_score: float
    counterfactual: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable score-threshold counterfactual."""
        return {
            "observed_score": self.observed_score,
            "required_score": self.required_score,
            "counterfactual": self.counterfactual,
        }


@dataclass(frozen=True)
class ValueParetoObjective:
    """A named objective delta that must stay on the review Pareto frontier."""

    name: str
    min_delta: float = 0.0
    max_regression: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Pareto objective name must be non-empty")
        _validate_non_negative_scalar(self.min_delta, "Pareto objective min_delta")
        _validate_non_negative_scalar(
            self.max_regression, "Pareto objective max_regression"
        )


@dataclass(frozen=True)
class ValueParetoViolation:
    """A failed Pareto objective review condition."""

    objective: str
    observed_delta: float | None
    required_delta: float
    allowed_regression: float
    counterfactual: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a serialisable Pareto violation record."""
        return {
            "objective": self.objective,
            "observed_delta": self.observed_delta,
            "required_delta": self.required_delta,
            "allowed_regression": self.allowed_regression,
            "counterfactual": self.counterfactual,
        }


@dataclass(frozen=True)
class ValueAlignmentPolicy:
    """Configured objective constraints and fallback actuation."""

    constraints: tuple[ValueConstraint, ...]
    fallback_actions: ActionTuple = ()
    minimum_score: float = 0.0
    pareto_objectives: tuple[ValueParetoObjective, ...] = ()

    def __post_init__(self) -> None:
        if not self.constraints:
            raise ValueError("at least one value constraint is required")
        _validate_unit_interval_scalar(self.minimum_score, "minimum_score")
        names = [objective.name for objective in self.pareto_objectives]
        if len(set(names)) != len(names):
            raise ValueError("Pareto objective names must be unique")


@dataclass(frozen=True)
class ValueAlignmentDecision:
    """Result of applying value constraints to proposed actions."""

    approved_actions: ActionTuple
    blocked_actions: ActionTuple
    fallback_actions: ActionTuple
    violations: tuple[ValueViolation, ...]
    score_counterfactuals: tuple[ValueScoreCounterfactual, ...]
    pareto_violations: tuple[ValueParetoViolation, ...]
    alignment_score: float
    minimum_score: float

    @property
    def satisfied(self) -> bool:
        """Return whether the proposed action set passed the guard."""
        return (
            not self.violations
            and not self.pareto_violations
            and self.alignment_score >= self.minimum_score
        )

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
            "pareto_violation_count": len(self.pareto_violations),
            "violations": [
                violation.to_audit_record() for violation in self.violations
            ],
            "pareto_violations": [
                violation.to_audit_record() for violation in self.pareto_violations
            ],
            "score_counterfactuals": [
                counterfactual.to_audit_record()
                for counterfactual in self.score_counterfactuals
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
        self,
        actions: list[ControlAction] | ActionTuple,
        *,
        objective_deltas: ObjectiveDeltas | None = None,
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
        pareto_violations = _pareto_violations(
            self.policy.pareto_objectives,
            objective_deltas,
        )
        return ValueAlignmentDecision(
            approved_actions=tuple(approved),
            blocked_actions=tuple(blocked),
            fallback_actions=self.policy.fallback_actions,
            violations=tuple(violations),
            pareto_violations=pareto_violations,
            score_counterfactuals=_score_counterfactuals(
                alignment_score,
                self.policy.minimum_score,
                violations,
            ),
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


def calibrate_value_alignment_replay_evidence(
    policy: ValueAlignmentPolicy,
    replay_cases: Mapping[str, list[ControlAction] | ActionTuple],
    *,
    evidence_label: str = "value_alignment_replay_calibration",
) -> dict[str, object]:
    """Calibrate a value-alignment policy against replayed action proposals.

    The returned artifact is deterministic and review-only: it records what the
    guard would approve, block, or divert to fallback on replayed cases, but it
    never authorises live actuation. This gives production reviewers evidence
    for guard behaviour before any deployment-tier enforcement is claimed.
    """

    if not isinstance(policy, ValueAlignmentPolicy):
        raise ValueError("policy must be a ValueAlignmentPolicy")
    if not replay_cases:
        raise ValueError(
            "value-alignment calibration requires at least one replay case"
        )
    if not isinstance(evidence_label, str) or not evidence_label:
        raise ValueError("evidence_label must be a non-empty string")

    guard = ValueAlignmentGuard(policy)
    records: list[dict[str, object]] = []
    approved_case_count = 0
    blocked_case_count = 0
    threshold_fallback_case_count = 0
    fallback_applied_case_count = 0

    for case_id, actions in replay_cases.items():
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(
                "value-alignment replay case id must be a non-empty string"
            )
        proposed_actions = _validated_replay_actions(actions, case_id)
        decision = guard.evaluate(proposed_actions)
        audit_record = decision.to_audit_record()

        if decision.satisfied:
            approved_case_count += 1
        if decision.violations:
            blocked_case_count += 1
        if decision.score_counterfactuals:
            threshold_fallback_case_count += 1
        if not decision.satisfied and decision.fallback_actions:
            fallback_applied_case_count += 1

        records.append(
            {
                "case_id": case_id,
                "proposed_action_count": len(proposed_actions),
                "satisfied": decision.satisfied,
                "alignment_score": decision.alignment_score,
                "minimum_score": decision.minimum_score,
                "approved_count": len(decision.approved_actions),
                "blocked_count": len(decision.blocked_actions),
                "fallback_count": len(decision.fallback_actions),
                "violation_count": len(decision.violations),
                "score_counterfactual_count": len(decision.score_counterfactuals),
                "actions_to_apply": audit_record["actions_to_apply"],
                "violations": audit_record["violations"],
                "score_counterfactuals": audit_record["score_counterfactuals"],
            }
        )

    artifact: dict[str, object] = {
        "schema": "scpn_value_alignment_replay_calibration_v1",
        "evidence_label": evidence_label,
        "replay_case_count": len(records),
        "approved_case_count": approved_case_count,
        "blocked_case_count": blocked_case_count,
        "threshold_fallback_case_count": threshold_fallback_case_count,
        "fallback_applied_case_count": fallback_applied_case_count,
        "calibration_actuation_permitted": False,
        "decision_records": records,
    }
    canonical = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    artifact["calibration_sha256"] = sha256(canonical.encode("utf-8")).hexdigest()
    return artifact


def value_alignment_policy_from_binding_spec(
    spec: object,
) -> ValueAlignmentPolicy | None:
    """Build a policy from ``BindingSpec.value_alignment`` when present."""

    template = getattr(spec, "value_alignment", None)
    if not template:
        return None
    if not isinstance(template, Mapping):
        raise ValueError("binding spec value_alignment must be a mapping")
    return value_alignment_policy_from_template(template)


def value_alignment_policy_from_template(
    template: Mapping[str, object],
) -> ValueAlignmentPolicy:
    """Build a value-alignment policy from a binding-spec template mapping.

    Expected shape::

        value_alignment:
          minimum_score: 0.8
          constraints:
            - name: limit-coupling
              knob: K
              max_abs_value: 0.1
          fallback_actions:
            - knob: zeta
              scope: global
              value: 0.0
              ttl_s: 1.0
              justification: safe hold
    """

    constraints = tuple(
        _constraint_from_template(item, index)
        for index, item in enumerate(_template_list(template, "constraints"))
    )
    fallback_actions = tuple(
        _action_from_template(item, index)
        for index, item in enumerate(_template_list(template, "fallback_actions"))
    )
    pareto_objectives = tuple(
        _pareto_objective_from_template(item, index)
        for index, item in enumerate(_template_list(template, "pareto_objectives"))
    )
    minimum_score = _template_float(template.get("minimum_score", 0.0))
    return ValueAlignmentPolicy(
        constraints=constraints,
        fallback_actions=fallback_actions,
        minimum_score=minimum_score,
        pareto_objectives=pareto_objectives,
    )


def _score_counterfactuals(
    alignment_score: float,
    minimum_score: float,
    violations: list[ValueViolation],
) -> tuple[ValueScoreCounterfactual, ...]:
    if violations or alignment_score >= minimum_score:
        return ()
    return (
        ValueScoreCounterfactual(
            observed_score=alignment_score,
            required_score=minimum_score,
            counterfactual=(
                "fallback_applied_because_alignment_score_below_policy_minimum"
            ),
        ),
    )


def _constraint_from_template(item: object, index: int) -> ValueConstraint:
    data = _template_mapping(item, f"constraints[{index}]")
    return ValueConstraint(
        name=_template_string(data.get("name"), f"constraints[{index}].name"),
        knob=_template_string(data.get("knob", "*"), f"constraints[{index}].knob"),
        scope=_template_string(data.get("scope", "*"), f"constraints[{index}].scope"),
        min_value=_optional_template_float(data.get("min_value")),
        max_value=_optional_template_float(data.get("max_value")),
        max_abs_value=_optional_template_float(data.get("max_abs_value")),
        weight=_template_float(data.get("weight", 1.0)),
    )


def _action_from_template(item: object, index: int) -> ControlAction:
    data = _template_mapping(item, f"fallback_actions[{index}]")
    return ControlAction(
        knob=_template_string(data.get("knob"), f"fallback_actions[{index}].knob"),
        scope=_template_string(data.get("scope"), f"fallback_actions[{index}].scope"),
        value=_template_float(data.get("value")),
        ttl_s=_template_float(data.get("ttl_s")),
        justification=_template_string(
            data.get("justification"),
            f"fallback_actions[{index}].justification",
        ),
    )


def _pareto_objective_from_template(item: object, index: int) -> ValueParetoObjective:
    data = _template_mapping(item, f"pareto_objectives[{index}]")
    return ValueParetoObjective(
        name=_template_string(data.get("name"), f"pareto_objectives[{index}].name"),
        min_delta=_template_float(data.get("min_delta", 0.0)),
        max_regression=_template_float(data.get("max_regression", 0.0)),
    )


def _template_list(template: Mapping[str, object], key: str) -> list[object]:
    value = template.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"value_alignment.{key} must be a list")
    return value


def _template_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"value_alignment.{context} must be a mapping")
    return value


def _template_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"value_alignment.{context} must be a non-empty string")
    return value


def _template_float(value: object) -> float:
    if not _is_real_numeric_scalar(value):
        raise ValueError("value_alignment numeric fields must be numbers")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("value_alignment numeric fields must be finite")
    return number


def _optional_template_float(value: object) -> float | None:
    if value is None:
        return None
    return _template_float(value)


def _validated_replay_actions(
    actions: list[ControlAction] | ActionTuple,
    case_id: str,
) -> ActionTuple:
    if not isinstance(actions, list | tuple):
        raise ValueError(f"value-alignment replay case {case_id!r} must be a sequence")
    proposed = tuple(actions)
    if not proposed:
        raise ValueError(f"value-alignment replay case {case_id!r} must not be empty")
    if not all(isinstance(action, ControlAction) for action in proposed):
        raise ValueError(
            f"value-alignment replay case {case_id!r} must contain ControlAction"
        )
    return proposed


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


def _pareto_violations(
    objectives: tuple[ValueParetoObjective, ...],
    objective_deltas: ObjectiveDeltas | None,
) -> tuple[ValueParetoViolation, ...]:
    if not objectives:
        return ()
    violations: list[ValueParetoViolation] = []
    observed: dict[str, float] = {}
    for objective in objectives:
        raw_delta = (
            None if objective_deltas is None else objective_deltas.get(objective.name)
        )
        if raw_delta is None:
            violations.append(
                ValueParetoViolation(
                    objective=objective.name,
                    observed_delta=None,
                    required_delta=objective.min_delta,
                    allowed_regression=objective.max_regression,
                    counterfactual="missing_pareto_objective_evidence_forces_fallback",
                )
            )
            continue
        delta = _finite_objective_delta(raw_delta, objective.name)
        observed[objective.name] = delta
        if delta < -objective.max_regression:
            violations.append(
                ValueParetoViolation(
                    objective=objective.name,
                    observed_delta=delta,
                    required_delta=objective.min_delta,
                    allowed_regression=objective.max_regression,
                    counterfactual=(
                        "fallback_applied_to_prevent_pareto_objective_regression"
                    ),
                )
            )
    if violations:
        return tuple(violations)
    improving_objectives = [
        objective
        for objective in objectives
        if objective.min_delta > 0.0 and observed[objective.name] >= objective.min_delta
    ]
    if (
        any(objective.min_delta > 0.0 for objective in objectives)
        and not improving_objectives
    ):
        objective = max(objectives, key=lambda item: item.min_delta)
        return (
            ValueParetoViolation(
                objective=objective.name,
                observed_delta=observed[objective.name],
                required_delta=objective.min_delta,
                allowed_regression=objective.max_regression,
                counterfactual=(
                    "fallback_applied_because_no_pareto_objective_improved"
                ),
            ),
        )
    return ()


def _finite_objective_delta(value: object, objective: str) -> float:
    if not _is_real_numeric_scalar(value):
        raise ValueError(f"Pareto objective {objective!r} delta must be numeric")
    delta = float(value)
    if not np.isfinite(delta):
        raise ValueError(f"Pareto objective {objective!r} delta must be finite")
    return delta


def _action_record(action: ControlAction) -> dict[str, object]:
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": action.value,
        "ttl_s": action.ttl_s,
        "justification": action.justification,
    }


def _validate_bound(value: float | None, name: str) -> None:
    if value is None:
        return
    if not _is_real_numeric_scalar(value) or not np.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_non_negative_scalar(value: object, name: str) -> None:
    if not _is_real_numeric_scalar(value):
        raise ValueError(f"{name} must be finite and non-negative")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _validate_unit_interval_scalar(value: object, name: str) -> None:
    if not _is_real_numeric_scalar(value):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _is_real_numeric_scalar(value: object) -> bool:
    return isinstance(value, int | float | np.integer | np.floating) and not isinstance(
        value, bool | np.bool_
    )
