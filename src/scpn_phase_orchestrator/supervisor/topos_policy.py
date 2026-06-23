# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos policy composition validation

"""Deterministic audit/proof-obligation validation for policy composition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
)

_SCHEMA_NAME = "policy_composition_category"
_SCHEMA_VERSION = "0.1.0"
_PROOF_BOUNDARY = "categorical_validation_prototype_not_formal_topos_proof"
_MAX_COMPOUND_CONDITIONS = 32
_ALLOWED_LOGICS = ("AND", "OR")
_ALLOWED_CONDITION_OPS = (">", ">=", "<", "<=", "==")

__all__ = [
    "PolicyCompositionMorphism",
    "PolicyCompositionObligation",
    "PolicyCompositionObject",
    "PolicyCompositionValidationReport",
    "validate_policy_composition_category",
]


@dataclass(frozen=True)
class PolicyCompositionObject:
    """Stable policy composition object derived from a PolicyRule."""

    name: str
    regimes: tuple[str, ...]
    action_labels: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "name": self.name,
            "regimes": list(self.regimes),
            "action_labels": list(self.action_labels),
        }


@dataclass(frozen=True)
class PolicyCompositionMorphism:
    """Deterministic relation between composition objects and labelled action slots."""

    source: str
    target: str
    label: str
    deterministic: bool = True

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class PolicyCompositionObligation:
    """Review-only proof obligation outcome."""

    name: str
    status: str
    evidence: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "name": self.name,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class PolicyCompositionValidationReport:
    """JSON-safe deterministic validation report for policy composition."""

    schema_name: str
    schema_version: str
    object_count: int
    morphism_count: int
    obligation_records: tuple[PolicyCompositionObligation, ...]
    objects: tuple[PolicyCompositionObject, ...]
    morphisms: tuple[PolicyCompositionMorphism, ...]
    passed: bool
    report_hash: str
    proof_boundary: str
    non_actuating: bool = True

    def to_audit_record(self) -> dict[str, object]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, object]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "object_count": self.object_count,
            "morphism_count": self.morphism_count,
            "obligation_records": [
                obligation.to_audit_record() for obligation in self.obligation_records
            ],
            "objects": [obj.to_audit_record() for obj in self.objects],
            "morphisms": [morphism.to_audit_record() for morphism in self.morphisms],
            "passed": self.passed,
            "report_hash": self.report_hash,
            "proof_boundary": self.proof_boundary,
            "non_actuating": self.non_actuating,
        }


def _build_report_hash(record: dict[str, object]) -> str:
    """Build a deterministic SHA-256 hash of the audit record payload."""
    payload = dict(record)
    payload.pop("report_hash", None)
    digestable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(digestable.encode("utf-8")).hexdigest()


def _as_action_label(action: PolicyAction) -> str:
    """Create a deterministic action label for morphism naming."""
    if not isinstance(action, PolicyAction):
        raise ValueError("policy actions must be PolicyAction objects")
    value = _as_finite_real(action.value, "policy action value")
    ttl = _as_finite_real(action.ttl_s, "policy action ttl_s", allow_negative=False)
    if not isinstance(action.knob, str) or not action.knob.strip():
        raise ValueError("policy action knob must be a non-empty string")
    if not isinstance(action.scope, str) or not action.scope.strip():
        raise ValueError("policy action scope must be a non-empty string")
    knob = action.knob.strip()
    scope = action.scope.strip()
    return f"action[{knob}|{scope}|{value:.17g}|{ttl:.17g}]"


def _as_finite_real(
    value: Any,
    message_prefix: str,
    *,
    allow_negative: bool = True,
) -> float:
    """Validate a finite real scalar and return a float."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{message_prefix} must be a finite real")
    parsed = float(value)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        raise ValueError(f"{message_prefix} must be a finite real")
    if not allow_negative and parsed < 0:
        raise ValueError(f"{message_prefix} must be non-negative")
    return parsed


def _normalised_logic(raw_logic: str) -> str:
    """Return the normalised compound-logic operator for a policy."""
    if not isinstance(raw_logic, str):
        raise ValueError("compound condition logic must be a string")
    logic = raw_logic.strip().upper()
    if logic not in _ALLOWED_LOGICS:
        raise ValueError("compound condition logic must be AND or OR")
    return logic


def _validate_policy_condition(condition: PolicyCondition) -> None:
    """Validate an atomic policy predicate used in a composition obligation."""
    if not isinstance(condition, PolicyCondition):
        raise ValueError("condition members must be PolicyCondition")
    if not isinstance(condition.metric, str) or not condition.metric.strip():
        raise ValueError("policy condition metric must be a non-empty string")
    if condition.op not in _ALLOWED_CONDITION_OPS:
        raise ValueError("policy condition op must be one of >, >=, <, <=, ==")
    if condition.layer is not None and (
        isinstance(condition.layer, bool)
        or not isinstance(condition.layer, Integral)
        or condition.layer < 0
    ):
        raise ValueError(
            "policy condition layer must be a non-negative integer or None"
        )
    _as_finite_real(condition.threshold, "policy condition threshold")


def _is_non_empty_str_list(value: Any) -> tuple[bool, tuple[str, ...], str | None]:
    """Return whether the value is a non-empty list of strings."""
    if not isinstance(value, list) and not isinstance(value, tuple):
        return False, (), "must be a list or tuple"
    if not value:
        return False, (), "must be non-empty"
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return False, (), "must contain non-empty strings only"
        cleaned.append(item.strip())
    return True, tuple(cleaned), None


def _add_obligation(
    obligations: list[PolicyCompositionObligation],
    *,
    name: str,
    passed: bool,
    evidence: str,
) -> None:
    """Append a proof obligation to the obligation list."""
    obligations.append(
        PolicyCompositionObligation(
            name=name,
            status="passed" if passed else "failed",
            evidence=evidence,
        )
    )


def validate_policy_composition_category(
    rules: tuple[PolicyRule, ...] | list[PolicyRule],
) -> PolicyCompositionValidationReport:
    """Validate PolicyRule collections as a categorical composition proof boundary.

    Parameters
    ----------
    rules : tuple[PolicyRule, ...] | list[PolicyRule]
        The policy rules to export or validate.

    Returns
    -------
    PolicyCompositionValidationReport
        The categorical composition validation report.

    Raises
    ------
    ValueError
        If the rules violate the composition proof boundary.
    """
    if isinstance(rules, tuple | list):
        rule_list = list(rules)
    else:
        raise ValueError("rules must be a tuple or list of PolicyRule objects")

    if not rule_list:
        raise ValueError("rules collection must be non-empty")

    if not all(isinstance(rule, PolicyRule) for rule in rule_list):
        raise ValueError("rules must contain only PolicyRule objects")

    canonical_rule_names: dict[int, str] = {}
    obligations: list[PolicyCompositionObligation] = []
    _add_obligation(
        obligations,
        name="rules_collection_valid",
        passed=True,
        evidence="rules collection is a non-empty tuple/list of PolicyRule",
    )

    name_failures: set[str] = set()
    raw_names = [rule.name for rule in rule_list]
    normalized_names: list[str] = []
    for rule, raw_name in zip(rule_list, raw_names, strict=True):
        if not isinstance(raw_name, str):
            raise ValueError("rule names must be strings")
        normalized_name = raw_name.strip()
        if not normalized_name:
            raise ValueError("rule names must be non-empty")
        canonical_rule_names[id(rule)] = normalized_name
        normalized_names.append(normalized_name)

    duplicates = {
        name for name in set(normalized_names) if normalized_names.count(name) > 1
    }
    if duplicates:
        name_failures.update(duplicates)
        _add_obligation(
            obligations,
            name="rule_names_unique",
            passed=False,
            evidence="rule names must be unique and stable",
        )
    else:
        _add_obligation(
            obligations,
            name="rule_names_unique",
            passed=True,
            evidence="rule names are unique",
        )

    objects: list[PolicyCompositionObject] = []
    morphisms: list[PolicyCompositionMorphism] = []
    morphism_labels: set[str] = set()
    all_ok = True

    for rule in sorted(rule_list, key=lambda item: canonical_rule_names[id(item)]):
        rule_name = canonical_rule_names[id(rule)]
        rule_ok = True
        if rule_name in name_failures:
            rule_ok = False

        regimes_ok, regime_values, regime_evidence = _is_non_empty_str_list(
            rule.regimes
        )
        if not regimes_ok:
            rule_ok = False
            _add_obligation(
                obligations,
                name=f"rule.{rule_name}.regimes",
                passed=False,
                evidence=f"invalid regimes: {regime_evidence}",
            )
        else:
            _add_obligation(
                obligations,
                name=f"rule.{rule_name}.regimes",
                passed=True,
                evidence="deterministic and non-empty regimes list",
            )

        try:
            cond_logic = None
            if isinstance(rule.condition, CompoundCondition):
                cond_logic = _normalised_logic(rule.condition.logic)
                cond_conditions = rule.condition.conditions
                if not isinstance(cond_conditions, list) and not isinstance(
                    cond_conditions, tuple
                ):
                    raise ValueError(
                        "compound condition entries must be a list or tuple"
                    )
                if not cond_conditions:
                    raise ValueError("compound condition must contain conditions")
                if len(cond_conditions) > _MAX_COMPOUND_CONDITIONS:
                    raise ValueError("compound condition is unbounded")
                if any(
                    not isinstance(cond, PolicyCondition) for cond in cond_conditions
                ):
                    raise ValueError(
                        "compound condition members must be PolicyCondition"
                    )
                for cond in cond_conditions:
                    _validate_policy_condition(cond)
                _add_obligation(
                    obligations,
                    name=f"rule.{rule_name}.condition",
                    passed=True,
                    evidence=f"compound condition with {cond_logic}",
                )
            elif isinstance(rule.condition, PolicyCondition):
                cond_logic = "ATOMIC"
                _validate_policy_condition(rule.condition)
                _add_obligation(
                    obligations,
                    name=f"rule.{rule_name}.condition",
                    passed=True,
                    evidence="atomic policy condition",
                )
            else:
                raise ValueError(
                    "condition must be PolicyCondition or CompoundCondition"
                )
        except ValueError as error:
            rule_ok = False
            _add_obligation(
                obligations,
                name=f"rule.{rule_name}.condition",
                passed=False,
                evidence=str(error),
            )
            cond_logic = None

        actions = rule.actions
        action_labels: list[str] = []
        if not isinstance(actions, list):
            rule_ok = False
            _add_obligation(
                obligations,
                name=f"rule.{rule_name}.actions",
                passed=False,
                evidence="actions must be a list",
            )
        else:
            try:
                if not actions:
                    raise ValueError("at least one action is required")
                for action in actions:
                    action_labels.append(_as_action_label(action))
                _add_obligation(
                    obligations,
                    name=f"rule.{rule_name}.actions",
                    passed=True,
                    evidence=f"{len(action_labels)} deterministic action label(s)",
                )
            except ValueError as error:
                rule_ok = False
                _add_obligation(
                    obligations,
                    name=f"rule.{rule_name}.actions",
                    passed=False,
                    evidence=str(error),
                )

        if rule_ok and cond_logic is not None and action_labels and regimes_ok:
            norm_regimes = tuple(sorted(set(regime_values)))
            obj = PolicyCompositionObject(
                name=rule_name,
                regimes=norm_regimes,
                action_labels=tuple(sorted(action_labels)),
            )
            objects.append(obj)

            for regime in obj.regimes:
                for action_label in obj.action_labels:
                    morphism_label = f"{obj.name}|{regime}|{action_label}"
                    if morphism_label in morphism_labels:
                        continue
                    morphism_labels.add(morphism_label)
                    morphisms.append(
                        PolicyCompositionMorphism(
                            source=obj.name,
                            target=obj.name,
                            label=morphism_label,
                        )
                    )

        all_ok = all_ok and rule_ok

    obligations = sorted(obligations, key=lambda item: item.name)
    objects = sorted(objects, key=lambda item: item.name)
    morphisms = sorted(
        morphisms,
        key=lambda item: (item.source, item.target, item.label),
    )
    report = PolicyCompositionValidationReport(
        schema_name=_SCHEMA_NAME,
        schema_version=_SCHEMA_VERSION,
        object_count=len(objects),
        morphism_count=len(morphisms),
        obligation_records=tuple(obligations),
        objects=tuple(objects),
        morphisms=tuple(morphisms),
        passed=all_ok and "failed" not in [ob.status for ob in obligations],
        report_hash="",
        proof_boundary=_PROOF_BOUNDARY,
        non_actuating=True,
    )
    return PolicyCompositionValidationReport(
        schema_name=report.schema_name,
        schema_version=report.schema_version,
        object_count=report.object_count,
        morphism_count=report.morphism_count,
        obligation_records=report.obligation_records,
        objects=report.objects,
        morphisms=report.morphisms,
        passed=report.passed,
        report_hash=_build_report_hash(report.to_audit_record()),
        proof_boundary=report.proof_boundary,
        non_actuating=report.non_actuating,
    )
