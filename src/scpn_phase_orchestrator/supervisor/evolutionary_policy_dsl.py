# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Offline review-only policy DSL mutation search

"""Policy DSL mutation helpers for offline evolutionary supervisor review."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, Final

__all__ = [
    "PolicyAction",
    "PolicyCondition",
    "PolicyMutationCandidate",
    "PolicyMutationPlan",
    "PolicyMutationSearchConfig",
    "PolicyMutationSearchReport",
    "PolicyRule",
    "parse_policy_dsl",
    "run_offline_evolutionary_policy_dsl_search",
]


_NUMBER_PATTERN: Final[str] = (
    r"[-+]?"  # optional sign
    r"(?:"  # mantissa
    r"\d+\.\d*"  # 1. / 1.0
    r"|"  # ...
    r"\.\d+"  # .5
    r"|"  # ...
    r"\d+"  # 1
    r")"
    r"(?:[eE][-+]?\d+)?"  # optional exponent
)

_RULE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*rule\s+(?P<name>[A-Za-z_][A-Za-z0-9_-]*)\s*:\s*if\s+"
    r"(?P<condition>.+?)\s+then\s+(?P<action>.+?)\s*$",
    re.IGNORECASE,
)
_ACTION_RE: Final[re.Pattern[str]] = re.compile(
    rf"^set\s+(?P<target>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<operator>\+=|-=|=)\s*(?P<value>{_NUMBER_PATTERN})\s*$",
    re.IGNORECASE,
)
_CONDITION_RE: Final[re.Pattern[str]] = re.compile(
    rf"^(?P<metric>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<operator>>=|<=|==|!=|>|<)\s*(?P<value>{_NUMBER_PATTERN})$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyCondition:
    """Atomic predicate in the policy mutation DSL."""

    metric: str
    operator: str
    threshold: float

    def to_dsl(self) -> str:
        """Return the deterministic policy DSL representation.

        Returns
        -------
        str
            Return the deterministic policy DSL representation.
        """
        return f"{self.metric} {self.operator} {_format_float(self.threshold)}"


@dataclass(frozen=True)
class PolicyAction:
    """Bounded action in the policy mutation DSL."""

    target: str
    operator: str
    value: float

    def to_dsl(self) -> str:
        """Return the deterministic policy DSL representation.

        Returns
        -------
        str
            Return the deterministic policy DSL representation.
        """
        return f"set {self.target} {self.operator} {_format_float(self.value)}"


@dataclass(frozen=True)
class PolicyRule:
    """Policy rule composed from conditions and actions."""

    name: str
    conditions: tuple[PolicyCondition, ...]
    action: PolicyAction

    def to_dsl(self) -> str:
        """Return the deterministic policy DSL representation.

        Returns
        -------
        str
            Return the deterministic policy DSL representation.
        """
        condition_text = " and ".join(
            condition.to_dsl() for condition in self.conditions
        )
        return f"rule {self.name}: if {condition_text} then {self.action.to_dsl()}"


@dataclass(frozen=True)
class PolicyMutationSearchConfig:
    """Configuration for deterministic policy mutation search."""

    generation_count: int = 2
    population_size: int = 6
    mutation_step: float = 0.05

    def __post_init__(self) -> None:
        _require_positive_int(self.generation_count, "generation_count")
        _require_positive_int(self.population_size, "population_size")
        _require_finite_positive(self.mutation_step, "mutation_step")


@dataclass(frozen=True)
class PolicyMutationPlan:
    """Planned mutation candidate for policy DSL review."""

    rule_name: str
    component: str
    component_index: int
    operator: str
    original_value: float
    mutated_value: float
    mutation_delta: float


@dataclass(frozen=True)
class PolicyMutationCandidate:
    """One non-actuating policy mutation candidate."""

    candidate_id: str
    generation: int
    mutation_index: int
    source_rule_name: str
    source_rule_text: str
    mutated_rule_text: str
    candidate_policy_dsl: str
    mutation_plan: PolicyMutationPlan
    blocked_reasons: tuple[str, ...]
    candidate_hash: str
    operator_review_required: bool = True
    execution_disabled: bool = True
    live_merge_permitted: bool = False
    hot_patch_permitted: bool = False
    actuation_permitted: bool = False

    @property
    def accepted(self) -> bool:
        """Return whether this candidate is accepted for review.

        Returns
        -------
        bool
            Return whether this candidate is accepted for review.
        """
        return not self.blocked_reasons

    def to_audit_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, Any]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "candidate_id": self.candidate_id,
            "generation": self.generation,
            "mutation_index": self.mutation_index,
            "source_rule_name": self.source_rule_name,
            "source_rule_text": self.source_rule_text,
            "mutated_rule_text": self.mutated_rule_text,
            "candidate_policy_dsl": self.candidate_policy_dsl,
            "mutation_plan": {
                "rule_name": self.mutation_plan.rule_name,
                "component": self.mutation_plan.component,
                "component_index": self.mutation_plan.component_index,
                "operator": self.mutation_plan.operator,
                "original_value": self.mutation_plan.original_value,
                "mutated_value": self.mutation_plan.mutated_value,
                "mutation_delta": self.mutation_plan.mutation_delta,
            },
            "blocked_reasons": list(self.blocked_reasons),
            "candidate_hash": self.candidate_hash,
            "status": "accepted" if self.accepted else "rejected",
            "operator_review_required": self.operator_review_required,
            "execution_disabled": self.execution_disabled,
            "live_merge_permitted": self.live_merge_permitted,
            "hot_patch_permitted": self.hot_patch_permitted,
            "actuation_permitted": self.actuation_permitted,
        }


@dataclass(frozen=True)
class PolicyMutationSearchReport:
    """Aggregate report for policy mutation search."""

    schema_name: str
    schema_version: str
    config: PolicyMutationSearchConfig
    source_policy_dsl: str
    source_policy_hash: str
    candidate_count: int
    accepted_count: int
    rejected_count: int
    candidates: tuple[PolicyMutationCandidate, ...]
    execution_disabled: bool
    hot_patch_permitted: bool
    live_merge_permitted: bool
    actuation_permitted: bool
    operator_review_required: bool
    non_actuating: bool
    report_hash: str

    def to_audit_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, Any]
            Return a deterministic JSON-safe audit record.
        """
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "generation_count": self.config.generation_count,
            "population_size": self.config.population_size,
            "mutation_step": self.config.mutation_step,
            "source_policy_dsl": self.source_policy_dsl,
            "source_policy_hash": self.source_policy_hash,
            "candidate_count": self.candidate_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "candidates": [
                candidate.to_audit_record() for candidate in self.candidates
            ],
            "execution_disabled": self.execution_disabled,
            "hot_patch_permitted": self.hot_patch_permitted,
            "live_merge_permitted": self.live_merge_permitted,
            "actuation_permitted": self.actuation_permitted,
            "operator_review_required": self.operator_review_required,
            "non_actuating": self.non_actuating,
            "report_hash": self.report_hash,
        }


@dataclass(frozen=True)
class _MutationAxis:
    """A named axis along which a policy rule can be mutated."""

    rule_index: int
    rule_name: str
    component: str
    component_index: int
    original_value: float
    operator: str


def parse_policy_dsl(policy_dsl: str) -> tuple[PolicyRule, ...]:
    """Parse immutable rule objects from a compact policy DSL string.

    Parameters
    ----------
    policy_dsl : str
        A compact policy-DSL source string.

    Returns
    -------
    tuple[PolicyRule, ...]
        The immutable policy rules parsed from the DSL.

    Raises
    ------
    ValueError
        If the DSL string is malformed.
    """
    if not isinstance(policy_dsl, str) or not policy_dsl.strip():
        raise ValueError("policy_dsl must be a non-empty string")

    rules: list[PolicyRule] = []
    seen_rule_names: set[str] = set()

    for raw_line in policy_dsl.splitlines():
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue

        match = _RULE_RE.match(content)
        if not match:
            raise ValueError(f"Malformed rule line: {raw_line}")

        name = match.group("name")
        if name in seen_rule_names:
            raise ValueError(f"Duplicate rule name: {name}")

        seen_rule_names.add(name)

        condition_text = match.group("condition").strip()
        action_text = match.group("action").strip()

        conditions = _parse_conditions(condition_text)
        action = _parse_action(action_text)
        rules.append(PolicyRule(name=name, conditions=conditions, action=action))

    if not rules:
        raise ValueError("policy_dsl must contain at least one rule")

    return tuple(rules)


def run_offline_evolutionary_policy_dsl_search(
    policy_dsl: str,
    *,
    generation_count: int = 2,
    population_size: int = 6,
    mutation_step: float = 0.05,
) -> PolicyMutationSearchReport:
    """Generate deterministic offline policy-DSl mutation candidates for review.

    Parameters
    ----------
    policy_dsl : str
        A compact policy-DSL source string.
    generation_count : int
        Number of search generations.
    population_size : int
        Number of candidates per generation.
    mutation_step : float
        Mutation step size applied per generation.

    Returns
    -------
    PolicyMutationSearchReport
        The offline policy-DSL mutation search report.

    Raises
    ------
    ValueError
        If the DSL string or search parameters are invalid.
    """
    config = PolicyMutationSearchConfig(
        generation_count=generation_count,
        population_size=population_size,
        mutation_step=mutation_step,
    )
    rules = parse_policy_dsl(policy_dsl)
    axes = _build_mutation_axes(rules)
    if not axes:
        raise ValueError("policy_dsl must contain at least one mutable component")

    source_policy_dsl = "\n".join(rule.to_dsl() for rule in rules)
    source_policy_hash = _stable_hash({"source_policy_dsl": source_policy_dsl})

    candidates: list[PolicyMutationCandidate] = []
    axis_count = len(axes)
    for generation in range(config.generation_count):
        for local_index in range(config.population_size):
            cursor = (generation * config.population_size + local_index) % axis_count
            axis = axes[cursor]
            base_delta = _deterministic_delta(
                axis_index=cursor,
                generation=generation,
                local_index=local_index,
                axis_count=axis_count,
                generation_count=config.generation_count,
                mutation_step=config.mutation_step,
            )

            original_rule = rules[axis.rule_index]
            blocked_reasons: list[str] = []
            mutated_rule = _mutate_rule(
                rule=original_rule,
                axis=axis,
                delta=base_delta,
                blocked_reasons=blocked_reasons,
            )

            mutated_rules = list(rules)
            mutated_rules[axis.rule_index] = mutated_rule
            candidate_policy = "\n".join(rule.to_dsl() for rule in mutated_rules)
            mutated_value = (
                mutated_rule.conditions[axis.component_index].threshold
                if axis.component == "condition"
                else mutated_rule.action.value
            )

            plan = PolicyMutationPlan(
                rule_name=axis.rule_name,
                component=axis.component,
                component_index=axis.component_index,
                operator=axis.operator,
                original_value=axis.original_value,
                mutated_value=mutated_value,
                mutation_delta=mutated_value - axis.original_value,
            )
            candidate = PolicyMutationCandidate(
                candidate_id=f"g{generation + 1:02d}-c{local_index + 1:03d}",
                generation=generation + 1,
                mutation_index=len(candidates),
                source_rule_name=axis.rule_name,
                source_rule_text=original_rule.to_dsl(),
                mutated_rule_text=mutated_rule.to_dsl(),
                candidate_policy_dsl=candidate_policy,
                mutation_plan=plan,
                blocked_reasons=tuple(blocked_reasons),
                candidate_hash="",
            )
            candidate = replace(
                candidate,
                candidate_hash=_build_candidate_hash(candidate),
            )
            candidates.append(candidate)

    accepted = [candidate for candidate in candidates if candidate.accepted]
    rejected = [candidate for candidate in candidates if not candidate.accepted]

    report = PolicyMutationSearchReport(
        schema_name="policy_dsl_evolution",
        schema_version="0.1.0",
        config=config,
        source_policy_dsl=source_policy_dsl,
        source_policy_hash=source_policy_hash,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        candidates=tuple(candidates),
        execution_disabled=True,
        hot_patch_permitted=False,
        live_merge_permitted=False,
        actuation_permitted=False,
        operator_review_required=True,
        non_actuating=True,
        report_hash="",
    )
    return replace(report, report_hash=_build_report_hash(report))


def _parse_conditions(condition_text: str) -> tuple[PolicyCondition, ...]:
    """Parse the policy conditions from a DSL mapping, else raise."""
    fragments = [
        fragment.strip()
        for fragment in re.split(r"\band\b", condition_text, flags=re.IGNORECASE)
    ]
    if not fragments:
        raise ValueError(f"Malformed condition: {condition_text}")

    conditions: list[PolicyCondition] = []
    for fragment in fragments:
        if not fragment:
            raise ValueError(f"Malformed condition: {condition_text}")

        match = _CONDITION_RE.match(fragment)
        if not match:
            raise ValueError(f"Malformed condition: {fragment}")

        conditions.append(
            PolicyCondition(
                metric=match.group("metric"),
                operator=match.group("operator"),
                threshold=_coerce_finite_real(
                    float(match.group("value")),
                    label="condition threshold",
                ),
            )
        )

    if not conditions:
        raise ValueError(f"Malformed condition: {condition_text}")

    return tuple(conditions)


def _parse_action(action_text: str) -> PolicyAction:
    """Parse a policy action from a DSL mapping, else raise."""
    match = _ACTION_RE.match(action_text)
    if not match:
        raise ValueError(f"Malformed action: {action_text}")

    return PolicyAction(
        target=match.group("target"),
        operator=match.group("operator"),
        value=_coerce_finite_real(
            float(match.group("value")),
            label="action value",
        ),
    )


def _build_mutation_axes(rules: tuple[PolicyRule, ...]) -> list[_MutationAxis]:
    """Return the mutation axes for the policy rules."""
    axes: list[_MutationAxis] = []
    for rule_index, rule in enumerate(rules):
        for condition_index, condition in enumerate(rule.conditions):
            axes.append(
                _MutationAxis(
                    rule_index=rule_index,
                    rule_name=rule.name,
                    component="condition",
                    component_index=condition_index,
                    original_value=condition.threshold,
                    operator=condition.operator,
                )
            )
        axes.append(
            _MutationAxis(
                rule_index=rule_index,
                rule_name=rule.name,
                component="action",
                component_index=0,
                original_value=rule.action.value,
                operator=rule.action.operator,
            )
        )
    return axes


def _mutate_rule(
    *,
    rule: PolicyRule,
    axis: _MutationAxis,
    delta: float,
    blocked_reasons: list[str],
) -> PolicyRule:
    """Return the rule mutated along an axis by a delta."""
    if axis.component == "condition":
        conditions = list(rule.conditions)
        updated_condition = replace(
            conditions[axis.component_index],
            threshold=_coerce_finite_real(
                conditions[axis.component_index].threshold + delta,
                label="mutated condition threshold",
            ),
        )
        conditions[axis.component_index] = updated_condition
        if not 0.0 <= updated_condition.threshold <= 1.0:
            blocked_reasons.append("condition_threshold_outside_0_to_1")
        return PolicyRule(
            name=rule.name, conditions=tuple(conditions), action=rule.action
        )

    if axis.component == "action":
        mutated_value = _coerce_finite_real(
            rule.action.value + delta,
            label="mutated action value",
        )
        if not 0.0 <= mutated_value <= 1.0:
            blocked_reasons.append("action_value_outside_0_to_1")
        return PolicyRule(
            name=rule.name,
            conditions=rule.conditions,
            action=replace(rule.action, value=mutated_value),
        )

    raise ValueError(f"Unsupported mutation component: {axis.component}")


def _deterministic_delta(
    *,
    axis_index: int,
    generation: int,
    local_index: int,
    axis_count: int,
    generation_count: int,
    mutation_step: float,
) -> float:
    """Return the deterministic mutation delta for a step."""
    direction = 1.0 if (generation + local_index) % 2 == 0 else -1.0
    position_scale = 1.0 + (axis_index / max(1, axis_count))
    generation_scale = 1.0 + (generation / max(1, generation_count))
    return direction * mutation_step * position_scale * generation_scale


def _build_candidate_hash(candidate: PolicyMutationCandidate) -> str:
    """Return the canonical hash of a candidate."""
    payload = candidate.to_audit_record()
    payload["candidate_hash"] = ""
    return _stable_hash(payload)


def _build_report_hash(report: PolicyMutationSearchReport) -> str:
    """Return the canonical hash of a report."""
    payload = report.to_audit_record()
    payload["report_hash"] = ""
    return _stable_hash(payload)


def _coerce_finite_real(value: Any, label: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be a finite real number")
    return converted


def _require_positive_int(value: Any, label: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _require_finite_positive(value: float, label: str) -> None:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")


def _format_float(value: float) -> str:
    """Return a stable string rendering of a float."""
    return f"{value:.12g}"


def _stable_hash(payload: Mapping[str, object]) -> str:
    """Return a stable SHA-256 hash of the inputs."""
    normalized = _coerce_json_safe(payload)
    return hashlib.sha256(
        json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _coerce_json_safe(value: Any) -> Any:
    """Return ``value`` as a JSON-safe value, else raise."""
    if isinstance(value, Mapping):
        return {key: _coerce_json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_coerce_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_coerce_json_safe(item) for item in value]
    return value
