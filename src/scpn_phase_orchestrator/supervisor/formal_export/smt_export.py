# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule SMT-LIB exporter

"""SMT-LIB text exporter for bounded supervisor policy rule feasibility."""

from __future__ import annotations

from decimal import Decimal

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyCondition,
    PolicyRule,
)

from ._shared import _identifier
from .policy_export import (
    _action_key,
    _action_mapping,
    _compound_logic,
    _policy_fire_bound,
    _policy_metric_key,
    _policy_metric_mapping,
    _regime_mapping,
    _rule_mapping,
    _validate_policy_rules_for_export,
)
from .verification_package import FormalTextArtifact


def _smt_real_literal(value: float) -> str:
    """Return ``value`` as an SMT-LIB real literal without exponent notation."""
    magnitude = f"{abs(value):.17g}"
    if "e" in magnitude.lower():
        magnitude = format(Decimal(magnitude), "f").rstrip("0").rstrip(".")
    if value < 0:
        return f"(- {magnitude})"
    return magnitude


def _smt_policy_condition_expr(
    condition: PolicyCondition,
    metric_names: dict[str, str],
) -> str:
    """Return the SMT-LIB expression for one policy condition."""
    metric = metric_names[_policy_metric_key(condition)]
    operator = "=" if condition.op == "==" else condition.op
    return f"({operator} {metric} {_smt_real_literal(condition.threshold)})"


def _smt_policy_guard_expr(
    condition: PolicyCondition | CompoundCondition,
    metric_names: dict[str, str],
) -> str:
    """Return the SMT-LIB guard expression for a policy rule condition."""
    if isinstance(condition, CompoundCondition):
        if not condition.conditions:
            raise PolicyError("compound policy condition must not be empty")
        operator = "or" if _compound_logic(condition) == "OR" else "and"
        parts = [
            _smt_policy_condition_expr(item, metric_names)
            for item in condition.conditions
        ]
        return f"({operator} {' '.join(parts)})"
    return _smt_policy_condition_expr(condition, metric_names)


def _smt_regime_guard_expr(rule: PolicyRule, regime_names: dict[str, int]) -> str:
    """Return the SMT-LIB guard expression for a policy rule's regimes."""
    regimes = [regime.upper() for regime in rule.regimes]
    parts = [f"(= regime {regime_names[regime]})" for regime in regimes]
    if len(parts) == 1:
        return parts[0]
    return f"(or {' '.join(parts)})"


def export_policy_rules_smt(
    rules: list[PolicyRule],
    *,
    module_name: str = "spo_policy",
) -> FormalTextArtifact:
    """Serialise policy rules into a bounded SMT-LIB feasibility model.

    The export declares the active regime, metric inputs, bounded rule-fire
    counters, rule firing predicates, and action emission predicates. The final
    assertion asks an SMT solver whether at least one policy rule can fire under
    the declared constraints. The function only generates deterministic text; it
    does not invoke Z3 or any other solver.

    Parameters
    ----------
    rules : list[PolicyRule]
        The policy rules to export or validate.
    module_name : str
        Name recorded in the emitted SMT-LIB comments.

    Returns
    -------
    FormalTextArtifact
        An SMT-LIB v2 artifact suitable for package hashing and Z3 execution.

    Raises
    ------
    PolicyError
        If the rules violate the shared formal-export policy.
    """
    _validate_policy_rules_for_export(rules)

    metric_names = _policy_metric_mapping(rules)
    rule_names = _rule_mapping(rules)
    action_names = _action_mapping(rules)
    regime_names = _regime_mapping(rules)
    module_identifier = _identifier(module_name, prefix="module")

    lines = [
        "(set-logic QF_LRA)",
        "; Generated from SCPN PolicyEngine rules for SMT feasibility checking.",
        f"; Module: {module_identifier}",
        "; Regime constants:",
    ]
    lines.extend(f";   {name} -> {value}" for name, value in regime_names.items())
    lines.extend(["", "(declare-const regime Real)"])
    lines.extend(
        f"(declare-const {metric_id} Real)" for metric_id in metric_names.values()
    )

    for rule in rules:
        rule_id = rule_names[rule.name]
        lines.append(f"(declare-const {rule_id}_fire_count Real)")

    valid_regimes = " ".join(
        f"(= regime {value})" for value in sorted(regime_names.values())
    )
    lines.extend(["", f"(assert (or {valid_regimes}))"])

    for rule in rules:
        rule_id = rule_names[rule.name]
        bound = _policy_fire_bound(rule)
        lines.append(f"(assert (>= {rule_id}_fire_count 0))")
        lines.append(f"(assert (<= {rule_id}_fire_count {bound}))")

    lines.append("")
    if metric_names:
        lines.append("; Policy metric mapping:")
        lines.extend(f";   {raw} -> {mapped}" for raw, mapped in metric_names.items())
        lines.append("")

    firing_terms: list[str] = []
    for rule in rules:
        rule_id = rule_names[rule.name]
        firing_terms.append(f"fires_{rule_id}")
        lines.append(f"; Rule {rule.name!r} -> {rule_id}")
        lines.append(
            f"(define-fun fires_{rule_id} () Bool "
            f"(and {_smt_regime_guard_expr(rule, regime_names)} "
            f"{_smt_policy_guard_expr(rule.condition, metric_names)} "
            f"(< {rule_id}_fire_count {_policy_fire_bound(rule)})))"
        )
        for action_index, action in enumerate(rule.actions):
            action_id = action_names[_action_key(rule, action_index)]
            lines.append(
                f"(define-fun emits_{action_id} () Bool fires_{rule_id})"
            )
            lines.append(
                f";   {action_id}: knob={action.knob!r}, "
                f"scope={action.scope!r}, value={action.value:.17g}, "
                f"ttl_s={action.ttl_s:.17g}"
            )
        lines.append("")

    if len(firing_terms) == 1:
        lines.append(f"(assert {firing_terms[0]})")
    else:
        lines.append(f"(assert (or {' '.join(firing_terms)}))")
    lines.append("(check-sat)")

    return FormalTextArtifact(
        artifact_type="smt2",
        text="\n".join(lines) + "\n",
    )
