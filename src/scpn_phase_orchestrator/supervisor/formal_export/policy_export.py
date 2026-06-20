# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule PRISM and TLA+ exporters

"""PRISM and TLA+ text exporters for supervisor policy rules."""

from __future__ import annotations

from math import isfinite

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyCondition,
    PolicyRule,
)

from ._shared import (
    PrismExport,
    TLAExport,
    _identifier,
    _tla_module_identifier,
    _unique_identifier,
)


def _policy_metric_key(condition: PolicyCondition) -> str:
    if condition.layer is None:
        return condition.metric
    return f"{condition.metric}.{condition.layer}"


def _policy_conditions(
    condition: PolicyCondition | CompoundCondition,
) -> list[PolicyCondition]:
    if isinstance(condition, CompoundCondition):
        return list(condition.conditions)
    return [condition]


def _compound_logic(condition: CompoundCondition) -> str:
    if not isinstance(condition.logic, str):
        raise PolicyError("compound policy condition logic must be AND or OR")
    logic = condition.logic.upper()
    if logic not in {"AND", "OR"}:
        raise PolicyError("compound policy condition logic must be AND or OR")
    return logic


def _policy_metric_mapping(rules: list[PolicyRule]) -> dict[str, str]:
    used: set[str] = set()
    metrics = sorted(
        {
            _policy_metric_key(condition)
            for rule in rules
            for condition in _policy_conditions(rule.condition)
        }
    )
    return {
        metric: _unique_identifier(metric, prefix="m", used=used) for metric in metrics
    }


def _rule_mapping(rules: list[PolicyRule]) -> dict[str, str]:
    used: set[str] = set()
    return {
        rule.name: _unique_identifier(rule.name, prefix="rule", used=used)
        for rule in rules
    }


def _action_key(rule: PolicyRule, action_index: int) -> str:
    action = rule.actions[action_index]
    return f"{rule.name}.{action.knob}.{action.scope}.{action_index}"


def _action_mapping(rules: list[PolicyRule]) -> dict[str, str]:
    used: set[str] = set()
    return {
        _action_key(rule, i): _unique_identifier(
            f"{rule.name}_{action.knob}_{action.scope}_{i}",
            prefix="action",
            used=used,
        )
        for rule in rules
        for i, action in enumerate(rule.actions)
    }


def _regime_mapping(rules: list[PolicyRule]) -> dict[str, int]:
    regimes = sorted({regime.upper() for rule in rules for regime in rule.regimes})
    return {regime: idx for idx, regime in enumerate(regimes)}


def _policy_condition_expr(
    condition: PolicyCondition,
    metric_names: dict[str, str],
) -> str:
    metric = metric_names[_policy_metric_key(condition)]
    return f"{metric} {condition.op} {condition.threshold:.17g}"


def _tla_policy_condition_expr(
    condition: PolicyCondition,
    metric_names: dict[str, str],
) -> str:
    return _policy_condition_expr(condition, metric_names).replace("==", "=")


def _policy_guard_expr(
    condition: PolicyCondition | CompoundCondition,
    metric_names: dict[str, str],
) -> str:
    if isinstance(condition, CompoundCondition):
        if not condition.conditions:
            raise PolicyError("compound policy condition must not be empty")
        op = "|" if _compound_logic(condition) == "OR" else "&"
        parts = [
            _policy_condition_expr(item, metric_names) for item in condition.conditions
        ]
        return "(" + f" {op} ".join(parts) + ")"
    return _policy_condition_expr(condition, metric_names)


def _tla_policy_guard_expr(
    condition: PolicyCondition | CompoundCondition,
    metric_names: dict[str, str],
) -> str:
    if isinstance(condition, CompoundCondition):
        if not condition.conditions:
            raise PolicyError("compound policy condition must not be empty")
        op = "\\/" if _compound_logic(condition) == "OR" else "/\\"
        parts = [
            _tla_policy_condition_expr(item, metric_names)
            for item in condition.conditions
        ]
        return "(" + f" {op} ".join(parts) + ")"
    return _tla_policy_condition_expr(condition, metric_names)


def _regime_guard_expr(rule: PolicyRule, regime_names: dict[str, int]) -> str:
    regimes = [regime.upper() for regime in rule.regimes]
    if not regimes:
        raise PolicyError(f"policy rule {rule.name!r} has no regimes")
    parts = [f"regime = {regime_names[regime]}" for regime in regimes]
    return "(" + " | ".join(parts) + ")"


def _tla_regime_guard_expr(rule: PolicyRule, regime_names: dict[str, int]) -> str:
    regimes = [regime.upper() for regime in rule.regimes]
    if not regimes:
        raise PolicyError(f"policy rule {rule.name!r} has no regimes")
    parts = [f"regime = {regime_names[regime]}" for regime in regimes]
    return "(" + " \\/ ".join(parts) + ")"


def _tla_unchanged_counter_lines(
    changed: str,
    rule_names: dict[str, str],
) -> list[str]:
    return [
        f"  /\\ {rule_id}_fires' = {rule_id}_fires"
        for rule_id in rule_names.values()
        if rule_id != changed
    ]


def _policy_fire_bound(rule: PolicyRule) -> int:
    return max(1, rule.max_fires)


def export_policy_rules_prism(
    rules: list[PolicyRule],
    *,
    module_name: str = "spo_policy",
) -> PrismExport:
    """Serialise policy rules into a bounded PRISM MDP model.

    Metrics and current regime are model inputs represented as PRISM
    constants. Each rule has a bounded fire counter; unlimited rules are
    represented as one-shot reachability counters for model-checking queries.

    Parameters
    ----------
    rules : list[PolicyRule]
        The policy rules to export or validate.
    module_name : str
        Name of the emitted model-checker module.

    Returns
    -------
    PrismExport
        The bounded PRISM MDP export of the policy rules.

    Raises
    ------
    PolicyError
        If the rules violate the export policy.
    """
    if not rules:
        raise PolicyError("cannot export policy rules without rules")

    for rule in rules:
        if not rule.name:
            raise PolicyError("policy rule names must not be empty")
        if not rule.regimes:
            raise PolicyError(f"policy rule {rule.name!r} has no regimes")
        if not rule.actions:
            raise PolicyError(f"policy rule {rule.name!r} has no actions")
        for condition in _policy_conditions(rule.condition):
            if not condition.metric:
                raise PolicyError(f"policy rule {rule.name!r} has an empty metric")
            if not isfinite(condition.threshold):
                raise PolicyError(f"policy rule {rule.name!r} has non-finite threshold")
            if condition.op not in {">", ">=", "<", "<=", "=="}:
                raise PolicyError(
                    f"policy rule {rule.name!r} has unsupported operator "
                    f"{condition.op!r}"
                )

    metric_names = _policy_metric_mapping(rules)
    rule_names = _rule_mapping(rules)
    action_names = _action_mapping(rules)
    regime_names = _regime_mapping(rules)
    module_identifier = _identifier(module_name, prefix="module")

    lines = [
        "mdp",
        "",
        "// Generated from SCPN PolicyEngine rules for PRISM model checking.",
        "// Regime constants:",
    ]
    lines.extend(f"//   {name} -> {value}" for name, value in regime_names.items())
    lines.append(f"const int regime; // 0..{max(regime_names.values())}")
    if metric_names:
        lines.append("// Policy metric constants:")
        lines.extend(f"//   {raw} -> {mapped}" for raw, mapped in metric_names.items())
        lines.extend(f"const double {mapped};" for mapped in metric_names.values())
    lines.extend(["", f"module {module_identifier}"])

    for rule in rules:
        rule_id = rule_names[rule.name]
        lines.append(f"  {rule_id}_fires : [0..{_policy_fire_bound(rule)}] init 0;")

    lines.append("")
    for rule in rules:
        rule_id = rule_names[rule.name]
        guard = " & ".join(
            [
                _regime_guard_expr(rule, regime_names),
                _policy_guard_expr(rule.condition, metric_names),
                f"{rule_id}_fires < {_policy_fire_bound(rule)}",
            ]
        )
        lines.append(f"  [{rule_id}] {guard} -> ({rule_id}_fires'={rule_id}_fires+1);")

    lines.extend(["endmodule", ""])
    for rule in rules:
        rule_id = rule_names[rule.name]
        lines.append(f'label "fires_{rule_id}" = {rule_id}_fires > 0;')
        for i, action in enumerate(rule.actions):
            action_id = action_names[_action_key(rule, i)]
            lines.append(f'label "emits_{action_id}" = {rule_id}_fires > 0;')
            lines.append(
                f"//   {action_id}: knob={action.knob!r}, "
                f"scope={action.scope!r}, value={action.value:.17g}, "
                f"ttl_s={action.ttl_s:.17g}"
            )

    return PrismExport(
        model="\n".join(lines) + "\n",
        place_names={},
        metric_names=metric_names,
        transition_names={},
        rule_names=rule_names,
        action_names=action_names,
    )


def export_policy_rules_tla(
    rules: list[PolicyRule],
    *,
    module_name: str = "SpoPolicy",
) -> TLAExport:
    """Serialise policy rules into a bounded TLA+ transition-system module.

    Parameters
    ----------
    rules : list[PolicyRule]
        The policy rules to export or validate.
    module_name : str
        Name of the emitted model-checker module.

    Returns
    -------
    TLAExport
        The bounded TLA+ export of the policy rules.

    Raises
    ------
    PolicyError
        If the rules violate the export policy.
    """
    if not rules:
        raise PolicyError("cannot export policy rules without rules")

    for rule in rules:
        if not rule.name:
            raise PolicyError("policy rule names must not be empty")
        if not rule.regimes:
            raise PolicyError(f"policy rule {rule.name!r} has no regimes")
        if not rule.actions:
            raise PolicyError(f"policy rule {rule.name!r} has no actions")
        for condition in _policy_conditions(rule.condition):
            if not condition.metric:
                raise PolicyError(f"policy rule {rule.name!r} has an empty metric")
            if not isfinite(condition.threshold):
                raise PolicyError(f"policy rule {rule.name!r} has non-finite threshold")
            if condition.op not in {">", ">=", "<", "<=", "=="}:
                raise PolicyError(
                    f"policy rule {rule.name!r} has unsupported operator "
                    f"{condition.op!r}"
                )

    metric_names = _policy_metric_mapping(rules)
    rule_names = _rule_mapping(rules)
    action_names = _action_mapping(rules)
    regime_names = _regime_mapping(rules)
    module_identifier = _tla_module_identifier(module_name)
    counters = [f"{rule_id}_fires" for rule_id in rule_names.values()]
    counter_tuple = "<<" + ", ".join(counters) + ">>"

    lines = [
        f"---- MODULE {module_identifier} ----",
        "EXTENDS Naturals, TLC",
        "",
        "\\* Generated from SCPN PolicyEngine rules for TLA+ model checking.",
        "\\* Regime constants:",
    ]
    lines.extend(f"\\*   {name} -> {value}" for name, value in regime_names.items())
    constants = ["regime", *metric_names.values()]
    lines.append("CONSTANTS " + ", ".join(constants))
    if metric_names:
        lines.append("\\* Policy metric constants:")
        lines.extend(f"\\*   {raw} -> {mapped}" for raw, mapped in metric_names.items())
    lines.extend(["", "VARIABLES " + ", ".join(counters), "", "Init =="])
    lines.extend(f"  /\\ {counter} = 0" for counter in counters)
    lines.extend(["", "TypeOK =="])
    for rule in rules:
        rule_id = rule_names[rule.name]
        lines.append(f"  /\\ {rule_id}_fires \\in 0..{_policy_fire_bound(rule)}")

    for rule in rules:
        rule_id = rule_names[rule.name]
        lines.extend(["", f"{rule_id} =="])
        lines.append(f"  /\\ {_tla_regime_guard_expr(rule, regime_names)}")
        lines.append(f"  /\\ {_tla_policy_guard_expr(rule.condition, metric_names)}")
        lines.append(f"  /\\ {rule_id}_fires < {_policy_fire_bound(rule)}")
        lines.append(f"  /\\ {rule_id}_fires' = {rule_id}_fires + 1")
        lines.extend(_tla_unchanged_counter_lines(rule_id, rule_names))

    next_terms = list(rule_names.values())
    lines.extend(["", "Next =="])
    first, *rest = next_terms
    lines.append(f"  \\/ {first}")
    lines.extend(f"  \\/ {term}" for term in rest)

    lines.extend(
        [
            "",
            f"Spec == Init /\\ [][Next]_{counter_tuple}",
            "Safety == TypeOK",
            "",
        ]
    )
    for rule in rules:
        rule_id = rule_names[rule.name]
        lines.append(f"Fires_{rule_id} == {rule_id}_fires > 0")
        for i, action in enumerate(rule.actions):
            action_id = action_names[_action_key(rule, i)]
            lines.append(f"Emits_{action_id} == {rule_id}_fires > 0")
            lines.append(
                f"\\*   {action_id}: knob={action.knob!r}, "
                f"scope={action.scope!r}, value={action.value:.17g}, "
                f"ttl_s={action.ttl_s:.17g}"
            )
    lines.append("====")

    return TLAExport(
        module="\n".join(lines) + "\n",
        place_names={},
        metric_names=metric_names,
        transition_names={},
        rule_names=rule_names,
        action_names=action_names,
    )
