# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal supervisor exporters

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet, Transition
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyCondition,
    PolicyRule,
)

__all__ = ["PrismExport", "export_petri_net_prism", "export_policy_rules_prism"]

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


@dataclass(frozen=True)
class PrismExport:
    """PRISM model text plus the identifier mapping used during export."""

    model: str
    place_names: dict[str, str]
    metric_names: dict[str, str]
    transition_names: dict[str, str]
    rule_names: dict[str, str] = field(default_factory=dict)
    action_names: dict[str, str] = field(default_factory=dict)


def _identifier(raw: str, *, prefix: str) -> str:
    cleaned = _IDENT_RE.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _unique_identifier(
    raw: str,
    *,
    prefix: str,
    used: set[str],
) -> str:
    base = _identifier(raw, prefix=prefix)
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _place_mapping(net: PetriNet) -> dict[str, str]:
    used: set[str] = set()
    return {
        name: _unique_identifier(name, prefix="p", used=used)
        for name in sorted(net.place_names)
    }


def _transition_mapping(net: PetriNet) -> dict[str, str]:
    used: set[str] = set()
    return {
        transition.name: _unique_identifier(transition.name, prefix="t", used=used)
        for transition in net.transitions
    }


def _metric_mapping(net: PetriNet) -> dict[str, str]:
    used: set[str] = set()
    metrics = sorted(
        {
            transition.guard.metric
            for transition in net.transitions
            if transition.guard is not None
        }
    )
    return {
        metric: _unique_identifier(metric, prefix="m", used=used) for metric in metrics
    }


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


def _guard_expr(transition: Transition, metric_names: dict[str, str]) -> str:
    if transition.guard is None:
        return "true"
    guard = transition.guard
    return f"{metric_names[guard.metric]} {guard.op} {guard.threshold:.17g}"


def _policy_condition_expr(
    condition: PolicyCondition,
    metric_names: dict[str, str],
) -> str:
    metric = metric_names[_policy_metric_key(condition)]
    return f"{metric} {condition.op} {condition.threshold:.17g}"


def _policy_guard_expr(
    condition: PolicyCondition | CompoundCondition,
    metric_names: dict[str, str],
) -> str:
    if isinstance(condition, CompoundCondition):
        if not condition.conditions:
            raise PolicyError("compound policy condition must not be empty")
        op = "|" if condition.logic.upper() == "OR" else "&"
        parts = [
            _policy_condition_expr(item, metric_names) for item in condition.conditions
        ]
        return "(" + f" {op} ".join(parts) + ")"
    return _policy_condition_expr(condition, metric_names)


def _regime_guard_expr(rule: PolicyRule, regime_names: dict[str, int]) -> str:
    regimes = [regime.upper() for regime in rule.regimes]
    if not regimes:
        raise PolicyError(f"policy rule {rule.name!r} has no regimes")
    parts = [f"regime = {regime_names[regime]}" for regime in regimes]
    return "(" + " | ".join(parts) + ")"


def _input_expr(transition: Transition, place_names: dict[str, str]) -> str:
    if not transition.inputs:
        return "true"
    return " & ".join(
        f"{place_names[arc.place]} >= {arc.weight}" for arc in transition.inputs
    )


def _update_expr(transition: Transition, place_names: dict[str, str]) -> str:
    deltas = dict.fromkeys(place_names, 0)
    for arc in transition.inputs:
        deltas[arc.place] -= arc.weight
    for arc in transition.outputs:
        deltas[arc.place] += arc.weight
    updates = []
    for place, identifier in place_names.items():
        delta = deltas[place]
        if delta == 0:
            updates.append(f"({identifier}'={identifier})")
        elif delta > 0:
            updates.append(f"({identifier}'={identifier}+{delta})")
        else:
            updates.append(f"({identifier}'={identifier}-{abs(delta)})")
    return " & ".join(updates)


def _token_upper_bound(
    net: PetriNet,
    initial_marking: Marking,
    *,
    minimum: int,
) -> int:
    observed = [minimum, *initial_marking.tokens.values()]
    for transition in net.transitions:
        observed.extend(arc.weight for arc in transition.inputs)
        observed.extend(arc.weight for arc in transition.outputs)
    return max(observed)


def _policy_fire_bound(rule: PolicyRule) -> int:
    return max(1, rule.max_fires)


def export_petri_net_prism(
    net: PetriNet,
    initial_marking: Marking,
    *,
    module_name: str = "spo_petri",
    max_tokens: int | None = None,
) -> PrismExport:
    """Serialise a Petri net into a bounded PRISM MDP model.

    Guard metrics become PRISM constants, so safety properties can be checked
    over scenario-specific metric assignments without changing the net model.
    """

    if not net.place_names:
        raise PolicyError("cannot export Petri net without places")
    if max_tokens is not None and max_tokens < 1:
        raise PolicyError("max_tokens must be >= 1")

    place_names = _place_mapping(net)
    transition_names = _transition_mapping(net)
    metric_names = _metric_mapping(net)
    token_bound = max_tokens or _token_upper_bound(net, initial_marking, minimum=1)
    module_identifier = _identifier(module_name, prefix="module")

    lines = [
        "mdp",
        "",
        "// Generated from SCPN PetriNet for PRISM model checking.",
    ]
    if any(raw != mapped for raw, mapped in place_names.items()):
        lines.append("// Place identifiers:")
        lines.extend(f"//   {raw} -> {mapped}" for raw, mapped in place_names.items())
    if metric_names:
        lines.append("// Guard metric constants:")
        lines.extend(f"//   {raw} -> {mapped}" for raw, mapped in metric_names.items())
        lines.extend(f"const double {mapped};" for mapped in metric_names.values())
    lines.extend(["", f"module {module_identifier}"])

    for raw_name, identifier in place_names.items():
        initial = initial_marking[raw_name]
        if initial > token_bound:
            raise PolicyError(
                f"initial marking for {raw_name!r} exceeds max_tokens={token_bound}"
            )
        lines.append(f"  {identifier} : [0..{token_bound}] init {initial};")

    lines.append("")
    for transition in net.transitions:
        guard = _guard_expr(transition, metric_names)
        inputs = _input_expr(transition, place_names)
        command_guard = f"{guard} & {inputs}"
        update = _update_expr(transition, place_names)
        action = transition_names[transition.name]
        lines.append(f"  [{action}] {command_guard} -> {update};")

    lines.extend(["endmodule", ""])
    for raw_name, identifier in place_names.items():
        lines.append(f'label "active_{identifier}" = {identifier} > 0;')
        if raw_name != identifier:
            lines.append(f"// active_{identifier} maps original place {raw_name!r}")

    return PrismExport(
        model="\n".join(lines) + "\n",
        place_names=place_names,
        metric_names=metric_names,
        transition_names=transition_names,
    )


def export_policy_rules_prism(
    rules: list[PolicyRule],
    *,
    module_name: str = "spo_policy",
) -> PrismExport:
    """Serialise policy rules into a bounded PRISM MDP model.

    Metrics and current regime are model inputs represented as PRISM
    constants. Each rule has a bounded fire counter; unlimited rules are
    represented as one-shot reachability counters for model-checking queries.
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
