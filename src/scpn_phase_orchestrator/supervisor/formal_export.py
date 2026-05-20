# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal supervisor exporters

"""Formal-model exporters for Petri nets, policy rules, and STL monitors.

The exporter functions convert already-validated supervisor structures into
PRISM or TLA+ text plus identifier maps, sanitizing names and preserving metric,
transition, rule, action, and STL mappings for auditability. Export routines are
pure text generation; they do not invoke model checkers, write files, or change
the source policy/Petri structures.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet, Transition
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyCondition,
    PolicyRule,
    PolicySTLSpec,
)

__all__ = [
    "FormalCheckerCommand",
    "FormalSafetyProperty",
    "FormalVerificationPackage",
    "PrismExport",
    "TLAExport",
    "build_formal_verification_package",
    "export_petri_net_prism",
    "export_petri_net_tla",
    "export_policy_rules_prism",
    "export_policy_rules_tla",
    "export_stl_specs_prism",
]

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_SIMPLE_STL_RE = re.compile(r"^(always|eventually)\s*\((.*)\)\s*$")
_STL_PREDICATE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|>|<=|<|==)\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class PrismExport:
    """PRISM model text plus the identifier mapping used during export."""

    model: str
    place_names: dict[str, str]
    metric_names: dict[str, str]
    transition_names: dict[str, str]
    rule_names: dict[str, str] = field(default_factory=dict)
    action_names: dict[str, str] = field(default_factory=dict)
    stl_names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TLAExport:
    """TLA+ module text plus the identifier mapping used during export."""

    module: str
    place_names: dict[str, str]
    metric_names: dict[str, str]
    transition_names: dict[str, str]
    rule_names: dict[str, str] = field(default_factory=dict)
    action_names: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FormalSafetyProperty:
    """Named model-checking property bound to one exported artefact."""

    name: str
    artifact_name: str
    checker: str
    expression: str
    description: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        _require_package_identifier(self.name, "property name")
        _require_package_identifier(self.artifact_name, "property artifact_name")
        if self.checker not in {"prism", "tlc"}:
            raise PolicyError("property checker must be 'prism' or 'tlc'")
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise PolicyError("property expression must be a non-empty string")
        if any(ord(char) < 32 for char in self.expression):
            raise PolicyError("property expression must not contain control characters")
        if any(ord(char) < 32 for char in self.description):
            raise PolicyError(
                "property description must not contain control characters"
            )

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe formal property record."""
        return {
            "name": self.name,
            "artifact_name": self.artifact_name,
            "checker": self.checker,
            "expression": self.expression,
            "description": self.description,
            "required": self.required,
        }


@dataclass(frozen=True)
class FormalCheckerCommand:
    """External model-checker command manifest for one property."""

    property_name: str
    checker: str
    artifact_name: str
    command: tuple[str, ...]
    execution_permitted: bool = False

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe external checker command record."""
        return {
            "property_name": self.property_name,
            "checker": self.checker,
            "artifact_name": self.artifact_name,
            "command": list(self.command),
            "execution_permitted": self.execution_permitted,
        }


@dataclass(frozen=True)
class FormalVerificationPackage:
    """Deterministic bundle for external formal-verification workflows."""

    package_name: str
    artifact_hashes: dict[str, str]
    artifact_types: dict[str, str]
    properties: tuple[FormalSafetyProperty, ...]
    checker_commands: tuple[FormalCheckerCommand, ...]
    package_hash: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe package manifest."""
        return {
            "package_name": self.package_name,
            "artifact_hashes": dict(sorted(self.artifact_hashes.items())),
            "artifact_types": dict(sorted(self.artifact_types.items())),
            "properties": [item.to_audit_record() for item in self.properties],
            "checker_commands": [
                command.to_audit_record() for command in self.checker_commands
            ],
            "package_hash": self.package_hash,
        }


def _require_package_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _PACKAGE_NAME_RE.fullmatch(value):
        raise PolicyError(
            f"{field_name} must start with a letter and contain only letters, "
            "digits, underscore, dot, or hyphen"
        )
    return value


def _artifact_text(export: PrismExport | TLAExport) -> str:
    if isinstance(export, PrismExport):
        return export.model
    return export.module


def _artifact_type(export: PrismExport | TLAExport) -> str:
    if isinstance(export, PrismExport):
        return "prism"
    return "tla"


def _checker_command(property_: FormalSafetyProperty) -> FormalCheckerCommand:
    if property_.checker == "prism":
        command = (
            "prism",
            f"{property_.artifact_name}.prism",
            "-pf",
            property_.expression,
        )
    else:
        command = (
            "tlc2.TLC",
            f"{property_.artifact_name}.tla",
            "-config",
            f"{property_.artifact_name}.cfg",
        )
    return FormalCheckerCommand(
        property_name=property_.name,
        checker=property_.checker,
        artifact_name=property_.artifact_name,
        command=command,
    )


def _checker_matches_artifact(
    property_: FormalSafetyProperty,
    artifact_type: str,
) -> bool:
    if property_.checker == "prism":
        return artifact_type == "prism"
    return artifact_type == "tla"


def build_formal_verification_package(
    artifacts: Mapping[str, PrismExport | TLAExport],
    properties: Sequence[FormalSafetyProperty],
    *,
    package_name: str = "spo-formal-verification",
) -> FormalVerificationPackage:
    """Build a deterministic manifest for external model-checker execution.

    The package records exported artefact hashes, property-library entries, and
    exact checker commands. It never writes files or invokes external tools;
    CI or operators can materialise the package and run the recorded commands
    in a controlled environment.
    """

    _require_package_identifier(package_name, "package_name")
    if not artifacts:
        raise PolicyError("formal verification package requires artifacts")
    if not properties:
        raise PolicyError("formal verification package requires properties")

    artifact_hashes: dict[str, str] = {}
    artifact_types: dict[str, str] = {}
    for artifact_name, export in sorted(artifacts.items()):
        _require_package_identifier(artifact_name, "artifact name")
        if not isinstance(export, PrismExport | TLAExport):
            raise PolicyError("formal artifacts must be PrismExport or TLAExport")
        artifact_text = _artifact_text(export)
        if not artifact_text.strip():
            raise PolicyError(f"formal artifact {artifact_name!r} is empty")
        artifact_hashes[artifact_name] = hashlib.sha256(
            artifact_text.encode("utf-8")
        ).hexdigest()
        artifact_types[artifact_name] = _artifact_type(export)

    property_names: set[str] = set()
    commands: list[FormalCheckerCommand] = []
    for property_ in properties:
        if property_.name in property_names:
            raise PolicyError(f"duplicate formal property {property_.name!r}")
        property_names.add(property_.name)
        if property_.artifact_name not in artifact_hashes:
            raise PolicyError(
                f"formal property {property_.name!r} references unknown artifact "
                f"{property_.artifact_name!r}"
            )
        if not _checker_matches_artifact(
            property_,
            artifact_types[property_.artifact_name],
        ):
            raise PolicyError(
                f"formal property {property_.name!r} checker does not match "
                f"artifact {property_.artifact_name!r}"
            )
        commands.append(_checker_command(property_))

    package_seed = {
        "package_name": package_name,
        "artifact_hashes": dict(sorted(artifact_hashes.items())),
        "artifact_types": dict(sorted(artifact_types.items())),
        "properties": [item.to_audit_record() for item in properties],
        "checker_commands": [command.to_audit_record() for command in commands],
    }
    package_hash = hashlib.sha256(
        json.dumps(package_seed, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return FormalVerificationPackage(
        package_name=package_name,
        artifact_hashes=artifact_hashes,
        artifact_types=artifact_types,
        properties=tuple(properties),
        checker_commands=tuple(commands),
        package_hash=package_hash,
    )


def _identifier(raw: str, *, prefix: str) -> str:
    cleaned = _IDENT_RE.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _tla_module_identifier(raw: str) -> str:
    identifier = _identifier(raw, prefix="SpoModule")
    if not identifier[0].isalpha():
        identifier = f"Spo{identifier}"
    return identifier


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


def _guard_expr(transition: Transition, metric_names: dict[str, str]) -> str:
    if transition.guard is None:
        return "true"
    guard = transition.guard
    return f"{metric_names[guard.metric]} {guard.op} {guard.threshold:.17g}"


def _tla_guard_expr(transition: Transition, metric_names: dict[str, str]) -> str:
    return _guard_expr(transition, metric_names).replace("==", "=")


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


def _input_expr(transition: Transition, place_names: dict[str, str]) -> str:
    if not transition.inputs:
        return "true"
    return " & ".join(
        f"{place_names[arc.place]} >= {arc.weight}" for arc in transition.inputs
    )


def _tla_input_expr(transition: Transition, place_names: dict[str, str]) -> str:
    if not transition.inputs:
        return "TRUE"
    return " /\\ ".join(
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


def _tla_next_value_lines(
    transition: Transition,
    place_names: dict[str, str],
) -> list[str]:
    deltas = dict.fromkeys(place_names, 0)
    for arc in transition.inputs:
        deltas[arc.place] -= arc.weight
    for arc in transition.outputs:
        deltas[arc.place] += arc.weight
    lines: list[str] = []
    for place, identifier in place_names.items():
        delta = deltas[place]
        if delta == 0:
            next_value = identifier
        elif delta > 0:
            next_value = f"{identifier} + {delta}"
        else:
            next_value = f"{identifier} - {abs(delta)}"
        lines.append(f"  /\\ {identifier}' = {next_value}")
    return lines


def _tla_unchanged_counter_lines(
    changed: str,
    rule_names: dict[str, str],
) -> list[str]:
    return [
        f"  /\\ {rule_id}_fires' = {rule_id}_fires"
        for rule_id in rule_names.values()
        if rule_id != changed
    ]


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


def _stl_mapping(specs: list[PolicySTLSpec]) -> dict[str, str]:
    used: set[str] = set()
    return {
        spec.name: _unique_identifier(spec.name, prefix="stl", used=used)
        for spec in specs
    }


def _stl_predicates(spec: PolicySTLSpec) -> tuple[str, list[tuple[str, str, float]]]:
    match = _SIMPLE_STL_RE.match(spec.spec.strip())
    if match is None:
        raise PolicyError(f"STL monitor {spec.name!r} uses unsupported export syntax")
    temporal_op = match.group(1)
    predicates: list[tuple[str, str, float]] = []
    for raw_predicate in re.split(r"\s+(?:and|&&)\s+", match.group(2)):
        predicate_match = _STL_PREDICATE_RE.match(raw_predicate)
        if predicate_match is None:
            raise PolicyError(
                f"STL monitor {spec.name!r} uses unsupported predicate syntax"
            )
        signal, op, threshold = predicate_match.groups()
        predicates.append((signal, op, float(threshold)))
    return temporal_op, predicates


def _stl_signal_mapping(
    specs: list[PolicySTLSpec],
) -> tuple[dict[str, str], dict[str, tuple[str, list[tuple[str, str, float]]]]]:
    parsed: dict[str, tuple[str, list[tuple[str, str, float]]]] = {}
    signals: set[str] = set()
    for spec in specs:
        temporal_op, predicates = _stl_predicates(spec)
        parsed[spec.name] = (temporal_op, predicates)
        signals.update(signal for signal, _op, _threshold in predicates)
    used: set[str] = set()
    signal_names = {
        signal: _unique_identifier(signal, prefix="s", used=used)
        for signal in sorted(signals)
    }
    return signal_names, parsed


def _stl_expr(
    predicates: list[tuple[str, str, float]],
    signal_names: dict[str, str],
) -> str:
    parts = [
        f"{signal_names[signal]} {op} {threshold:.17g}"
        for signal, op, threshold in predicates
    ]
    return " & ".join(parts) if parts else "true"


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


def export_petri_net_tla(
    net: PetriNet,
    initial_marking: Marking,
    *,
    module_name: str = "SpoPetri",
    max_tokens: int | None = None,
) -> TLAExport:
    """Serialise a Petri net into a bounded TLA+ transition-system module.

    Guard metrics become TLA+ constants. Places become bounded natural-number
    variables, and each Petri transition becomes a named next-state action that
    preserves all unaffected places explicitly.
    """

    if not net.place_names:
        raise PolicyError("cannot export Petri net without places")
    if max_tokens is not None and max_tokens < 1:
        raise PolicyError("max_tokens must be >= 1")

    place_names = _place_mapping(net)
    transition_names = _transition_mapping(net)
    metric_names = _metric_mapping(net)
    token_bound = max_tokens or _token_upper_bound(net, initial_marking, minimum=1)
    module_identifier = _tla_module_identifier(module_name)
    variables = list(place_names.values())
    variable_tuple = "<<" + ", ".join(variables) + ">>"

    lines = [
        f"---- MODULE {module_identifier} ----",
        "EXTENDS Naturals, TLC",
        "",
        "\\* Generated from SCPN PetriNet for TLA+ model checking.",
    ]
    if any(raw != mapped for raw, mapped in place_names.items()):
        lines.append("\\* Place identifiers:")
        lines.extend(f"\\*   {raw} -> {mapped}" for raw, mapped in place_names.items())
    if metric_names:
        lines.append("\\* Guard metric constants:")
        lines.extend(f"\\*   {raw} -> {mapped}" for raw, mapped in metric_names.items())
        lines.append("CONSTANTS " + ", ".join(metric_names.values()))
    lines.extend(["", "VARIABLES " + ", ".join(variables), "", "Init =="])

    for raw_name, identifier in place_names.items():
        initial = initial_marking[raw_name]
        if initial > token_bound:
            raise PolicyError(
                f"initial marking for {raw_name!r} exceeds max_tokens={token_bound}"
            )
        lines.append(f"  /\\ {identifier} = {initial}")

    lines.extend(["", "TypeOK =="])
    lines.extend(
        f"  /\\ {identifier} \\in 0..{token_bound}" for identifier in variables
    )

    for transition in net.transitions:
        action = transition_names[transition.name]
        lines.extend(["", f"{action} =="])
        guard = _tla_guard_expr(transition, metric_names)
        if guard != "true":
            lines.append(f"  /\\ {guard}")
        inputs = _tla_input_expr(transition, place_names)
        if inputs != "TRUE":
            lines.append(f"  /\\ {inputs}")
        lines.extend(_tla_next_value_lines(transition, place_names))

    next_terms = [transition_names[transition.name] for transition in net.transitions]
    lines.extend(["", "Next =="])
    if next_terms:
        first, *rest = next_terms
        lines.append(f"  \\/ {first}")
        lines.extend(f"  \\/ {term}" for term in rest)
    else:
        lines.append("  /\\ UNCHANGED " + variable_tuple)

    lines.extend(
        [
            "",
            f"Spec == Init /\\ [][Next]_{variable_tuple}",
            "Safety == TypeOK",
            "",
        ]
    )
    for raw_name, identifier in place_names.items():
        lines.append(f"Active_{identifier} == {identifier} > 0")
        if raw_name != identifier:
            lines.append(f"\\* Active_{identifier} maps original place {raw_name!r}")
    lines.append("====")

    return TLAExport(
        module="\n".join(lines) + "\n",
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


def export_policy_rules_tla(
    rules: list[PolicyRule],
    *,
    module_name: str = "SpoPolicy",
) -> TLAExport:
    """Serialise policy rules into a bounded TLA+ transition-system module."""

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


def export_stl_specs_prism(
    specs: list[PolicySTLSpec],
    *,
    module_name: str = "spo_stl",
) -> PrismExport:
    """Serialise policy-declared STL monitors into PRISM label surfaces.

    This export covers the builtin STL subset used by ``STLMonitor``:
    ``always (...)`` and ``eventually (...)`` over numeric predicate
    conjunctions. The model is a single-state abstraction with signal
    constants and per-monitor satisfied/violated labels for property checks.
    """

    if not specs:
        raise PolicyError("cannot export STL monitors without specs")
    for spec in specs:
        if not spec.name:
            raise PolicyError("STL monitor names must not be empty")
        if spec.severity not in {"soft", "hard"}:
            raise PolicyError(f"STL monitor {spec.name!r} has unsupported severity")

    stl_names = _stl_mapping(specs)
    signal_names, parsed = _stl_signal_mapping(specs)
    module_identifier = _identifier(module_name, prefix="module")

    lines = [
        "mdp",
        "",
        "// Generated from SCPN policy STL monitors for PRISM model checking.",
        "// Signal constants represent one sampled trace point or scenario bound.",
    ]
    if signal_names:
        lines.append("// STL signal constants:")
        lines.extend(f"//   {raw} -> {mapped}" for raw, mapped in signal_names.items())
        lines.extend(f"const double {mapped};" for mapped in signal_names.values())
    lines.extend(
        [
            "",
            f"module {module_identifier}",
            "  state : [0..0] init 0;",
            "endmodule",
            "",
        ]
    )

    for spec in specs:
        stl_id = stl_names[spec.name]
        temporal_op, predicates = parsed[spec.name]
        expr = _stl_expr(predicates, signal_names)
        lines.append(
            f"// STL {spec.name!r}: {temporal_op} monitor, severity={spec.severity}"
        )
        lines.append(f'label "stl_{stl_id}_satisfied" = {expr};')
        lines.append(f'label "stl_{stl_id}_violated" = !({expr});')

    return PrismExport(
        model="\n".join(lines) + "\n",
        place_names={},
        metric_names=signal_names,
        transition_names={},
        stl_names=stl_names,
    )
