# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Petri net PRISM and TLA+ exporters

"""PRISM and TLA+ text exporters for supervisor Petri nets."""

from __future__ import annotations

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet, Transition

from ._shared import (
    PrismExport,
    TLAExport,
    _identifier,
    _tla_module_identifier,
    _unique_identifier,
)


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


def _guard_expr(transition: Transition, metric_names: dict[str, str]) -> str:
    if transition.guard is None:
        return "true"
    guard = transition.guard
    return f"{metric_names[guard.metric]} {guard.op} {guard.threshold:.17g}"


def _tla_guard_expr(transition: Transition, metric_names: dict[str, str]) -> str:
    return _guard_expr(transition, metric_names).replace("==", "=")


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

    Parameters
    ----------
    net : PetriNet
        The Petri net to export.
    initial_marking : Marking
        The initial Petri net marking.
    module_name : str
        Name of the emitted model-checker module.
    max_tokens : int | None
        Maximum token bound per place, or ``None``.

    Returns
    -------
    PrismExport
        The bounded PRISM MDP export of the Petri net.

    Raises
    ------
    PolicyError
        If the net or bounds violate the export policy.
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

    Parameters
    ----------
    net : PetriNet
        The Petri net to export.
    initial_marking : Marking
        The initial Petri net marking.
    module_name : str
        Name of the emitted model-checker module.
    max_tokens : int | None
        Maximum token bound per place, or ``None``.

    Returns
    -------
    TLAExport
        The bounded TLA+ export of the Petri net.

    Raises
    ------
    PolicyError
        If the net or bounds violate the export policy.
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
