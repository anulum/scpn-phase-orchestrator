# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Formal verification exporters

"""Formal verification exporters for guard-gated Petri nets."""

from __future__ import annotations

import re

from scpn_phase_orchestrator.exceptions import PolicyError
from scpn_phase_orchestrator.supervisor.petri_net import Marking, PetriNet, Transition

__all__ = ["export_petri_net_to_prism"]

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")
_PRISM_OPS = {"==": "=", ">": ">", ">=": ">=", "<": "<", "<=": "<="}


def export_petri_net_to_prism(
    net: PetriNet,
    initial: Marking,
    *,
    module_name: str = "supervisor",
    max_tokens: int | None = None,
    include_idle: bool = True,
) -> str:
    """Export a guard-gated Petri net as a finite PRISM MDP model.

    The exporter preserves the runtime engine's first-enabled transition
    priority by blocking each command when an earlier transition is enabled.
    Guard metrics become PRISM constants so verification jobs can bind them
    explicitly for a scenario.
    """
    if max_tokens is not None and max_tokens < 1:
        raise PolicyError("max_tokens must be >= 1")

    place_names = sorted(net.place_names)
    place_ids = _identifier_map(place_names, prefix="p")
    transition_ids = _identifier_map(
        [transition.name for transition in net.transitions], prefix="t"
    )
    metric_ids = _identifier_map(_guard_metric_names(net.transitions), prefix="m")
    token_bound = max_tokens or _default_token_bound(net, initial)
    module_id = _safe_identifier(module_name, "module")

    lines = ["mdp", ""]
    for _metric, metric_id in metric_ids.items():
        lines.append(f"const double {metric_id};")
    if metric_ids:
        lines.append("")
    lines.append(f"module {module_id}")
    for place in place_names:
        init_count = initial[place]
        if init_count > token_bound:
            raise PolicyError(
                f"initial marking for {place!r} exceeds max_tokens={token_bound}"
            )
        lines.append(f"  {place_ids[place]} : [0..{token_bound}] init {init_count};")
    lines.append("")

    enabled_formulas: list[str] = []
    for transition in net.transitions:
        transition_id = transition_ids[transition.name]
        expr = _enabled_expr(transition, place_ids, metric_ids, token_bound)
        enabled_name = f"enabled_{transition_id}"
        enabled_formulas.append(enabled_name)
        lines.append(f"  formula {enabled_name} = {expr};")
    lines.append("")

    earlier: list[str] = []
    for transition in net.transitions:
        transition_id = transition_ids[transition.name]
        enabled_name = f"enabled_{transition_id}"
        priority_guard = enabled_name
        if earlier:
            priority_guard = f"{priority_guard} & !({' | '.join(earlier)})"
        update = _update_expr(transition, place_ids)
        lines.append(f"  [{transition_id}] {priority_guard} -> {update};")
        earlier.append(enabled_name)
    if include_idle:
        if enabled_formulas:
            lines.append(f"  [idle] !({' | '.join(enabled_formulas)}) -> true;")
        else:
            lines.append("  [idle] true -> true;")
    lines.append("endmodule")
    lines.append("")
    return "\n".join(lines)


def _guard_metric_names(transitions: list[Transition]) -> list[str]:
    names = {
        transition.guard.metric
        for transition in transitions
        if transition.guard is not None
    }
    return sorted(names)


def _identifier_map(names: list[str], *, prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        candidate = _safe_identifier(name, prefix)
        unique = candidate
        counter = 2
        while unique in used:
            unique = f"{candidate}_{counter}"
            counter += 1
        used.add(unique)
        result[name] = unique
    return result


def _safe_identifier(name: str, prefix: str) -> str:
    cleaned = _IDENT_RE.sub("_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = prefix
    if cleaned[0].isdigit():
        cleaned = f"{prefix}_{cleaned}"
    return cleaned


def _default_token_bound(net: PetriNet, initial: Marking) -> int:
    initial_total = sum(initial[place] for place in net.place_names)
    max_output = max(
        (
            sum(arc.weight for arc in transition.outputs)
            for transition in net.transitions
        ),
        default=0,
    )
    return max(1, initial_total, max_output)


def _enabled_expr(
    transition: Transition,
    place_ids: dict[str, str],
    metric_ids: dict[str, str],
    max_tokens: int,
) -> str:
    parts = [f"{place_ids[arc.place]} >= {arc.weight}" for arc in transition.inputs]
    for place, delta in _transition_deltas(transition).items():
        if delta > 0:
            parts.append(f"{place_ids[place]} <= {max_tokens - delta}")
    if transition.guard is not None:
        op = _PRISM_OPS.get(transition.guard.op)
        if op is None:
            raise PolicyError(f"unsupported guard operator {transition.guard.op!r}")
        metric_id = metric_ids[transition.guard.metric]
        parts.append(f"{metric_id} {op} {transition.guard.threshold:g}")
    return " & ".join(parts) if parts else "true"


def _transition_deltas(transition: Transition) -> dict[str, int]:
    deltas: dict[str, int] = {}
    for arc in transition.inputs:
        deltas[arc.place] = deltas.get(arc.place, 0) - arc.weight
    for arc in transition.outputs:
        deltas[arc.place] = deltas.get(arc.place, 0) + arc.weight
    return deltas


def _update_expr(transition: Transition, place_ids: dict[str, str]) -> str:
    updates = []
    for place, delta in sorted(_transition_deltas(transition).items()):
        place_id = place_ids[place]
        if delta < 0:
            updates.append(f"({place_id}'={place_id}{delta})")
        elif delta > 0:
            updates.append(f"({place_id}'={place_id}+{delta})")
    return " & ".join(updates) if updates else "true"
