# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule dry-run diagnostics

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from scpn_phase_orchestrator.supervisor.policy_rules import (
    PolicyEngine,
    PolicyRule,
)
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

__all__ = [
    "PolicyDryRunReport",
    "PolicyDryRunStep",
    "dry_run_policy_rules",
]


@dataclass(frozen=True)
class PolicyDryRunStep:
    """One audit step and the policy rules that fired on it."""

    step: int
    regime: str
    fired_rules: tuple[str, ...]
    actions: tuple[str, ...]


@dataclass(frozen=True)
class PolicyDryRunReport:
    """Summary of replayed policy behaviour over an audit log."""

    steps: int
    rules: tuple[str, ...]
    fire_counts: dict[str, int]
    action_counts: dict[str, int]
    unreachable_rules: tuple[str, ...]
    overlapping_steps: tuple[int, ...]
    action_collision_steps: tuple[int, ...]
    step_reports: tuple[PolicyDryRunStep, ...]


def _regime_from_entry(entry: dict[str, Any]) -> Regime:
    raw = str(entry.get("regime", "nominal")).lower()
    for regime in Regime:
        if regime.value == raw or regime.name.lower() == raw:
            return regime
    return Regime.NOMINAL


def _state_from_entry(entry: dict[str, Any]) -> UPDEState:
    raw_layers = entry.get("layers", [])
    layers = []
    for raw_layer in raw_layers:
        layer = raw_layer if isinstance(raw_layer, dict) else {}
        layers.append(
            LayerState(
                R=float(layer.get("R", 0.0)),
                psi=float(layer.get("psi", 0.0)),
                mean_amplitude=float(layer.get("mean_amplitude", 0.0)),
                amplitude_spread=float(layer.get("amplitude_spread", 0.0)),
            )
        )
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.zeros((len(layers), len(layers))),
        stability_proxy=float(entry.get("stability", 0.0)),
        regime_id=str(entry.get("regime", "nominal")),
        mean_amplitude=float(entry.get("mean_amplitude", 0.0)),
        pac_max=float(entry.get("pac_max", 0.0)),
        subcritical_fraction=float(entry.get("subcritical_fraction", 0.0)),
        boundary_violation_count=int(entry.get("boundary_violation_count", 0)),
        imprint_mean=float(entry.get("imprint_mean", 0.0)),
    )


def _rule_name_from_justification(justification: str) -> str:
    prefix = "policy rule: "
    if justification.startswith(prefix):
        return justification[len(prefix) :]
    return justification or "<unknown>"


def _step_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in entries if "step" in entry and "layers" in entry]


def dry_run_policy_rules(
    rules: list[PolicyRule],
    entries: list[dict[str, Any]],
    *,
    good_layers: list[int],
    bad_layers: list[int],
) -> PolicyDryRunReport:
    """Replay policy rules over audit steps without applying actuation."""

    engine = PolicyEngine(rules)
    rule_names = tuple(rule.name for rule in rules)
    fire_counts = dict.fromkeys(rule_names, 0)
    action_counts: dict[str, int] = {}
    step_reports: list[PolicyDryRunStep] = []
    overlapping_steps: list[int] = []
    action_collision_steps: list[int] = []

    for entry in _step_entries(entries):
        step_no = int(entry.get("step", 0))
        regime = _regime_from_entry(entry)
        state = _state_from_entry(entry)
        actions = engine.evaluate(regime, state, good_layers, bad_layers)
        fired_rules = tuple(
            _rule_name_from_justification(action.justification) for action in actions
        )
        distinct_rules = tuple(dict.fromkeys(fired_rules))
        if len(distinct_rules) > 1:
            overlapping_steps.append(step_no)

        action_keys = tuple(f"{action.knob}:{action.scope}" for action in actions)
        if len(set(action_keys)) < len(action_keys):
            action_collision_steps.append(step_no)
        for rule_name in fired_rules:
            fire_counts[rule_name] = fire_counts.get(rule_name, 0) + 1
        for action_key in action_keys:
            action_counts[action_key] = action_counts.get(action_key, 0) + 1

        step_reports.append(
            PolicyDryRunStep(
                step=step_no,
                regime=regime.value,
                fired_rules=distinct_rules,
                actions=action_keys,
            )
        )
        engine.advance_clock(1.0)

    unreachable = tuple(name for name in rule_names if fire_counts.get(name, 0) == 0)
    return PolicyDryRunReport(
        steps=len(step_reports),
        rules=rule_names,
        fire_counts=fire_counts,
        action_counts=action_counts,
        unreachable_rules=unreachable,
        overlapping_steps=tuple(overlapping_steps),
        action_collision_steps=tuple(action_collision_steps),
        step_reports=tuple(step_reports),
    )
