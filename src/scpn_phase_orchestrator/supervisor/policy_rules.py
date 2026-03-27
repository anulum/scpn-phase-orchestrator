# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule definitions

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = [
    "PolicyCondition",
    "CompoundCondition",
    "PolicyAction",
    "PolicyRule",
    "PolicyEngine",
    "load_policy_rules",
]

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


@dataclass(frozen=True)
class PolicyCondition:
    """Known metrics: R, R_good, R_bad, stability_proxy, pac_max,
    mean_amplitude, subcritical_fraction, amplitude_spread (per-layer),
    mean_amplitude_layer (per-layer)."""

    metric: str
    layer: int | None
    op: str
    threshold: float


@dataclass(frozen=True)
class CompoundCondition:
    """AND/OR combinator over multiple PolicyConditions."""

    conditions: list[PolicyCondition]
    logic: str = "AND"  # "AND" or "OR"


@dataclass(frozen=True)
class PolicyAction:
    """Action emitted by a policy rule: knob, scope, target value, and TTL."""

    knob: str
    scope: str
    value: float
    ttl_s: float


@dataclass(frozen=True)
class PolicyRule:
    """Named rule: fires actions when regime and condition match."""

    name: str
    regimes: list[str]
    condition: PolicyCondition | CompoundCondition
    actions: list[PolicyAction]
    cooldown_s: float = 0.0
    max_fires: int = 0  # 0 = unlimited


class PolicyEngine:
    """Evaluate domainpack policy rules against current state."""

    def __init__(self, rules: list[PolicyRule]) -> None:
        self._rules = rules
        self._fire_counts: dict[str, int] = {}
        self._last_fire_t: dict[str, float] = {}
        self._clock: float = 0.0

    def advance_clock(self, dt: float) -> None:
        """Advance the internal clock used for cooldown tracking."""
        self._clock += dt

    def evaluate(
        self,
        regime: Regime,
        upde_state: UPDEState,
        good_layers: list[int],
        bad_layers: list[int],
    ) -> list[ControlAction]:
        """Evaluate all rules against current state and return triggered actions."""
        actions: list[ControlAction] = []
        for rule in self._rules:
            if regime.value.upper() not in rule.regimes:
                continue
            if not self._check_condition(
                rule.condition, upde_state, good_layers, bad_layers
            ):
                continue
            if rule.cooldown_s > 0:
                last = self._last_fire_t.get(rule.name, -rule.cooldown_s - 1)
                if self._clock - last < rule.cooldown_s:
                    continue
            fires = self._fire_counts.get(rule.name, 0)
            if rule.max_fires > 0 and fires >= rule.max_fires:
                continue
            self._fire_counts[rule.name] = self._fire_counts.get(rule.name, 0) + 1
            self._last_fire_t[rule.name] = self._clock
            for pa in rule.actions:
                actions.append(
                    ControlAction(
                        knob=pa.knob,
                        scope=pa.scope,
                        value=pa.value,
                        ttl_s=pa.ttl_s,
                        justification=f"policy rule: {rule.name}",
                    )
                )
        return actions

    def _check_condition(
        self,
        cond: PolicyCondition | CompoundCondition,
        state: UPDEState,
        good_layers: list[int],
        bad_layers: list[int],
    ) -> bool:
        if isinstance(cond, CompoundCondition):
            results = [
                self._eval_single(c, state, good_layers, bad_layers)
                for c in cond.conditions
            ]
            if cond.logic == "OR":
                return any(results)
            return all(results)
        return self._eval_single(cond, state, good_layers, bad_layers)

    @staticmethod
    def _eval_single(
        cond: PolicyCondition,
        state: UPDEState,
        good_layers: list[int],
        bad_layers: list[int],
    ) -> bool:
        val = _extract_metric(cond, state, good_layers, bad_layers)
        if val is None:
            return False
        cmp = _OPS.get(cond.op)
        if cmp is None:
            return False
        return bool(cmp(val, cond.threshold))

    # Keep old interface for callers that relied on _extract_metric
    _extract_metric = staticmethod(
        lambda cond, state, good, bad: _extract_metric(cond, state, good, bad)
    )


def _extract_metric(
    cond: PolicyCondition,
    state: UPDEState,
    good_layers: list[int],
    bad_layers: list[int],
) -> float | None:
    if cond.metric == "stability_proxy":
        return state.stability_proxy
    if cond.metric == "R" and cond.layer is not None:
        if cond.layer < len(state.layers):
            return state.layers[cond.layer].R
        return None
    if cond.metric == "R_good":
        idx = cond.layer
        if idx is not None and idx < len(good_layers):
            layer_idx = good_layers[idx]
            if layer_idx < len(state.layers):
                return state.layers[layer_idx].R
        return None
    if cond.metric == "R_bad":
        idx = cond.layer
        if idx is not None and idx < len(bad_layers):
            layer_idx = bad_layers[idx]
            if layer_idx < len(state.layers):
                return state.layers[layer_idx].R
        return None
    if cond.metric == "pac_max":
        return state.pac_max
    if cond.metric == "mean_amplitude":
        return state.mean_amplitude
    if cond.metric == "subcritical_fraction":
        return state.subcritical_fraction
    if cond.metric == "amplitude_spread" and cond.layer is not None:
        if cond.layer < len(state.layers):
            return state.layers[cond.layer].amplitude_spread
        return None
    if cond.metric == "mean_amplitude_layer" and cond.layer is not None:
        if cond.layer < len(state.layers):
            return state.layers[cond.layer].mean_amplitude
        return None
    if cond.metric == "boundary_violation_count":
        return float(state.boundary_violation_count)
    if cond.metric == "imprint_mean":
        return state.imprint_mean
    return None


def _parse_condition(raw: dict) -> PolicyCondition:
    return PolicyCondition(
        metric=raw["metric"],
        layer=raw.get("layer"),
        op=raw["op"],
        threshold=float(raw["threshold"]),
    )


def _parse_action(raw: dict) -> PolicyAction:
    return PolicyAction(
        knob=raw["knob"],
        scope=raw["scope"],
        value=float(raw["value"]),
        ttl_s=float(raw["ttl_s"]),
    )


def load_policy_rules(path: str | Path) -> list[PolicyRule]:
    """Load policy rules from a YAML file.

    Supports both v0.1 (single condition/action) and v0.2 (compound
    conditions with logic, action chains) formats.
    """
    import yaml

    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or "rules" not in data:
        return []
    rules: list[PolicyRule] = []
    for r in data["rules"]:
        # --- condition(s) ---
        if "conditions" in r:
            cond: PolicyCondition | CompoundCondition = CompoundCondition(
                conditions=[_parse_condition(c) for c in r["conditions"]],
                logic=r.get("logic", "AND").upper(),
            )
        else:
            cond = _parse_condition(r["condition"])

        # --- action(s) ---
        if "actions" in r:
            action_list = [_parse_action(a) for a in r["actions"]]
        else:
            action_list = [_parse_action(r["action"])]

        rules.append(
            PolicyRule(
                name=r["name"],
                regimes=[s.upper() for s in r.get("regime", [])],
                condition=cond,
                actions=action_list,
                cooldown_s=float(r.get("cooldown_s", 0.0)),
                max_fires=int(r.get("max_fires", 0)),
            )
        )
    return rules
