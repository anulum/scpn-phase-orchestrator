# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy rule definitions

from __future__ import annotations

import operator
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, NoReturn

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

_MAX_POLICY_RULES = 1000
_MAX_CONDITIONS_PER_RULE = 32
_MAX_ACTIONS_PER_RULE = 32


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


def _policy_error(message: str) -> NoReturn:
    raise ValueError(f"invalid policy rules: {message}")


def _require_mapping(raw: Any, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        _policy_error(f"{context} must be a mapping")
    return raw


def _require_field(raw: dict[str, Any], key: str, context: str) -> Any:
    try:
        return raw[key]
    except KeyError:
        _policy_error(f"{context} missing required key {key!r}")


def _require_text(raw: dict[str, Any], key: str, context: str) -> str:
    value = _require_field(raw, key, context)
    if not isinstance(value, str) or not value:
        _policy_error(f"{context}.{key} must be a non-empty string")
    return value


def _finite_float(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        _policy_error(f"{context} must be a finite number")
    if not isfinite(result):
        _policy_error(f"{context} must be a finite number")
    return result


def _non_negative_float(value: Any, context: str) -> float:
    result = _finite_float(value, context)
    if result < 0.0:
        _policy_error(f"{context} must be non-negative")
    return result


def _non_negative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        _policy_error(f"{context} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        _policy_error(f"{context} must be a non-negative integer")
    if result < 0:
        _policy_error(f"{context} must be non-negative")
    return result


def _require_sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _policy_error(f"{context} must be a list")
    return value


def _parse_condition(raw: Any) -> PolicyCondition:
    data = _require_mapping(raw, "condition")
    op = _require_text(data, "op", "condition")
    if op not in _OPS:
        _policy_error("condition.op is unsupported")
    layer_raw = data.get("layer")
    if layer_raw is not None:
        layer = _non_negative_int(layer_raw, "condition.layer")
    else:
        layer = None
    return PolicyCondition(
        metric=_require_text(data, "metric", "condition"),
        layer=layer,
        op=op,
        threshold=_finite_float(
            _require_field(data, "threshold", "condition"), "condition.threshold"
        ),
    )


def _parse_action(raw: Any) -> PolicyAction:
    data = _require_mapping(raw, "action")
    return PolicyAction(
        knob=_require_text(data, "knob", "action"),
        scope=_require_text(data, "scope", "action"),
        value=_finite_float(_require_field(data, "value", "action"), "action.value"),
        ttl_s=_non_negative_float(
            _require_field(data, "ttl_s", "action"), "action.ttl_s"
        ),
    )


def load_policy_rules(path: str | Path) -> list[PolicyRule]:
    """Load policy rules from a YAML file.

    Supports both v0.1 (single condition/action) and v0.2 (compound
    conditions with logic, action chains) formats.
    """
    import yaml

    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ValueError(f"cannot read policy rules: {reason}") from None
    try:
        data = yaml.safe_load(raw)
    except (RecursionError, yaml.YAMLError):
        raise ValueError("policy rules YAML parse error") from None
    if not isinstance(data, dict) or "rules" not in data:
        return []
    rule_data = _require_sequence(data["rules"], "rules")
    if len(rule_data) > _MAX_POLICY_RULES:
        _policy_error("too many rules")
    rules: list[PolicyRule] = []
    for raw_rule in rule_data:
        r = _require_mapping(raw_rule, "rule")
        # --- condition(s) ---
        if "conditions" in r:
            conditions = _require_sequence(r["conditions"], "rule.conditions")
            if not conditions:
                _policy_error("rule.conditions must not be empty")
            if len(conditions) > _MAX_CONDITIONS_PER_RULE:
                _policy_error("too many rule conditions")
            logic = r.get("logic", "AND")
            if not isinstance(logic, str):
                _policy_error("rule.logic must be a string")
            logic = logic.upper()
            if logic not in ("AND", "OR"):
                _policy_error("rule.logic must be AND or OR")
            cond: PolicyCondition | CompoundCondition = CompoundCondition(
                conditions=[_parse_condition(c) for c in conditions],
                logic=logic,
            )
        else:
            cond = _parse_condition(_require_field(r, "condition", "rule"))

        # --- action(s) ---
        if "actions" in r:
            actions = _require_sequence(r["actions"], "rule.actions")
            if not actions:
                _policy_error("rule.actions must not be empty")
            if len(actions) > _MAX_ACTIONS_PER_RULE:
                _policy_error("too many rule actions")
            action_list = [_parse_action(a) for a in actions]
        else:
            action_list = [_parse_action(_require_field(r, "action", "rule"))]

        regimes = r.get("regime", [])
        if not isinstance(regimes, list) or not all(
            isinstance(item, str) and item for item in regimes
        ):
            _policy_error("rule.regime must be a list of non-empty strings")

        rules.append(
            PolicyRule(
                name=_require_text(r, "name", "rule"),
                regimes=[s.upper() for s in regimes],
                condition=cond,
                actions=action_list,
                cooldown_s=_non_negative_float(
                    r.get("cooldown_s", 0.0), "rule.cooldown_s"
                ),
                max_fires=_non_negative_int(r.get("max_fires", 0), "rule.max_fires"),
            )
        )
    return rules
