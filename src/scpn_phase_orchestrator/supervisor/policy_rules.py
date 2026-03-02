# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import operator
from dataclasses import dataclass
from pathlib import Path

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor.regimes import Regime
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["PolicyRule", "PolicyEngine", "load_policy_rules"]

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
}


@dataclass(frozen=True)
class PolicyCondition:
    metric: str  # "R_good", "R_bad", "R"
    layer: int | None
    op: str
    threshold: float


@dataclass(frozen=True)
class PolicyRule:
    name: str
    regimes: list[str]
    condition: PolicyCondition
    knob: str
    scope: str
    value: float
    ttl_s: float


class PolicyEngine:
    """Evaluate domainpack policy rules against current state."""

    def __init__(self, rules: list[PolicyRule]) -> None:
        self._rules = rules

    def evaluate(
        self,
        regime: Regime,
        upde_state: UPDEState,
        good_layers: list[int],
        bad_layers: list[int],
    ) -> list[ControlAction]:
        actions: list[ControlAction] = []
        for rule in self._rules:
            if regime.value.upper() not in rule.regimes:
                continue
            val = self._extract_metric(
                rule.condition, upde_state, good_layers, bad_layers,
            )
            if val is None:
                continue
            cmp = _OPS.get(rule.condition.op)
            if cmp is None:
                continue
            if cmp(val, rule.condition.threshold):
                actions.append(
                    ControlAction(
                        knob=rule.knob,
                        scope=rule.scope,
                        value=rule.value,
                        ttl_s=rule.ttl_s,
                        justification=f"policy rule: {rule.name}",
                    )
                )
        return actions

    @staticmethod
    def _extract_metric(
        cond: PolicyCondition,
        state: UPDEState,
        good_layers: list[int],
        bad_layers: list[int],
    ) -> float | None:
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
        return None


def load_policy_rules(path: str | Path) -> list[PolicyRule]:
    """Load policy rules from a YAML file."""
    import yaml

    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or "rules" not in data:
        return []
    rules: list[PolicyRule] = []
    for r in data["rules"]:
        cond = r["condition"]
        rules.append(
            PolicyRule(
                name=r["name"],
                regimes=[s.upper() for s in r.get("regime", [])],
                condition=PolicyCondition(
                    metric=cond["metric"],
                    layer=cond.get("layer"),
                    op=cond["op"],
                    threshold=float(cond["threshold"]),
                ),
                knob=r["action"]["knob"],
                scope=r["action"]["scope"],
                value=float(r["action"]["value"]),
                ttl_s=float(r["action"]["ttl_s"]),
            )
        )
    return rules
