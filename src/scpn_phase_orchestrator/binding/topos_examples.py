# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topos domain obligations examples

"""Deterministic topos obligation examples for binding review surfaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from scpn_phase_orchestrator.binding.semantic import compile_symbolic_binding
from scpn_phase_orchestrator.binding.types import BindingSpec
from scpn_phase_orchestrator.supervisor.policy_rules import (
    CompoundCondition,
    PolicyAction,
    PolicyCondition,
    PolicyRule,
)

BOUNDARY_TAG = "categorical_validation_prototype_not_formal_topos_proof"

__all__ = [
    "ToposProofObligation",
    "ToposDomainObligation",
    "build_topos_domain_obligation_examples",
]


@dataclass(frozen=True)
class ToposProofObligation:
    """Single proof obligation attached to one domain example."""

    name: str
    description: str
    passed: bool = True

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe obligation audit record."""
        return {
            "name": str(self.name).strip(),
            "description": str(self.description).strip(),
            "passed": bool(self.passed),
        }


@dataclass(frozen=True)
class ToposDomainObligation:
    """Concrete domain obligation example record.

    The object keeps live repository objects for compilation correctness and
    converts them into deterministic audit material through ``to_audit_record``.
    """

    domain: str
    symbolic_prompt: str
    binding_spec: BindingSpec
    policy_rules: tuple[PolicyRule, ...]
    obligations: tuple[ToposProofObligation, ...]
    binding_object_count: int
    policy_object_count: int
    non_actuating: bool
    proof_boundary: str
    passed: bool

    def to_audit_record(self) -> dict[str, object]:
        """Convert this example into a deterministic JSON-safe audit record."""
        self._validate()
        obligations = [obligation.to_audit_record() for obligation in self.obligations]
        obligation_names = [entry["name"] for entry in obligations]
        record: dict[str, object] = {
            "domain": self.domain,
            "symbolic_prompt": self.symbolic_prompt,
            "binding_object_count": self.binding_object_count,
            "policy_object_count": self.policy_object_count,
            "obligation_names": obligation_names,
            "passed": self.passed,
            "non_actuating": self.non_actuating,
            "proof_boundary": self.proof_boundary,
        }
        record["example_hash"] = self._example_hash(record)
        return record

    def _example_hash(self, record: dict[str, object]) -> str:
        payload = dict(record)
        payload.pop("example_hash", None)
        payload["policy_rules"] = [
            _policy_rule_to_dict(rule) for rule in self.policy_rules
        ]
        payload["obligations"] = [
            obligation.to_audit_record() for obligation in self.obligations
        ]
        payload["binding_spec_name"] = self.binding_spec.name
        payload["binding_spec_signature"] = _binding_signature(self.binding_spec)
        payload_bytes = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload_bytes).hexdigest()

    def _validate(self) -> None:
        if not isinstance(self.domain, str) or not self.domain.strip():
            raise ValueError("example domain must be a non-empty string")
        if not isinstance(self.symbolic_prompt, str):
            raise ValueError("symbolic prompt must be a non-empty string")
        symbolic_prompt_safe = self.symbolic_prompt.strip()
        if not symbolic_prompt_safe:
            raise ValueError("symbolic prompt must be a non-empty string")
        if not isinstance(self.binding_spec, BindingSpec):
            raise ValueError("binding spec must be a BindingSpec")
        if not isinstance(self.policy_rules, tuple) or not self.policy_rules:
            raise ValueError("policy rules must be a non-empty tuple")
        if not isinstance(self.obligations, tuple) or not self.obligations:
            raise ValueError("obligations must be a non-empty tuple")
        if any(not isinstance(item, ToposProofObligation) for item in self.obligations):
            raise ValueError(
                "obligations must contain only ToposProofObligation values"
            )
        if any(not obligation.name.strip() for obligation in self.obligations):
            raise ValueError("obligation names must be non-empty")
        if self.proof_boundary != BOUNDARY_TAG:
            raise ValueError(f"unexpected proof boundary: {self.proof_boundary!r}")
        if (
            not isinstance(self.binding_object_count, int)
            or self.binding_object_count < 1
        ):
            raise ValueError("binding_object_count must be a positive integer")
        if (
            not isinstance(self.policy_object_count, int)
            or self.policy_object_count < 1
        ):
            raise ValueError("policy_object_count must be a positive integer")
        if not isinstance(self.non_actuating, bool) or self.non_actuating is not True:
            raise ValueError("non_actuating must be True")
        if not isinstance(self.passed, bool) or not self.passed:
            raise ValueError("all obligations must be passed")


def _binding_signature(binding: BindingSpec) -> dict[str, object]:
    return {
        "name": binding.name,
        "safety_tier": binding.safety_tier,
        "version": binding.version,
        "layer_count": len(binding.layers),
        "oscillator_count": sum(len(layer.oscillator_ids) for layer in binding.layers),
        "boundary_count": len(binding.boundaries),
        "channel_count": len(binding.channels),
        "actuator_count": len(binding.actuators),
        "cross_channel_coupling_count": len(binding.cross_channel_couplings),
    }


def _policy_rule_to_dict(rule: PolicyRule) -> dict[str, object]:
    condition = rule.condition
    if isinstance(condition, CompoundCondition):
        condition_data: dict[str, object] = {
            "logic": condition.logic,
            "conditions": [
                _policy_condition_to_dict(item) for item in condition.conditions
            ],
        }
    else:
        condition_data = _policy_condition_to_dict(condition)

    return {
        "name": rule.name,
        "regimes": tuple(rule.regimes),
        "condition": condition_data,
        "actions": [
            {
                "knob": action.knob,
                "scope": action.scope,
                "value": action.value,
                "ttl_s": action.ttl_s,
            }
            for action in rule.actions
        ],
        "cooldown_s": rule.cooldown_s,
        "max_fires": rule.max_fires,
    }


def _policy_condition_to_dict(condition: PolicyCondition) -> dict[str, object]:
    return {
        "metric": condition.metric,
        "layer": condition.layer,
        "op": condition.op,
        "threshold": condition.threshold,
    }


def _count_binding_objects(binding: BindingSpec) -> int:
    return (
        len(binding.layers)
        + sum(len(layer.oscillator_ids) for layer in binding.layers)
        + len(binding.oscillator_families)
        + len(binding.channels)
        + len(binding.actuators)
        + len(binding.boundaries)
        + len(binding.channel_groups)
        + len(binding.cross_channel_couplings)
    )


def _count_policy_objects(rules: tuple[PolicyRule, ...]) -> int:
    total = 0
    for rule in rules:
        condition = rule.condition
        if isinstance(condition, CompoundCondition):
            total += 1 + len(condition.conditions)
        else:
            total += 2
        total += len(rule.actions) + 1
    return total


def _build_domain_example(
    *,
    domain: str,
    symbolic_prompt: str,
    compilation_name: str,
    oscillators_per_layer: int,
    dry_run_steps: int,
    policy_rules: tuple[PolicyRule, ...],
    obligations: tuple[tuple[str, str], ...],
) -> ToposDomainObligation:
    artifacts = compile_symbolic_binding(
        symbolic_prompt,
        name=compilation_name,
        oscillators_per_layer=oscillators_per_layer,
        dry_run_steps=dry_run_steps,
        retrieval_root=None,
        docs_root=None,
    )
    if artifacts.validation_errors:
        raise ValueError(
            f"compilation for {domain!r} failed: {artifacts.validation_errors}"
        )
    if not policy_rules:
        raise ValueError("policy rules must be non-empty")

    # Ensure determinism in case caller passes mutable values.
    policy_rules_tuple = tuple(policy_rules)
    obligations_tuple = tuple(
        ToposProofObligation(name=name, description=description)
        for name, description in obligations
    )

    return ToposDomainObligation(
        domain=domain,
        symbolic_prompt=symbolic_prompt,
        binding_spec=artifacts.binding_spec,
        policy_rules=policy_rules_tuple,
        obligations=obligations_tuple,
        binding_object_count=_count_binding_objects(artifacts.binding_spec),
        policy_object_count=_count_policy_objects(policy_rules_tuple),
        non_actuating=True,
        proof_boundary=BOUNDARY_TAG,
        passed=all(obligation.passed for obligation in obligations_tuple),
    )


def _power_grid_policy_rules() -> tuple[PolicyRule, ...]:
    return (
        PolicyRule(
            name="grid_coherence_recovery",
            regimes=["DEGRADED", "CRITICAL"],
            condition=PolicyCondition(metric="R_good", layer=0, op="<", threshold=0.68),
            actions=[
                PolicyAction(
                    knob="K",
                    scope="global",
                    value=0.10,
                    ttl_s=6.0,
                )
            ],
        ),
        PolicyRule(
            name="grid_load_spike_guard",
            regimes=["DEGRADED"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition(
                        metric="pac_max",
                        layer=None,
                        op=">",
                        threshold=0.25,
                    ),
                    PolicyCondition(
                        metric="mean_amplitude",
                        layer=0,
                        op=">",
                        threshold=0.62,
                    ),
                ],
                logic="OR",
            ),
            actions=[
                PolicyAction(knob="alpha", scope="global", value=-0.02, ttl_s=8.0),
                PolicyAction(knob="K", scope="layer_0", value=-0.05, ttl_s=8.0),
            ],
        ),
    )


def _cardiac_policy_rules() -> tuple[PolicyRule, ...]:
    return (
        PolicyRule(
            name="cardiac_arrhythmia_suppression",
            regimes=["DEGRADED", "CRITICAL"],
            condition=PolicyCondition(
                metric="mean_amplitude_layer",
                layer=0,
                op=">",
                threshold=0.52,
            ),
            actions=[
                PolicyAction(
                    knob="zeta",
                    scope="global",
                    value=-0.03,
                    ttl_s=10.0,
                )
            ],
        ),
        PolicyRule(
            name="cardiac_rhythm_stability",
            regimes=["CRITICAL"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition(
                        metric="stability_proxy",
                        layer=None,
                        op="<",
                        threshold=0.46,
                    ),
                    PolicyCondition(
                        metric="R",
                        layer=1,
                        op="<",
                        threshold=0.5,
                    ),
                ],
                logic="AND",
            ),
            actions=[
                PolicyAction(knob="alpha", scope="global", value=0.04, ttl_s=12.0),
            ],
        ),
    )


def _cyber_industrial_policy_rules() -> tuple[PolicyRule, ...]:
    return (
        PolicyRule(
            name="industrial_boundary_repair",
            regimes=["DEGRADED", "CRITICAL"],
            condition=PolicyCondition(
                metric="boundary_violation_count", layer=None, op=">", threshold=0.0
            ),
            actions=[
                PolicyAction(
                    knob="K",
                    scope="global",
                    value=0.07,
                    ttl_s=6.0,
                )
            ],
        ),
        PolicyRule(
            name="industrial_threat_isolation",
            regimes=["CRITICAL"],
            condition=CompoundCondition(
                conditions=[
                    PolicyCondition(
                        metric="imprint_mean",
                        layer=None,
                        op=">",
                        threshold=0.72,
                    ),
                    PolicyCondition(
                        metric="R_good",
                        layer=0,
                        op="<",
                        threshold=0.58,
                    ),
                ],
                logic="OR",
            ),
            actions=[
                PolicyAction(knob="zeta", scope="layer_0", value=-0.06, ttl_s=9.0),
            ],
        ),
    )


def build_topos_domain_obligation_examples() -> tuple[dict[str, object], ...]:
    """Build deterministic topos obligation examples for benchmark consumption."""

    examples = (
        _build_domain_example(
            domain="power_grid",
            symbolic_prompt=(
                "3-layer power grid synchronization with oscillatory frequency"
                " coherence and voltage phase balancing under load shifts"
            ),
            compilation_name="topos_power_grid",
            oscillators_per_layer=2,
            dry_run_steps=2,
            policy_rules=_power_grid_policy_rules(),
            obligations=(
                (
                    "power_grid_coherence_guard",
                    "Maintain categorical coherence boundaries under "
                    "load perturbations.",
                ),
                (
                    "grid_frequency_protective_limit",
                    "Prove stability under stepped frequency excursions.",
                ),
            ),
        ),
        _build_domain_example(
            domain="cardiac_rhythm",
            symbolic_prompt=(
                "2-layer cardiac rhythm monitoring for atrial arrhythmia"
                " synchrony and phase reset timing."
            ),
            compilation_name="topos_cardiac_rhythm",
            oscillators_per_layer=3,
            dry_run_steps=2,
            policy_rules=_cardiac_policy_rules(),
            obligations=(
                (
                    "cardiac_rhythm_variability_guard",
                    "Track rhythm-domain invariants across coupled cardiac layers.",
                ),
                (
                    "cardiac_synchrony_cat_proof",
                    "Preserve rhythm recovery envelope under symbolic perturbations.",
                ),
            ),
        ),
        _build_domain_example(
            domain="cyber_industrial",
            symbolic_prompt=(
                "4-layer cyber industrial control pilot with defensive"
                " synchronization, segmentation, and actuator isolation rules"
            ),
            compilation_name="topos_cyber_industrial",
            oscillators_per_layer=2,
            dry_run_steps=3,
            policy_rules=_cyber_industrial_policy_rules(),
            obligations=(
                (
                    "cyber_industrial_boundary_containment",
                    "Ensure categorical separation between operational "
                    "and threat modes.",
                ),
                (
                    "industrial_attack_mitigation_guard",
                    "Prevent policy drift during incident escalation windows.",
                ),
            ),
        ),
    )

    records = [example.to_audit_record() for example in examples]

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("example manifest must be a dict")
        example_hash = record.get("example_hash")
        if not isinstance(example_hash, str) or not example_hash:
            raise ValueError("each example must have a stable example_hash")

    return tuple(records)
