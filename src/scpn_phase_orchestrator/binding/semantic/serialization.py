# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Semantic compiler YAML serialisation

"""Binding-spec and policy YAML serialisation and confidence scoring."""

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import (
    BindingSpec,
)


def _confidence(
    *,
    matched_keywords: list[str],
    has_layer_count: bool,
    domain_family: str,
    retrieval_score: float,
) -> float:
    score = 0.35 + min(0.25, 0.05 * len(matched_keywords))
    if has_layer_count:
        score += 0.15
    if domain_family != "generic":
        score += 0.2
    if retrieval_score > 0.0:
        score += min(0.15, 0.05 + 0.1 * retrieval_score)
    return round(min(score, 0.95), 3)


def _policy_yaml_for(spec: BindingSpec) -> str:
    import yaml

    policy = {
        "rules": [
            {
                "name": "recover_low_coherence",
                "regime": ["DEGRADED", "CRITICAL"],
                "condition": {
                    "metric": "R_good",
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {
                    "knob": "K",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": max(spec.control_period_s, 1.0),
                },
            }
        ]
    }
    rendered: str = yaml.safe_dump(policy, sort_keys=False)
    return rendered


def _binding_spec_to_yaml(spec: BindingSpec) -> str:
    import yaml

    data = {
        "name": spec.name,
        "version": spec.version,
        "safety_tier": spec.safety_tier,
        "sample_period_s": spec.sample_period_s,
        "control_period_s": spec.control_period_s,
        "layers": [
            {
                "name": layer.name,
                "index": layer.index,
                "oscillator_ids": layer.oscillator_ids,
                "omegas": layer.omegas,
                "family": layer.family,
            }
            for layer in spec.layers
        ],
        "oscillator_families": {
            key: {
                "channel": family.channel,
                "extractor_type": family.extractor_type,
                "config": family.config,
            }
            for key, family in spec.oscillator_families.items()
        },
        "coupling": {
            "base_strength": spec.coupling.base_strength,
            "decay_alpha": spec.coupling.decay_alpha,
            "templates": spec.coupling.templates,
        },
        "drivers": {
            "physical": spec.drivers.physical,
            "informational": spec.drivers.informational,
            "symbolic": spec.drivers.symbolic,
        },
        "channels": {
            key: {
                "role": channel.role,
                "required": channel.required,
                "units": channel.units,
                "metric_semantics": channel.metric_semantics,
                "coupling_participation": channel.coupling_participation,
                "audit_serialisation": channel.audit_serialisation,
                "replay_semantics": channel.replay_semantics,
                "supervisor_visibility": channel.supervisor_visibility,
                "derived_from": channel.derived_from,
                "derive_rule": channel.derive_rule,
            }
            for key, channel in spec.channels.items()
        },
        "channel_groups": {
            key: {
                "channels": group.channels,
                "required": group.required,
                "description": group.description,
            }
            for key, group in spec.channel_groups.items()
        },
        "objectives": {
            "good_layers": spec.objectives.good_layers,
            "bad_layers": spec.objectives.bad_layers,
            "good_weight": spec.objectives.good_weight,
            "bad_weight": spec.objectives.bad_weight,
        },
        "boundaries": [
            {
                "name": boundary.name,
                "variable": boundary.variable,
                "lower": boundary.lower,
                "upper": boundary.upper,
                "severity": boundary.severity,
            }
            for boundary in spec.boundaries
        ],
        "actuators": [
            {
                "name": actuator.name,
                "knob": actuator.knob,
                "scope": actuator.scope,
                "limits": list(actuator.limits),
            }
            for actuator in spec.actuators
        ],
        "protocol_net": {
            "places": spec.protocol_net.places if spec.protocol_net else [],
            "initial": spec.protocol_net.initial if spec.protocol_net else {},
            "place_regime": spec.protocol_net.place_regime if spec.protocol_net else {},
            "transitions": [
                {
                    "name": transition.name,
                    "inputs": transition.inputs,
                    "outputs": transition.outputs,
                    "guard": transition.guard,
                }
                for transition in (
                    spec.protocol_net.transitions if spec.protocol_net else []
                )
            ],
        },
    }
    rendered: str = yaml.safe_dump(data, sort_keys=False)
    return rendered
