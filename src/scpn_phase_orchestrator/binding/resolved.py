# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Resolved binding configuration summary

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import BindingSpec, resolve_extractor_type

__all__ = ["format_resolved_binding_config", "resolved_binding_config"]


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def resolved_binding_config(spec: BindingSpec) -> dict[str, object]:
    """Build a deterministic, JSON-safe summary of binding runtime choices.

    The summary intentionally exposes structural choices, enabled features, and
    driver key names only. It does not copy raw driver configuration values into
    audit metadata because production driver blocks may contain endpoints or
    deployment-local identifiers.
    """
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    control_interval_steps = max(1, round(spec.control_period_s / spec.sample_period_s))
    family_channels = {
        name: family.channel
        for name, family in sorted(spec.oscillator_families.items())
    }
    driver_configs = spec.drivers.all_channel_configs()
    channels = sorted(spec.used_channels())

    family_summaries: dict[str, dict[str, object]] = {}
    for name, family in sorted(spec.oscillator_families.items()):
        family_summaries[name] = {
            "channel": family.channel,
            "extractor_type": family.extractor_type,
            "resolved_extractor_type": resolve_extractor_type(family.extractor_type),
            "config_keys": sorted(family.config),
        }

    layer_summaries: list[dict[str, object]] = []
    for layer in sorted(spec.layers, key=lambda item: item.index):
        channel = family_channels.get(layer.family) if layer.family else None
        layer_summaries.append(
            {
                "name": layer.name,
                "index": layer.index,
                "family": layer.family,
                "channel": channel,
                "oscillator_count": len(layer.oscillator_ids),
            }
        )

    channel_summaries: dict[str, dict[str, object]] = {}
    for channel in channels:
        family_names = sorted(
            name
            for name, family_channel in family_channels.items()
            if family_channel == channel
        )
        channel_layers = [
            layer
            for layer in spec.layers
            if layer.family is not None and layer.family in family_names
        ]
        extractors = sorted(
            {
                resolve_extractor_type(spec.oscillator_families[name].extractor_type)
                for name in family_names
            }
        )
        driver_config = driver_configs.get(channel, {})
        channel_spec = spec.channels.get(channel)
        channel_summaries[channel] = {
            "families": family_names,
            "extractors": extractors,
            "driver_configured": bool(driver_config),
            "driver_keys": sorted(driver_config),
            "layer_count": len(channel_layers),
            "oscillator_count": sum(
                len(layer.oscillator_ids) for layer in channel_layers
            ),
            "declared": channel_spec is not None,
            "role": channel_spec.role if channel_spec is not None else None,
            "required": channel_spec.required if channel_spec is not None else None,
            "units": channel_spec.units if channel_spec is not None else None,
            "metric_semantics": (
                channel_spec.metric_semantics if channel_spec is not None else None
            ),
            "coupling_participation": (
                channel_spec.coupling_participation
                if channel_spec is not None
                else None
            ),
            "audit_serialisation": (
                channel_spec.audit_serialisation if channel_spec is not None else None
            ),
            "replay_semantics": (
                channel_spec.replay_semantics if channel_spec is not None else None
            ),
            "supervisor_visibility": (
                channel_spec.supervisor_visibility if channel_spec is not None else None
            ),
            "derived_from": (
                list(channel_spec.derived_from) if channel_spec is not None else []
            ),
            "derive_rule": channel_spec.derive_rule
            if channel_spec is not None
            else None,
        }

    features = {
        "amplitude": spec.amplitude is not None,
        "geometry_prior": spec.geometry_prior is not None,
        "imprint_model": spec.imprint_model is not None,
        "protocol_net": spec.protocol_net is not None,
    }

    return {
        "name": spec.name,
        "version": spec.version,
        "safety_tier": spec.safety_tier,
        "sample_period_s": spec.sample_period_s,
        "control_period_s": spec.control_period_s,
        "control_interval_steps": control_interval_steps,
        "engine_mode": "stuart_landau" if spec.amplitude is not None else "kuramoto",
        "layer_count": len(spec.layers),
        "oscillator_count": n_osc,
        "channels": channel_summaries,
        "channel_groups": {
            name: {
                "channels": list(group.channels),
                "required": group.required,
                "description": group.description,
            }
            for name, group in sorted(spec.channel_groups.items())
        },
        "cross_channel_couplings": [
            {
                "source": coupling.source,
                "target": coupling.target,
                "strength": coupling.strength,
                "mode": coupling.mode,
                "template": coupling.template,
            }
            for coupling in spec.cross_channel_couplings
        ],
        "families": family_summaries,
        "layers": layer_summaries,
        "unassigned_layer_count": sum(
            1 for layer in spec.layers if layer.family is None
        ),
        "coupling": {
            "base_strength": spec.coupling.base_strength,
            "decay_alpha": spec.coupling.decay_alpha,
            "templates": sorted(spec.coupling.templates),
        },
        "objectives": {
            "good_layers": list(spec.objectives.good_layers),
            "bad_layers": list(spec.objectives.bad_layers),
            "good_weight": spec.objectives.good_weight,
            "bad_weight": spec.objectives.bad_weight,
        },
        "boundaries": [
            {
                "name": boundary.name,
                "variable": boundary.variable,
                "severity": boundary.severity,
            }
            for boundary in sorted(spec.boundaries, key=lambda item: item.name)
        ],
        "actuators": [
            {
                "name": actuator.name,
                "knob": actuator.knob,
                "scope": actuator.scope,
                "limits": list(actuator.limits),
            }
            for actuator in sorted(spec.actuators, key=lambda item: item.name)
        ],
        "features": features,
    }


def format_resolved_binding_config(summary: dict[str, object]) -> list[str]:
    """Render a compact, human-readable summary for CLI output."""
    channels = summary["channels"]
    assert isinstance(channels, dict)  # nosec B101 — internal summary shape
    features = summary["features"]
    assert isinstance(features, dict)  # nosec B101 — internal summary shape
    enabled_features = sorted(name for name, enabled in features.items() if enabled)
    feature_text = ", ".join(enabled_features) if enabled_features else "none"
    channel_names = ", ".join(sorted(str(channel) for channel in channels)) or "none"

    lines = [
        "Resolved configuration:",
        (
            f"  domain: {summary['name']} v{summary['version']} "
            f"({summary['safety_tier']})"
        ),
        (
            f"  timing: sample={summary['sample_period_s']}s "
            f"control={summary['control_period_s']}s "
            f"interval={summary['control_interval_steps']} steps"
        ),
        (
            f"  structure: layers={summary['layer_count']} "
            f"oscillators={summary['oscillator_count']} channels={channel_names}"
        ),
        f"  engine: {summary['engine_mode']} features={feature_text}",
    ]

    for channel, raw_info in sorted(channels.items(), key=lambda item: str(item[0])):
        assert isinstance(raw_info, dict)  # nosec B101 — internal summary shape
        families = _string_list(raw_info.get("families")) or ["none"]
        extractors = _string_list(raw_info.get("extractors")) or ["none"]
        driver_keys = _string_list(raw_info.get("driver_keys")) or ["none"]
        lines.append(
            f"  channel {channel}: families={','.join(families)} "
            f"extractors={','.join(extractors)} "
            f"driver_keys={','.join(driver_keys)} "
            f"layers={raw_info.get('layer_count', 0)} "
            f"oscillators={raw_info.get('oscillator_count', 0)}"
        )
        if raw_info.get("declared"):
            role = raw_info.get("role") or "domain"
            replay = raw_info.get("replay_semantics") or "phase"
            derived_from = _string_list(raw_info.get("derived_from"))
            derived = f" derived_from={','.join(derived_from)}" if derived_from else ""
            lines.append(
                f"    metadata: role={role} replay={replay} "
                f"supervisor={raw_info.get('supervisor_visibility')}{derived}"
            )

    groups = summary.get("channel_groups", {})
    if isinstance(groups, dict) and groups:
        group_names = ", ".join(sorted(str(name) for name in groups))
        lines.append(f"  channel_groups: {group_names}")

    couplings = summary.get("cross_channel_couplings", [])
    if isinstance(couplings, list) and couplings:
        lines.append(f"  cross_channel_couplings: {len(couplings)}")

    unassigned = summary.get("unassigned_layer_count", 0)
    if unassigned:
        lines.append(
            f"  note: {unassigned} layer(s) have no explicit oscillator family binding"
        )
    return lines
