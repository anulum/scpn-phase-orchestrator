# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Resolved binding summaries

from __future__ import annotations

from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.binding.types import BindingSpec

__all__ = ["resolve_binding_summary"]

_RUNTIME_RATE_LIMITS = {"K": 0.1, "zeta": 0.2, "alpha": 0.1, "Psi": 0.5}
_RUNTIME_VALUE_BOUNDS = {
    "K": (-0.5, 0.5),
    "zeta": (0.0, 0.5),
    "alpha": (-1.0, 1.0),
}


def resolve_binding_summary(
    spec: BindingSpec,
    *,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a serialisable summary of runtime-relevant binding resolution.

    The summary mirrors defaults and derived values used by the CLI run path:
    oscillator counts, layer-to-index ranges, omega defaults, control interval,
    actuator bounds, drive defaults, optional subsystems, and validation-facing
    metadata. It is intentionally read-only and does not construct engines or
    adapters.
    """
    n_oscillators = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_ranges = _resolve_layer_ranges(spec)
    omegas = spec.get_omegas()
    actuator_bounds = _resolve_actuator_bounds(spec)
    driver_configs = spec.drivers.all_channel_configs()
    zeta_initial = max(
        (float(cfg.get("zeta", 0.0)) for cfg in driver_configs.values()),
        default=0.0,
    )

    return {
        "source": Path(spec_path).name if spec_path is not None else None,
        "name": spec.name,
        "version": spec.version,
        "safety_tier": spec.safety_tier,
        "timing": {
            "sample_period_s": spec.sample_period_s,
            "control_period_s": spec.control_period_s,
            "control_interval_steps": max(
                1, round(spec.control_period_s / spec.sample_period_s)
            ),
        },
        "counts": {
            "layers": len(spec.layers),
            "oscillators": n_oscillators,
            "families": len(spec.oscillator_families),
            "boundaries": len(spec.boundaries),
            "actuators": len(spec.actuators),
        },
        "layers": [
            {
                "name": layer.name,
                "index": layer.index,
                "oscillator_count": len(layer.oscillator_ids),
                "oscillator_ids": list(layer.oscillator_ids),
                "range": layer_ranges[layer.index],
                "family": layer.family,
                "omegas": omegas[layer_ranges[layer.index][0] : layer_ranges[
                    layer.index
                ][1]],
                "omega_source": "explicit" if layer.omegas is not None else "default",
            }
            for layer in spec.layers
        ],
        "oscillator_families": {
            name: {
                "channel": family.channel,
                "extractor_type": family.extractor_type,
                "config_keys": sorted(family.config),
            }
            for name, family in sorted(spec.oscillator_families.items())
        },
        "coupling": {
            "base_strength": spec.coupling.base_strength,
            "decay_alpha": spec.coupling.decay_alpha,
            "template_count": len(spec.coupling.templates),
        },
        "drivers": {
            "channels": sorted(driver_configs),
            "zeta_initial": zeta_initial,
            "psi_initial": float(spec.drivers.physical.get("psi", 0.0)),
            "psi_driver": _resolve_psi_driver(spec),
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
                "lower": boundary.lower,
                "upper": boundary.upper,
                "severity": boundary.severity,
            }
            for boundary in spec.boundaries
        ],
        "actuation": {
            "rate_limits": dict(_RUNTIME_RATE_LIMITS),
            "value_bounds": actuator_bounds,
            "value_bounds_source": "actuators"
            if spec.actuators
            else "runtime_defaults",
            "actuators": [
                {
                    "name": actuator.name,
                    "knob": actuator.knob,
                    "scope": actuator.scope,
                    "limits": list(actuator.limits),
                }
                for actuator in spec.actuators
            ],
        },
        "optional": {
            "amplitude_mode": spec.amplitude is not None,
            "imprint_model": spec.imprint_model is not None,
            "geometry_prior": spec.geometry_prior is not None,
            "protocol_net": spec.protocol_net is not None,
        },
        "defaults_applied": {
            "omegas": [
                layer.name for layer in spec.layers if layer.omegas is None
            ],
            "actuator_bounds": []
            if spec.actuators
            else sorted(_RUNTIME_VALUE_BOUNDS),
            "drivers": [
                channel
                for channel, cfg in sorted(driver_configs.items())
                if not cfg
            ],
        },
    }


def _resolve_layer_ranges(spec: BindingSpec) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    cursor = 0
    for layer in spec.layers:
        next_cursor = cursor + len(layer.oscillator_ids)
        ranges[layer.index] = (cursor, next_cursor)
        cursor = next_cursor
    return ranges


def _resolve_actuator_bounds(spec: BindingSpec) -> dict[str, tuple[float, float]]:
    value_bounds: dict[str, tuple[float, float]] = {}
    for actuator in spec.actuators:
        if actuator.limits and len(actuator.limits) == 2:
            value_bounds[actuator.knob] = (
                float(actuator.limits[0]),
                float(actuator.limits[1]),
            )
    return value_bounds or dict(_RUNTIME_VALUE_BOUNDS)


def _resolve_psi_driver(spec: BindingSpec) -> str | None:
    if "frequency" in spec.drivers.physical:
        return "physical.frequency"
    if "cadence_hz" in spec.drivers.informational:
        return "informational.cadence_hz"
    if "sequence" in spec.drivers.symbolic:
        return "symbolic.sequence"
    return None
