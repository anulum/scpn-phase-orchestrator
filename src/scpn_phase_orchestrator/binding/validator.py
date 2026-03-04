# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from scpn_phase_orchestrator.binding.types import (
    VALID_CHANNELS,
    VALID_EXTRACTORS,
    VALID_KNOBS,
    VALID_SAFETY_TIERS,
    VALID_SEVERITIES,
    BindingSpec,
)

__all__ = ["validate_binding_spec"]


def validate_binding_spec(spec: BindingSpec) -> list[str]:
    """Validate a BindingSpec. Returns list of error strings; empty means valid."""
    errors: list[str] = []

    if not spec.name:
        errors.append("name must be non-empty")

    parts = spec.version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        errors.append(f"version must be major.minor.patch, got {spec.version!r}")

    if spec.safety_tier not in VALID_SAFETY_TIERS:
        errors.append(
            f"safety_tier must be one of {VALID_SAFETY_TIERS}, got {spec.safety_tier!r}"
        )

    if spec.sample_period_s <= 0:
        errors.append(f"sample_period_s must be > 0, got {spec.sample_period_s}")

    if spec.control_period_s <= 0:
        errors.append(f"control_period_s must be > 0, got {spec.control_period_s}")

    if spec.control_period_s < spec.sample_period_s:
        errors.append("control_period_s must be >= sample_period_s")

    if not spec.layers:
        errors.append("at least one layer is required")

    layer_indices = {lay.index for lay in spec.layers}

    for family_name, fam in spec.oscillator_families.items():
        if fam.channel not in VALID_CHANNELS:
            errors.append(
                f"oscillator_family {family_name!r}: channel must be one of "
                f"{VALID_CHANNELS}, got {fam.channel!r}"
            )
        if fam.extractor_type not in VALID_EXTRACTORS:
            errors.append(
                f"oscillator_family {family_name!r}: extractor_type must be one of "
                f"{sorted(VALID_EXTRACTORS)}, got {fam.extractor_type!r}"
            )

    if not spec.objectives.good_layers and not spec.objectives.bad_layers:
        errors.append("objectives must define at least one good or bad layer")

    for ref in spec.objectives.good_layers + spec.objectives.bad_layers:
        if ref not in layer_indices:
            errors.append(f"objectives reference layer index {ref} not in layers")

    for bdef in spec.boundaries:
        if bdef.severity not in VALID_SEVERITIES:
            errors.append(
                f"boundary {bdef.name!r}: severity must be one of "
                f"{VALID_SEVERITIES}, got {bdef.severity!r}"
            )
        if (
            bdef.lower is not None
            and bdef.upper is not None
            and bdef.lower > bdef.upper
        ):
            errors.append(
                f"boundary {bdef.name!r}: lower ({bdef.lower}) "
                f"must be <= upper ({bdef.upper})"
            )

    valid_scopes = {"global"} | {f"layer_{lay.index}" for lay in spec.layers}
    for act in spec.actuators:
        if act.knob not in VALID_KNOBS:
            errors.append(
                f"actuator {act.name!r}: knob must be one of "
                f"{VALID_KNOBS}, got {act.knob!r}"
            )
        if len(act.limits) != 2 or act.limits[0] > act.limits[1]:
            errors.append(
                f"actuator {act.name!r}: limits must be (lo, hi) with lo <= hi"
            )
        if act.scope not in valid_scopes:
            errors.append(
                f"actuator {act.name!r}: scope {act.scope!r} does not match any "
                f"layer index; valid scopes: {sorted(valid_scopes)}"
            )

    if spec.imprint_model is not None:
        if spec.imprint_model.decay_rate < 0.0:
            errors.append(
                f"imprint_model.decay_rate must be >= 0, "
                f"got {spec.imprint_model.decay_rate}"
            )
        if spec.imprint_model.saturation <= 0.0:
            errors.append(
                f"imprint_model.saturation must be > 0, "
                f"got {spec.imprint_model.saturation}"
            )

    if spec.amplitude is not None:
        import math

        if not math.isfinite(spec.amplitude.mu):
            errors.append("amplitude.mu must be finite")
        if spec.amplitude.epsilon < 0.0:
            errors.append(
                f"amplitude.epsilon must be >= 0, got {spec.amplitude.epsilon}"
            )

    return errors
