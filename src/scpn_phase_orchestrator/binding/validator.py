# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding spec validator

"""Cross-field validation for loaded binding specifications.

`validate_binding_spec` returns every actionable configuration error it can
find instead of failing at the first issue. It checks version shape, safety
tier, timing, layer/objective references, N-channel declarations, extractor
aliases, boundary and actuator scopes, imprint, amplitude, geometry, and
protocol-net consistency before a binding is used by runtime code.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass

from scpn_phase_orchestrator.binding.types import (
    STANDARD_CHANNELS,
    VALID_EXTRACTORS,
    VALID_KNOBS,
    VALID_SAFETY_TIERS,
    VALID_SEVERITIES,
    VALID_VALIDATION_TIERS,
    BindingSpec,
    ProtocolNetSpec,
    is_valid_channel_id,
)
from scpn_phase_orchestrator.exceptions import PolicyError

__all__ = ["validate_binding_spec", "validate_binding_spec_security"]

_VALID_CROSS_CHANNEL_MODES = frozenset(
    {"bidirectional", "directed", "excitatory", "inhibitory"}
)
_VALID_REPLAY_SEMANTICS = frozenset({"phase", "event", "state", "derived", "external"})
_EXECUTABLE_CONFIG_MARKERS = (
    "!!python",
    "python/object",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "subprocess",
    "os.system",
    "pickle.loads",
    "marshal.loads",
    "lambda ",
    "builtins.",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
)


def validate_binding_spec(spec: BindingSpec) -> list[str]:
    """Validate a BindingSpec and return a list of error strings.

    Parameters
    ----------
    spec : BindingSpec
        The binding specification to validate.

    Returns
    -------
    list[str]
        Human-readable validation error messages; an empty list means the
        spec is structurally and cross-field valid.
    """
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

    if spec.validation_tier not in VALID_VALIDATION_TIERS:
        errors.append(
            "validation_tier must be one of "
            f"{VALID_VALIDATION_TIERS}, got {spec.validation_tier!r}"
        )

    if not math.isfinite(spec.sample_period_s) or spec.sample_period_s <= 0:
        errors.append(
            f"sample_period_s must be finite and > 0, got {spec.sample_period_s}"
        )

    if not math.isfinite(spec.control_period_s) or spec.control_period_s <= 0:
        errors.append(
            f"control_period_s must be finite and > 0, got {spec.control_period_s}"
        )

    if spec.control_period_s < spec.sample_period_s:
        errors.append("control_period_s must be >= sample_period_s")

    if not spec.layers:
        errors.append("at least one layer is required")

    layer_indices = {lay.index for lay in spec.layers}
    oscillator_family_names = set(spec.oscillator_families)

    for layer in spec.layers:
        if layer.family is not None and layer.family not in oscillator_family_names:
            errors.append(
                f"layer {layer.name!r}: family {layer.family!r} is not defined "
                "in oscillator_families"
            )

    used_channels = spec.used_channels()
    for channel_id in used_channels:
        if not is_valid_channel_id(channel_id):
            errors.append(
                f"channel {channel_id!r}: must match [A-Za-z][A-Za-z0-9_-]{{0,63}}"
            )

    family_driver_channels = {
        family.channel for family in spec.oscillator_families.values()
    }
    family_driver_channels.update(spec.drivers.all_channel_configs())
    declared_or_used = family_driver_channels | set(spec.channels)
    undeclared_named_channels = sorted(
        channel
        for channel in family_driver_channels
        if channel not in STANDARD_CHANNELS and channel not in spec.channels
    )
    for channel in undeclared_named_channels:
        errors.append(
            f"channel {channel!r}: named N-channel drivers/families must be "
            "declared under channels"
        )

    for channel_name, channel_spec in spec.channels.items():
        if channel_spec.replay_semantics not in _VALID_REPLAY_SEMANTICS:
            errors.append(
                f"channel {channel_name!r}: replay_semantics must be one of "
                f"{sorted(_VALID_REPLAY_SEMANTICS)}, "
                f"got {channel_spec.replay_semantics!r}"
            )
        for source in channel_spec.derived_from:
            if source == channel_name:
                errors.append(
                    f"channel {channel_name!r}: derived_from must not include itself"
                )
            if source not in declared_or_used:
                errors.append(
                    f"channel {channel_name!r}: derived_from references unknown "
                    f"channel {source!r}"
                )
        if channel_spec.derived_from and not channel_spec.derive_rule:
            errors.append(
                f"channel {channel_name!r}: derive_rule is required when "
                "derived_from is set"
            )
        if channel_spec.derived_from and channel_spec.replay_semantics != "derived":
            errors.append(
                f"channel {channel_name!r}: derived_from channels must use "
                "replay_semantics='derived'"
            )
        if channel_spec.replay_semantics == "derived" and not channel_spec.derived_from:
            errors.append(
                f"channel {channel_name!r}: replay_semantics='derived' requires "
                "derived_from"
            )
        if channel_spec.derive_rule and not channel_spec.derived_from:
            errors.append(
                f"channel {channel_name!r}: derive_rule requires derived_from"
            )
        if (
            channel_spec.required
            and not channel_spec.derived_from
            and channel_name not in family_driver_channels
        ):
            errors.append(
                f"channel {channel_name!r}: required channel must be backed by "
                "an oscillator family or driver"
            )

    for group_name, group in spec.channel_groups.items():
        if not group.channels:
            errors.append(f"channel_group {group_name!r}: channels must not be empty")
        for channel in group.channels:
            if channel not in declared_or_used:
                errors.append(
                    f"channel_group {group_name!r}: references unknown channel "
                    f"{channel!r}"
                )

    for i, coupling in enumerate(spec.cross_channel_couplings):
        if coupling.source not in declared_or_used:
            errors.append(
                f"cross_channel_couplings[{i}]: source references unknown channel "
                f"{coupling.source!r}"
            )
        if coupling.target not in declared_or_used:
            errors.append(
                f"cross_channel_couplings[{i}]: target references unknown channel "
                f"{coupling.target!r}"
            )
        if coupling.source == coupling.target:
            errors.append(
                f"cross_channel_couplings[{i}]: source and target must differ"
            )
        if not math.isfinite(coupling.strength) or coupling.strength < 0.0:
            errors.append(
                f"cross_channel_couplings[{i}].strength must be finite and >= 0"
            )
        if coupling.mode not in _VALID_CROSS_CHANNEL_MODES:
            errors.append(
                f"cross_channel_couplings[{i}].mode must be one of "
                f"{sorted(_VALID_CROSS_CHANNEL_MODES)}, got {coupling.mode!r}"
            )

    for family_name, fam in spec.oscillator_families.items():
        if not is_valid_channel_id(fam.channel):
            errors.append(
                f"oscillator_family {family_name!r}: channel must be a non-empty "
                f"identifier matching [A-Za-z][A-Za-z0-9_-]{{0,63}}, got "
                f"{fam.channel!r}"
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
        if (
            len(act.limits) != 2
            or not all(math.isfinite(limit) for limit in act.limits)
            or act.limits[0] > act.limits[1]
        ):
            errors.append(f"actuator {act.name!r}: limits must be finite and lo <= hi")
        if act.scope not in valid_scopes:
            errors.append(
                f"actuator {act.name!r}: scope {act.scope!r} does not match any "
                f"layer index; valid scopes: {sorted(valid_scopes)}"
            )
    errors.extend(_actuator_knob_consistency_errors(spec))
    errors.extend(_protocol_net_errors(spec))

    if spec.imprint_model is not None:
        if (
            not math.isfinite(spec.imprint_model.decay_rate)
            or spec.imprint_model.decay_rate < 0.0
        ):
            errors.append(
                "imprint_model.decay_rate must be finite and >= 0, "
                f"got {spec.imprint_model.decay_rate}"
            )
        if (
            not math.isfinite(spec.imprint_model.saturation)
            or spec.imprint_model.saturation <= 0.0
        ):
            errors.append(
                "imprint_model.saturation must be finite and > 0, "
                f"got {spec.imprint_model.saturation}"
            )

    if spec.amplitude is not None:
        if not math.isfinite(spec.amplitude.mu):
            errors.append("amplitude.mu must be finite")
        if not math.isfinite(spec.amplitude.epsilon) or spec.amplitude.epsilon < 0.0:
            errors.append(
                "amplitude.epsilon must be finite and >= 0, "
                f"got {spec.amplitude.epsilon}"
            )

    return errors


def _actuator_knob_consistency_errors(spec: BindingSpec) -> list[str]:
    """Return errors for actuators that give one knob different bounds.

    The runtime ``ActionProjector`` holds one value bound and one rate limit per
    knob and refuses a binding whose actuator records disagree (records without
    a ``rate_limit_per_step`` do not take part in the rate check), so such a
    spec would pass validation and then fail on the first ``simulate`` call.
    """
    errors: list[str] = []
    bounds_by_knob: dict[str, tuple[str, tuple[float, ...]]] = {}
    rate_by_knob: dict[str, tuple[str, float]] = {}
    for act in spec.actuators:
        limits = tuple(float(limit) for limit in act.limits)
        seen_bounds = bounds_by_knob.setdefault(act.knob, (act.name, limits))
        if seen_bounds[1] != limits:
            errors.append(
                f"actuator {act.name!r}: knob {act.knob!r} limits {list(limits)} "
                f"differ from actuator {seen_bounds[0]!r} {list(seen_bounds[1])}; "
                "the runtime projector holds one bound per knob"
            )
        if act.rate_limit_per_step is None:
            continue
        rate = float(act.rate_limit_per_step)
        seen_rate = rate_by_knob.setdefault(act.knob, (act.name, rate))
        if seen_rate[1] != rate:
            errors.append(
                f"actuator {act.name!r}: knob {act.knob!r} rate_limit_per_step "
                f"{rate!r} differs from actuator {seen_rate[0]!r} {seen_rate[1]!r}; "
                "the runtime projector holds one rate limit per knob"
            )
    return errors


def _protocol_net_errors(spec: BindingSpec) -> list[str]:
    """Return protocol-net errors the runtime Petri-net builder would hit."""
    net = spec.protocol_net
    if net is None:
        return []
    # Imported here: the supervisor package's import chain reaches back into
    # this package through monitor.boundaries, so a module-level import makes
    # ``import scpn_phase_orchestrator.monitor.boundaries`` fail when first.
    from scpn_phase_orchestrator.supervisor.petri_net import parse_guard

    errors: list[str] = []
    places = set(net.places)
    if not all(isinstance(place, str) and place for place in net.places):
        errors.append("protocol_net.places must be non-empty strings")
    for place, tokens in net.initial.items():
        if place not in places:
            errors.append(f"protocol_net.initial: unknown place {place!r}")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            errors.append(
                f"protocol_net.initial[{place!r}] must be a non-negative integer"
            )
    for place in net.place_regime:
        if place not in places:
            errors.append(f"protocol_net.place_regime: unknown place {place!r}")
    for transition in net.transitions:
        if transition.guard is not None:
            try:
                guard = parse_guard(transition.guard)
            except (PolicyError, TypeError, ValueError) as exc:
                errors.append(
                    f"protocol_net.transition {transition.name!r}: guard "
                    f"{transition.guard!r} is not 'metric op threshold': {exc}"
                )
            else:
                if guard.metric in places:
                    errors.append(
                        f"protocol_net.transition {transition.name!r}: guard metric "
                        f"{guard.metric!r} is a place name; guards read context "
                        "metrics, and token availability is set by input arcs"
                    )
        for side, arcs in (
            ("inputs", transition.inputs),
            ("outputs", transition.outputs),
        ):
            for index, arc in enumerate(arcs):
                where = f"protocol_net.transition {transition.name!r} {side}[{index}]"
                if not isinstance(arc, Mapping) or "place" not in arc:
                    errors.append(
                        f"{where} must be a mapping {{place: <name>, weight: <int>}}, "
                        f"got {arc!r}"
                    )
                    continue
                if arc["place"] not in places:
                    errors.append(f"{where}: unknown place {arc['place']!r}")
                weight = arc.get("weight", 1)
                if (
                    isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or weight < 1
                ):
                    errors.append(f"{where}: weight must be a positive integer")
    if errors:
        return errors
    return _protocol_net_build_errors(net)


def _protocol_net_build_errors(net: ProtocolNetSpec) -> list[str]:
    """Build the protocol net exactly as ``simulate`` does and report refusals.

    The structural checks above give precise messages; this catches every
    remaining rule of the runtime builders (for example an empty or unknown
    ``place_regime``), so a spec that validates is one ``simulate`` accepts.
    """
    # Imported here for the same import-cycle reason as in _protocol_net_errors.
    from scpn_phase_orchestrator.supervisor.petri_adapter import PetriNetAdapter
    from scpn_phase_orchestrator.supervisor.petri_net import petri_net_from_protocol

    try:
        built, marking = petri_net_from_protocol(net)
        PetriNetAdapter(built, marking, net.place_regime)
    except (PolicyError, TypeError, ValueError) as exc:
        return [f"protocol_net is refused by the runtime: {exc}"]
    return []


def validate_binding_spec_security(spec: BindingSpec) -> list[str]:
    """Return security-review findings for a loaded binding spec.

    Normal binding validation checks structure and cross-field consistency.
    This stricter pass is intended for ``spo validate --security`` and rejects
    executable-looking payloads in free-form configuration fields. Binding specs
    remain declarative data; they must not carry Python code, loader tags,
    import expressions, subprocess references, or deserialisation gadgets.

    Parameters
    ----------
    spec : BindingSpec
        The loaded binding specification to security-review.

    Returns
    -------
    list[str]
        Security-review findings; an empty list means no executable-looking
        payloads were detected in free-form configuration fields.
    """
    findings: list[str] = []
    for location, value in _walk_security_values(spec, "binding"):
        if isinstance(value, str):
            normalised = value.strip().lower()
            marker = next(
                (
                    marker
                    for marker in _EXECUTABLE_CONFIG_MARKERS
                    if marker in normalised
                ),
                None,
            )
            if marker is not None:
                findings.append(
                    f"{location}: executable-looking marker {marker!r} is not "
                    "allowed in binding specs"
                )
    return findings


def _walk_security_values(value: object, location: str) -> list[tuple[str, object]]:
    """Recursively walk the spec values for the security scan."""
    if is_dataclass(value) and not isinstance(value, type):
        items: list[tuple[str, object]] = []
        for field in fields(value):
            child = getattr(value, field.name)
            items.extend(_walk_security_values(child, f"{location}.{field.name}"))
        return items
    if isinstance(value, Mapping):
        items = []
        for key, child in value.items():
            key_location = f"{location}.{key}"
            items.append((key_location, key))
            items.extend(_walk_security_values(child, key_location))
        return items
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        items = []
        for idx, child in enumerate(value):
            items.extend(_walk_security_values(child, f"{location}[{idx}]"))
        return items
    return [(location, value)]
