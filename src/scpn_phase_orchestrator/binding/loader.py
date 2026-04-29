# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding spec loader

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    AmplitudeSpec,
    BindingSpec,
    BoundaryDef,
    ChannelGroupSpec,
    ChannelSpec,
    CouplingSpec,
    CrossChannelCouplingSpec,
    DriverSpec,
    GeometrySpec,
    HierarchyLayer,
    ImprintSpec,
    ObjectivePartition,
    OscillatorFamily,
    ProtocolNetSpec,
    ProtocolTransitionSpec,
    is_valid_channel_id,
    resolve_extractor_type,
)
from scpn_phase_orchestrator.exceptions import BindingError

__all__ = ["load_binding_spec"]


class BindingLoadError(BindingError):
    """Raised when a binding spec cannot be parsed."""


def _require(data: dict, key: str, context: str = "") -> Any:
    """Extract a required key from *data*, raising BindingLoadError if absent."""
    try:
        return data[key]
    except KeyError:
        loc = f" in {context}" if context else ""
        raise BindingLoadError(f"missing required key {key!r}{loc}") from None


def _expected(kind: str, context: str, value: object) -> BindingLoadError:
    return BindingLoadError(f"expected {kind} in {context}, got {type(value).__name__}")


def _require_mapping(value: object, context: str) -> dict:
    if not isinstance(value, dict):
        raise _expected("mapping", context, value)
    return value


def _optional_mapping(value: object, context: str) -> dict:
    if value is None:
        return {}
    return _require_mapping(value, context)


def _require_list(value: object, context: str) -> list:
    if not isinstance(value, list):
        raise _expected("list", context, value)
    return value


def _optional_list(value: object, context: str) -> list:
    if value is None:
        return []
    return _require_list(value, context)


def _require_str(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise _expected("string", context, value)
    return value


def _require_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _expected("integer", context, value)
    return value


def _require_number(value: object, context: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise _expected("number", context, value)
    return float(value)


def _require_bool(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise _expected("boolean", context, value)
    return value


def _optional_bool(value: object, context: str, default: bool) -> bool:
    if value is None:
        return default
    return _require_bool(value, context)


def _optional_number(value: object, context: str) -> float | None:
    if value is None:
        return None
    return _require_number(value, context)


def _optional_str(value: object, context: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, context)


def _optional_str_list(value: object, context: str) -> list[str]:
    items = _optional_list(value, context)
    return [_require_str(item, f"{context}[]") for item in items]


def _optional_number_list(value: object, context: str) -> list[float] | None:
    if value is None:
        return None
    items = _require_list(value, context)
    return [_require_number(item, f"{context}[]") for item in items]


def _require_number_list(value: object, context: str) -> list[float]:
    items = _require_list(value, context)
    return [_require_number(item, f"{context}[]") for item in items]


def _require_number_pair(value: object, context: str) -> tuple[float, float]:
    items = _require_number_list(value, context)
    if len(items) != 2:
        raise BindingLoadError(
            f"expected two numbers in {context}, got {len(items)} item(s)"
        )
    return (items[0], items[1])


def _load_drivers(data: dict) -> DriverSpec:
    """Load standard and named driver channel configuration maps."""
    drivers_data = _require_mapping(_require(data, "drivers", "root"), "drivers")
    standard_driver_keys = {"physical", "informational", "symbolic"}
    driver_maps: dict[str, dict] = {}
    for key, value in drivers_data.items():
        driver_key = _require_str(key, "drivers key")
        if driver_key not in standard_driver_keys and not is_valid_channel_id(
            driver_key
        ):
            raise BindingLoadError(
                f"drivers.{driver_key}: invalid driver channel identifier"
            )
        driver_maps[driver_key] = _require_mapping(value, f"drivers.{driver_key}")

    extra_drivers = {
        key: value
        for key, value in driver_maps.items()
        if key not in standard_driver_keys
    }
    return DriverSpec(
        physical=driver_maps.get("physical", {}),
        informational=driver_maps.get("informational", {}),
        symbolic=driver_maps.get("symbolic", {}),
        extra=extra_drivers,
    )


def _load_channels(data: dict) -> dict[str, ChannelSpec]:
    """Load optional typed N-channel declarations."""
    channels_data = _optional_mapping(data.get("channels"), "channels")
    channels: dict[str, ChannelSpec] = {}
    for raw_name, raw_channel in channels_data.items():
        name = _require_str(raw_name, "channels key")
        if not is_valid_channel_id(name):
            raise BindingLoadError(f"channels.{name}: invalid channel identifier")
        channel_data = _require_mapping(raw_channel, f"channels.{name}")
        channels[name] = ChannelSpec(
            role=_require_str(
                channel_data.get("role", "domain"), f"channels.{name}.role"
            ),
            required=_optional_bool(
                channel_data.get("required"), f"channels.{name}.required", True
            ),
            units=_optional_str(channel_data.get("units"), f"channels.{name}.units"),
            metric_semantics=_optional_str(
                channel_data.get("metric_semantics"),
                f"channels.{name}.metric_semantics",
            ),
            coupling_participation=_optional_bool(
                channel_data.get("coupling_participation"),
                f"channels.{name}.coupling_participation",
                True,
            ),
            audit_serialisation=_optional_bool(
                channel_data.get("audit_serialisation"),
                f"channels.{name}.audit_serialisation",
                True,
            ),
            replay_semantics=_require_str(
                channel_data.get("replay_semantics", "phase"),
                f"channels.{name}.replay_semantics",
            ),
            supervisor_visibility=_optional_bool(
                channel_data.get("supervisor_visibility"),
                f"channels.{name}.supervisor_visibility",
                True,
            ),
            derived_from=_optional_str_list(
                channel_data.get("derived_from"), f"channels.{name}.derived_from"
            ),
            derive_rule=_optional_str(
                channel_data.get("derive_rule"), f"channels.{name}.derive_rule"
            ),
        )
    return channels


def _load_channel_groups(data: dict) -> dict[str, ChannelGroupSpec]:
    """Load optional N-channel group declarations."""
    groups_data = _optional_mapping(data.get("channel_groups"), "channel_groups")
    groups: dict[str, ChannelGroupSpec] = {}
    for raw_name, raw_group in groups_data.items():
        name = _require_str(raw_name, "channel_groups key")
        if not is_valid_channel_id(name):
            raise BindingLoadError(
                f"channel_groups.{name}: invalid channel group identifier"
            )
        group_data = _require_mapping(raw_group, f"channel_groups.{name}")
        groups[name] = ChannelGroupSpec(
            channels=_optional_str_list(
                _require(group_data, "channels", f"channel_groups.{name}"),
                f"channel_groups.{name}.channels",
            ),
            required=_optional_bool(
                group_data.get("required"), f"channel_groups.{name}.required", True
            ),
            description=_optional_str(
                group_data.get("description"), f"channel_groups.{name}.description"
            ),
        )
    return groups


def _load_cross_channel_couplings(data: dict) -> list[CrossChannelCouplingSpec]:
    """Load optional cross-channel coupling declarations."""
    couplings = []
    for i, raw_coupling in enumerate(
        _optional_list(data.get("cross_channel_couplings"), "cross_channel_couplings")
    ):
        c = _require_mapping(raw_coupling, f"cross_channel_couplings[{i}]")
        couplings.append(
            CrossChannelCouplingSpec(
                source=_require_str(
                    _require(c, "source", "cross_channel_couplings[]"),
                    "cross_channel_couplings[].source",
                ),
                target=_require_str(
                    _require(c, "target", "cross_channel_couplings[]"),
                    "cross_channel_couplings[].target",
                ),
                strength=_require_number(
                    _require(c, "strength", "cross_channel_couplings[]"),
                    "cross_channel_couplings[].strength",
                ),
                mode=_require_str(
                    c.get("mode", "bidirectional"),
                    "cross_channel_couplings[].mode",
                ),
                template=_optional_str(
                    c.get("template"), "cross_channel_couplings[].template"
                ),
            )
        )
    return couplings


def load_binding_spec(path: str | Path) -> BindingSpec:
    """Load a BindingSpec from a YAML or JSON file."""
    path = Path(path)
    # Filename only in surfaced error messages — full filesystem paths must
    # not leak into logs or API clients.
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, IsADirectoryError) as exc:
        reason = exc.strerror or type(exc).__name__
        raise BindingLoadError(f"cannot read {path.name}: {reason}") from exc

    if path.suffix in (".yaml", ".yml"):
        import yaml

        try:
            data = yaml.safe_load(raw)
        except (RecursionError, yaml.YAMLError) as exc:
            raise BindingLoadError(f"YAML parse error in {path.name}: {exc}") from exc
    elif path.suffix == ".json":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise BindingLoadError(f"JSON parse error in {path.name}: {exc}") from exc
    else:
        raise BindingLoadError(f"Unsupported file extension: {path.suffix}")

    if not isinstance(data, dict):
        raise BindingLoadError(
            f"expected mapping at top level, got {type(data).__name__}"
        )

    layers_data = _require_list(_require(data, "layers", "root"), "layers")
    layers = []
    for i, raw_layer in enumerate(layers_data):
        lay = _require_mapping(raw_layer, f"layers[{i}]")
        layers.append(
            HierarchyLayer(
                name=_require_str(_require(lay, "name", "layers[]"), "layers[].name"),
                index=_require_int(
                    _require(lay, "index", "layers[]"), "layers[].index"
                ),
                oscillator_ids=_optional_str_list(
                    lay.get("oscillator_ids"), "layers[].oscillator_ids"
                ),
                omegas=_optional_number_list(lay.get("omegas"), "layers[].omegas"),
                family=_optional_str(lay.get("family"), "layers[].family"),
            )
        )

    families_data = _require_mapping(
        _require(data, "oscillator_families", "root"), "oscillator_families"
    )
    osc_families = {}
    for key, raw_family in families_data.items():
        family_name = _require_str(key, "oscillator_families key")
        family_data = _require_mapping(raw_family, f"oscillator_families.{family_name}")
        extractor_type = _require_str(
            _require(
                family_data, "extractor_type", f"oscillator_families.{family_name}"
            ),
            f"oscillator_families.{family_name}.extractor_type",
        )
        osc_families[family_name] = OscillatorFamily(
            channel=_require_str(
                _require(family_data, "channel", f"oscillator_families.{family_name}"),
                f"oscillator_families.{family_name}.channel",
            ),
            extractor_type=resolve_extractor_type(extractor_type),
            config=_optional_mapping(
                family_data.get("config"), f"oscillator_families.{family_name}.config"
            ),
        )

    coupling_data = _require_mapping(_require(data, "coupling", "root"), "coupling")
    coupling = CouplingSpec(
        base_strength=_require_number(
            _require(coupling_data, "base_strength", "coupling"),
            "coupling.base_strength",
        ),
        decay_alpha=_require_number(
            _require(coupling_data, "decay_alpha", "coupling"),
            "coupling.decay_alpha",
        ),
        templates=_optional_mapping(
            coupling_data.get("templates"), "coupling.templates"
        ),
    )

    drivers = _load_drivers(data)
    channels = _load_channels(data)
    channel_groups = _load_channel_groups(data)
    cross_channel_couplings = _load_cross_channel_couplings(data)

    obj = _require_mapping(_require(data, "objectives", "root"), "objectives")
    objectives = ObjectivePartition(
        good_layers=_require_list(
            _require(obj, "good_layers", "objectives"), "objectives.good_layers"
        ),
        bad_layers=_require_list(
            _require(obj, "bad_layers", "objectives"), "objectives.bad_layers"
        ),
        good_weight=_require_number(
            obj.get("good_weight", 1.0), "objectives.good_weight"
        ),
        bad_weight=_require_number(obj.get("bad_weight", 1.0), "objectives.bad_weight"),
    )

    boundaries = []
    for i, raw_boundary in enumerate(
        _optional_list(data.get("boundaries"), "boundaries")
    ):
        b = _require_mapping(raw_boundary, f"boundaries[{i}]")
        boundaries.append(
            BoundaryDef(
                name=_require_str(
                    _require(b, "name", "boundaries[]"), "boundaries[].name"
                ),
                variable=_require_str(
                    _require(b, "variable", "boundaries[]"), "boundaries[].variable"
                ),
                lower=_optional_number(b.get("lower"), "boundaries[].lower"),
                upper=_optional_number(b.get("upper"), "boundaries[].upper"),
                severity=_require_str(
                    _require(b, "severity", "boundaries[]"), "boundaries[].severity"
                ),
            )
        )

    actuators = []
    for i, raw_actuator in enumerate(
        _optional_list(data.get("actuators"), "actuators")
    ):
        a = _require_mapping(raw_actuator, f"actuators[{i}]")
        actuators.append(
            ActuatorMapping(
                name=_require_str(
                    _require(a, "name", "actuators[]"), "actuators[].name"
                ),
                knob=_require_str(
                    _require(a, "knob", "actuators[]"), "actuators[].knob"
                ),
                scope=_require_str(
                    _require(a, "scope", "actuators[]"), "actuators[].scope"
                ),
                limits=_require_number_pair(
                    _require(a, "limits", "actuators[]"), "actuators[].limits"
                ),
            )
        )

    imprint_data = data.get("imprint_model") or data.get("imprint")
    imprint = None
    if imprint_data:
        imprint_map = _require_mapping(imprint_data, "imprint_model")
        imprint = ImprintSpec(
            decay_rate=_require_number(
                _require(imprint_map, "decay_rate", "imprint_model"),
                "imprint_model.decay_rate",
            ),
            saturation=_require_number(
                _require(imprint_map, "saturation", "imprint_model"),
                "imprint_model.saturation",
            ),
            modulates=_optional_list(
                imprint_map.get("modulates"), "imprint_model.modulates"
            ),
        )

    geo_data = data.get("geometry_prior")
    geometry = None
    if geo_data:
        geo_map = _require_mapping(geo_data, "geometry_prior")
        geometry = GeometrySpec(
            constraint_type=_require_str(
                _require(geo_map, "constraint_type", "geometry_prior"),
                "geometry_prior.constraint_type",
            ),
            params=_optional_mapping(geo_map.get("params"), "geometry_prior.params"),
        )

    pnet_data = data.get("protocol_net")
    protocol_net = None
    if pnet_data:
        pnet_map = _require_mapping(pnet_data, "protocol_net")
        pnet_transitions = []
        for i, raw_transition in enumerate(
            _require_list(
                _require(pnet_map, "transitions", "protocol_net"),
                "protocol_net.transitions",
            )
        ):
            t = _require_mapping(raw_transition, f"protocol_net.transitions[{i}]")
            pnet_transitions.append(
                ProtocolTransitionSpec(
                    name=_require_str(
                        _require(t, "name", "protocol_net.transitions[]"),
                        "protocol_net.transitions[].name",
                    ),
                    inputs=_optional_list(
                        t.get("inputs"), "protocol_net.transitions[].inputs"
                    ),
                    outputs=_optional_list(
                        t.get("outputs"), "protocol_net.transitions[].outputs"
                    ),
                    guard=t.get("guard"),
                )
            )
        protocol_net = ProtocolNetSpec(
            places=_require_list(
                _require(pnet_map, "places", "protocol_net"), "protocol_net.places"
            ),
            initial=_require_mapping(
                _require(pnet_map, "initial", "protocol_net"), "protocol_net.initial"
            ),
            place_regime=_optional_mapping(
                pnet_map.get("place_regime"), "protocol_net.place_regime"
            ),
            transitions=pnet_transitions,
        )

    amp_data = data.get("amplitude")
    amplitude = None
    if amp_data:
        amp_map = _require_mapping(amp_data, "amplitude")
        amplitude = AmplitudeSpec(
            mu=_require_number(_require(amp_map, "mu", "amplitude"), "amplitude.mu"),
            epsilon=_require_number(
                _require(amp_map, "epsilon", "amplitude"), "amplitude.epsilon"
            ),
            amp_coupling_strength=_require_number(
                amp_map.get("amp_coupling_strength", 0.0),
                "amplitude.amp_coupling_strength",
            ),
            amp_coupling_decay=_require_number(
                amp_map.get("amp_coupling_decay", 0.3),
                "amplitude.amp_coupling_decay",
            ),
        )

    return BindingSpec(
        name=_require_str(_require(data, "name", "root"), "name"),
        version=_require_str(_require(data, "version", "root"), "version"),
        safety_tier=_require_str(_require(data, "safety_tier", "root"), "safety_tier"),
        sample_period_s=_require_number(
            _require(data, "sample_period_s", "root"), "sample_period_s"
        ),
        control_period_s=_require_number(
            _require(data, "control_period_s", "root"), "control_period_s"
        ),
        layers=layers,
        oscillator_families=osc_families,
        coupling=coupling,
        drivers=drivers,
        objectives=objectives,
        boundaries=boundaries,
        actuators=actuators,
        imprint_model=imprint,
        geometry_prior=geometry,
        protocol_net=protocol_net,
        amplitude=amplitude,
        channels=channels,
        channel_groups=channel_groups,
        cross_channel_couplings=cross_channel_couplings,
    )
