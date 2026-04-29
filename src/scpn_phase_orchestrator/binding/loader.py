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
    CouplingSpec,
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


def _require_mapping(value: Any, context: str) -> dict:
    """Return *value* as a mapping or raise a location-specific load error."""
    if not isinstance(value, dict):
        raise BindingLoadError(
            f"{context} must be a mapping, got {type(value).__name__}"
        )
    return value


def _load_drivers(data: dict) -> DriverSpec:
    """Load standard and named driver channel configuration maps."""
    drivers_data = _require_mapping(_require(data, "drivers", "root"), "drivers")
    standard_driver_keys = {"physical", "informational", "symbolic"}
    driver_maps: dict[str, dict] = {}
    for key, value in drivers_data.items():
        if key not in standard_driver_keys and not is_valid_channel_id(key):
            raise BindingLoadError(f"drivers.{key}: invalid driver channel identifier")
        driver_maps[key] = _require_mapping(value, f"drivers.{key}")

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

    layers = [
        HierarchyLayer(
            name=_require(lay, "name", "layers[]"),
            index=_require(lay, "index", "layers[]"),
            oscillator_ids=lay.get("oscillator_ids", []),
            omegas=lay.get("omegas"),
            family=lay.get("family"),
        )
        for lay in _require(data, "layers", "root")
    ]

    osc_families = {
        k: OscillatorFamily(
            channel=_require(v, "channel", f"oscillator_families.{k}"),
            extractor_type=resolve_extractor_type(
                _require(v, "extractor_type", f"oscillator_families.{k}")
            ),
            config=v.get("config", {}),
        )
        for k, v in _require(data, "oscillator_families", "root").items()
    }

    coupling_data = _require(data, "coupling", "root")
    coupling = CouplingSpec(
        base_strength=_require(coupling_data, "base_strength", "coupling"),
        decay_alpha=_require(coupling_data, "decay_alpha", "coupling"),
        templates=coupling_data.get("templates", {}),
    )

    drivers = _load_drivers(data)

    obj = _require(data, "objectives", "root")
    objectives = ObjectivePartition(
        good_layers=_require(obj, "good_layers", "objectives"),
        bad_layers=_require(obj, "bad_layers", "objectives"),
        good_weight=obj.get("good_weight", 1.0),
        bad_weight=obj.get("bad_weight", 1.0),
    )

    boundaries = [
        BoundaryDef(
            name=_require(b, "name", "boundaries[]"),
            variable=_require(b, "variable", "boundaries[]"),
            lower=b.get("lower"),
            upper=b.get("upper"),
            severity=_require(b, "severity", "boundaries[]"),
        )
        for b in data.get("boundaries", [])
    ]

    actuators = [
        ActuatorMapping(
            name=_require(a, "name", "actuators[]"),
            knob=_require(a, "knob", "actuators[]"),
            scope=_require(a, "scope", "actuators[]"),
            limits=tuple(_require(a, "limits", "actuators[]")),
        )
        for a in data.get("actuators", [])
    ]

    imprint_data = data.get("imprint_model") or data.get("imprint")
    imprint = None
    if imprint_data:
        imprint = ImprintSpec(
            decay_rate=_require(imprint_data, "decay_rate", "imprint_model"),
            saturation=_require(imprint_data, "saturation", "imprint_model"),
            modulates=imprint_data.get("modulates", []),
        )

    geo_data = data.get("geometry_prior")
    geometry = None
    if geo_data:
        geometry = GeometrySpec(
            constraint_type=_require(geo_data, "constraint_type", "geometry_prior"),
            params=geo_data.get("params", {}),
        )

    pnet_data = data.get("protocol_net")
    protocol_net = None
    if pnet_data:
        pnet_transitions = []
        for t in _require(pnet_data, "transitions", "protocol_net"):
            pnet_transitions.append(
                ProtocolTransitionSpec(
                    name=_require(t, "name", "protocol_net.transitions[]"),
                    inputs=t.get("inputs", []),
                    outputs=t.get("outputs", []),
                    guard=t.get("guard"),
                )
            )
        protocol_net = ProtocolNetSpec(
            places=_require(pnet_data, "places", "protocol_net"),
            initial=_require(pnet_data, "initial", "protocol_net"),
            place_regime=pnet_data.get("place_regime", {}),
            transitions=pnet_transitions,
        )

    amp_data = data.get("amplitude")
    amplitude = None
    if amp_data:
        amplitude = AmplitudeSpec(
            mu=_require(amp_data, "mu", "amplitude"),
            epsilon=_require(amp_data, "epsilon", "amplitude"),
            amp_coupling_strength=amp_data.get("amp_coupling_strength", 0.0),
            amp_coupling_decay=amp_data.get("amp_coupling_decay", 0.3),
        )

    return BindingSpec(
        name=_require(data, "name", "root"),
        version=_require(data, "version", "root"),
        safety_tier=_require(data, "safety_tier", "root"),
        sample_period_s=_require(data, "sample_period_s", "root"),
        control_period_s=_require(data, "control_period_s", "root"),
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
    )
