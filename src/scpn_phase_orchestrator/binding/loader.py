# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import json
from pathlib import Path

from scpn_phase_orchestrator.binding.types import (
    ActuatorMapping,
    BindingSpec,
    BoundaryDef,
    CouplingSpec,
    DriverSpec,
    GeometrySpec,
    HierarchyLayer,
    ImprintSpec,
    ObjectivePartition,
    OscillatorFamily,
)

__all__ = ["load_binding_spec"]


def load_binding_spec(path: str | Path) -> BindingSpec:
    """Load a BindingSpec from a YAML or JSON file."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(raw)
    elif path.suffix == ".json":
        data = json.loads(raw)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    layers = [
        HierarchyLayer(
            name=lay["name"],
            index=lay["index"],
            oscillator_ids=lay.get("oscillator_ids", []),
        )
        for lay in data["layers"]
    ]

    osc_families = {
        k: OscillatorFamily(
            channel=v["channel"],
            extractor_type=v["extractor_type"],
            config=v.get("config", {}),
        )
        for k, v in data["oscillator_families"].items()
    }

    coupling = CouplingSpec(
        base_strength=data["coupling"]["base_strength"],
        decay_alpha=data["coupling"]["decay_alpha"],
        templates=data["coupling"].get("templates", {}),
    )

    drivers = DriverSpec(
        physical=data["drivers"].get("physical", {}),
        informational=data["drivers"].get("informational", {}),
        symbolic=data["drivers"].get("symbolic", {}),
    )

    obj = data["objectives"]
    objectives = ObjectivePartition(
        good_layers=obj["good_layers"],
        bad_layers=obj["bad_layers"],
        good_weight=obj.get("good_weight", 1.0),
        bad_weight=obj.get("bad_weight", 1.0),
    )

    boundaries = [
        BoundaryDef(
            name=b["name"],
            variable=b["variable"],
            lower=b.get("lower"),
            upper=b.get("upper"),
            severity=b["severity"],
        )
        for b in data.get("boundaries", [])
    ]

    actuators = [
        ActuatorMapping(
            name=a["name"],
            knob=a["knob"],
            scope=a["scope"],
            limits=tuple(a["limits"]),
        )
        for a in data.get("actuators", [])
    ]

    imprint_data = data.get("imprint_model")
    imprint = None
    if imprint_data:
        imprint = ImprintSpec(
            decay_rate=imprint_data["decay_rate"],
            saturation=imprint_data["saturation"],
            modulates=imprint_data.get("modulates", []),
        )

    geo_data = data.get("geometry_prior")
    geometry = None
    if geo_data:
        geometry = GeometrySpec(
            constraint_type=geo_data["constraint_type"],
            params=geo_data.get("params", {}),
        )

    return BindingSpec(
        name=data["name"],
        version=data["version"],
        safety_tier=data["safety_tier"],
        sample_period_s=data["sample_period_s"],
        control_period_s=data["control_period_s"],
        layers=layers,
        oscillator_families=osc_families,
        coupling=coupling,
        drivers=drivers,
        objectives=objectives,
        boundaries=boundaries,
        actuators=actuators,
        imprint_model=imprint,
        geometry_prior=geometry,
    )
