# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "HierarchyLayer",
    "OscillatorFamily",
    "CouplingSpec",
    "DriverSpec",
    "ObjectivePartition",
    "BoundaryDef",
    "ActuatorMapping",
    "ImprintSpec",
    "GeometrySpec",
    "VALID_CHANNELS",
    "VALID_SEVERITIES",
    "VALID_KNOBS",
    "VALID_SAFETY_TIERS",
    "BindingSpec",
]


@dataclass(frozen=True)
class HierarchyLayer:
    name: str
    index: int
    oscillator_ids: list[str]


@dataclass(frozen=True)
class OscillatorFamily:
    channel: str  # "P", "I", or "S"
    extractor_type: str
    config: dict


@dataclass(frozen=True)
class CouplingSpec:
    base_strength: float
    decay_alpha: float
    templates: dict[str, str]


@dataclass(frozen=True)
class DriverSpec:
    physical: dict
    informational: dict
    symbolic: dict


@dataclass(frozen=True)
class ObjectivePartition:
    good_layers: list[int]
    bad_layers: list[int]
    good_weight: float = 1.0
    bad_weight: float = 1.0


@dataclass(frozen=True)
class BoundaryDef:
    name: str
    variable: str
    lower: float | None
    upper: float | None
    severity: str  # "soft" or "hard"


@dataclass(frozen=True)
class ActuatorMapping:
    name: str
    knob: str
    scope: str
    limits: tuple[float, float]


@dataclass(frozen=True)
class ImprintSpec:
    decay_rate: float
    saturation: float
    modulates: list[str]


@dataclass(frozen=True)
class GeometrySpec:
    constraint_type: str
    params: dict


VALID_CHANNELS = frozenset({"P", "I", "S"})
VALID_SEVERITIES = frozenset({"soft", "hard"})
VALID_KNOBS = frozenset({"K", "alpha", "zeta", "Psi"})
VALID_SAFETY_TIERS = frozenset({"research", "clinical", "consumer"})


@dataclass
class BindingSpec:
    name: str
    version: str
    safety_tier: str
    sample_period_s: float
    control_period_s: float
    layers: list[HierarchyLayer]
    oscillator_families: dict[str, OscillatorFamily]
    coupling: CouplingSpec
    drivers: DriverSpec
    objectives: ObjectivePartition
    boundaries: list[BoundaryDef]
    actuators: list[ActuatorMapping]
    imprint_model: ImprintSpec | None = None
    geometry_prior: GeometrySpec | None = None
