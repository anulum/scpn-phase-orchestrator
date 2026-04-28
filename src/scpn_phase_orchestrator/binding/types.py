# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding type definitions

from __future__ import annotations

from dataclasses import dataclass

from scpn_phase_orchestrator.exceptions import ValidationError

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
    "ProtocolNetSpec",
    "AmplitudeSpec",
    "VALID_CHANNELS",
    "VALID_SEVERITIES",
    "VALID_KNOBS",
    "VALID_SAFETY_TIERS",
    "EXTRACTOR_ALIASES",
    "VALID_EXTRACTORS",
    "resolve_extractor_type",
    "BindingSpec",
]


@dataclass(frozen=True)
class HierarchyLayer:
    """Single layer in the SCPN oscillator hierarchy."""

    name: str
    index: int
    oscillator_ids: list[str]
    omegas: list[float] | None = None  # natural frequencies (rad/s) per oscillator
    family: str | None = None  # maps to oscillator_families key


@dataclass(frozen=True)
class OscillatorFamily:
    """Phase extraction configuration for one oscillator group (P/I/S channel)."""

    channel: str  # "P", "I", or "S"
    extractor_type: str
    config: dict


@dataclass(frozen=True)
class CouplingSpec:
    """Parameters for K_nm coupling matrix construction."""

    base_strength: float
    decay_alpha: float
    templates: dict[str, str]


@dataclass(frozen=True)
class DriverSpec:
    """Configuration for P/I/S external driver channels."""

    physical: dict
    informational: dict
    symbolic: dict


@dataclass(frozen=True)
class ObjectivePartition:
    """Partition of layers into good (synchronise) and bad (desynchronise) subsets."""

    good_layers: list[int]
    bad_layers: list[int]
    good_weight: float = 1.0
    bad_weight: float = 1.0


@dataclass(frozen=True)
class BoundaryDef:
    """Defines a soft or hard boundary on a monitored variable."""

    name: str
    variable: str
    lower: float | None
    upper: float | None
    severity: str  # "soft" or "hard"

    def __post_init__(self) -> None:
        if (
            self.lower is not None
            and self.upper is not None
            and self.lower >= self.upper
        ):
            msg = (
                f"BoundaryDef {self.name!r}: "
                f"lower ({self.lower}) must be < upper ({self.upper})"
            )
            raise ValidationError(msg)


@dataclass(frozen=True)
class ActuatorMapping:
    """Maps a control knob to a named actuator with scope and limits."""

    name: str
    knob: str
    scope: str
    limits: tuple[float, float]


@dataclass(frozen=True)
class ImprintSpec:
    """Parameters for the L9 memory imprint model."""

    decay_rate: float
    saturation: float
    modulates: list[str]


@dataclass(frozen=True)
class GeometrySpec:
    """Geometry constraint type and parameters for K_nm projection."""

    constraint_type: str
    params: dict


@dataclass(frozen=True)
class ProtocolTransitionSpec:
    """One transition in the Petri net protocol specification."""

    name: str
    inputs: list[dict]
    outputs: list[dict]
    guard: str | None = None


@dataclass(frozen=True)
class ProtocolNetSpec:
    """Full Petri net specification: places, initial marking, and transitions."""

    places: list[str]
    initial: dict[str, int]
    place_regime: dict[str, str]
    transitions: list[ProtocolTransitionSpec]


@dataclass(frozen=True)
class AmplitudeSpec:
    """Amplitude dynamics parameters (Stuart-Landau bifurcation)."""

    mu: float
    epsilon: float
    amp_coupling_strength: float = 0.0
    amp_coupling_decay: float = 0.3


VALID_CHANNELS = frozenset({"P", "I", "S"})
VALID_SEVERITIES = frozenset({"soft", "hard"})
VALID_KNOBS = frozenset({"K", "alpha", "zeta", "Psi"})
VALID_SAFETY_TIERS = frozenset({"research", "clinical", "consumer", "production"})

# Algorithm-level extractor types
_ALGORITHM_EXTRACTORS = frozenset(
    {
        "hilbert",
        "wavelet",
        "zero_crossing",
        "event",
        "ring",
        "graph",
    }
)
# Channel-level aliases used in domainpacks → default algorithm
EXTRACTOR_ALIASES: dict[str, str] = {
    "physical": "hilbert",
    "informational": "event",
    "symbolic": "ring",
}
VALID_EXTRACTORS = _ALGORITHM_EXTRACTORS | frozenset(EXTRACTOR_ALIASES)


def resolve_extractor_type(raw: str) -> str:
    """Map alias to algorithm name; pass algorithm names through unchanged."""
    return EXTRACTOR_ALIASES.get(raw, raw)


@dataclass
class BindingSpec:
    """Complete domainpack binding: layers, coupling, drivers, and actuators."""

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
    protocol_net: ProtocolNetSpec | None = None
    amplitude: AmplitudeSpec | None = None

    def get_omegas(self) -> list[float]:
        """Collect natural frequencies from all layers.

        Falls back to 1.0 rad/s per oscillator if omegas is not defined.
        Raises ValueError if omegas is defined but length mismatches.
        """
        result: list[float] = []
        for layer in self.layers:
            n = len(layer.oscillator_ids)
            if layer.omegas is not None:
                if len(layer.omegas) != n:
                    msg = (
                        f"Layer {layer.name!r}: omegas length {len(layer.omegas)}"
                        f" != oscillator count {n}"
                    )
                    raise ValueError(msg)
                result.extend(layer.omegas)
            else:
                result.extend([1.0] * n)
        return result
