# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Binding type definitions

"""Typed dataclass model for SPO domain binding specifications.

These dataclasses are the in-memory contract produced by the YAML loader and
consumed by validators, CLIs, engines, supervisors, audit summaries, and
digital-twin exporters. Constructors keep lightweight invariant checks where
local consistency is unambiguous; cross-field and deployment-policy checks live
in `binding.validator` so error reporting can stay complete and actionable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

from scpn_phase_orchestrator.exceptions import ValidationError

__all__ = [
    "HierarchyLayer",
    "OscillatorFamily",
    "CouplingSpec",
    "DriverSpec",
    "ChannelSpec",
    "ChannelGroupSpec",
    "CrossChannelCouplingSpec",
    "ObjectivePartition",
    "BoundaryDef",
    "ActuatorMapping",
    "ImprintSpec",
    "GeometrySpec",
    "ProtocolNetSpec",
    "AmplitudeSpec",
    "VALID_CHANNELS",
    "STANDARD_CHANNELS",
    "VALID_SEVERITIES",
    "VALID_KNOBS",
    "VALID_SAFETY_TIERS",
    "VALID_VALIDATION_TIERS",
    "VALIDATION_TIER_SCAFFOLD",
    "VALIDATION_TIER_PARTIAL",
    "VALIDATION_TIER_EXTERNALLY_VALIDATED",
    "DEFAULT_VALIDATION_TIER",
    "EXTRACTOR_ALIASES",
    "VALID_EXTRACTORS",
    "is_valid_channel_id",
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
    """Phase extraction configuration for one oscillator group."""

    channel: str
    extractor_type: str
    config: dict[str, Any]


@dataclass(frozen=True)
class CouplingSpec:
    """Parameters for K_nm coupling matrix construction."""

    base_strength: float
    decay_alpha: float
    templates: dict[str, str]


@dataclass(frozen=True)
class DriverSpec:
    """Configuration for standard and named external driver channels."""

    physical: dict[str, Any]
    informational: dict[str, Any]
    symbolic: dict[str, Any]
    extra: dict[str, dict[str, Any]] | None = None

    def channel_config(self, channel: str) -> dict[str, Any]:
        """Return driver config for a standard or named channel.

        Parameters
        ----------
        channel : str
            A standard channel (``P``/``physical``, ``I``/``informational``,
            ``S``/``symbolic``) or a named extension channel id.

        Returns
        -------
        dict[str, Any]
            The driver configuration mapping for *channel*, or an empty mapping
            when the channel has no configured driver.
        """
        standard = {
            "P": self.physical,
            "physical": self.physical,
            "I": self.informational,
            "informational": self.informational,
            "S": self.symbolic,
            "symbolic": self.symbolic,
        }
        if channel in standard:
            return standard[channel]
        return (self.extra or {}).get(channel, {})

    def all_channel_configs(self) -> dict[str, dict[str, Any]]:
        """Return standard driver configs plus named extension channels.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of channel id (``P``/``I``/``S`` plus any named extension
            channels) to its driver configuration mapping.
        """
        configs = {
            "P": self.physical,
            "I": self.informational,
            "S": self.symbolic,
        }
        configs.update(self.extra or {})
        return configs


@dataclass(frozen=True)
class ChannelSpec:
    """Typed binding channel metadata for N-channel domainpacks."""

    role: str
    required: bool = True
    units: str | None = None
    metric_semantics: str | None = None
    coupling_participation: bool = True
    audit_serialisation: bool = True
    replay_semantics: str = "phase"
    supervisor_visibility: bool = True
    derived_from: list[str] = field(default_factory=list)
    derive_rule: str | None = None


@dataclass(frozen=True)
class ChannelGroupSpec:
    """Named set of channels used for validation and supervisor summaries."""

    channels: list[str]
    required: bool = True
    description: str | None = None


@dataclass(frozen=True)
class CrossChannelCouplingSpec:
    """Declared coupling relation between two binding channels."""

    source: str
    target: str
    strength: float
    mode: str = "bidirectional"
    template: str | None = None


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
    rate_limit_per_step: float | None = None

    def __post_init__(self) -> None:
        lo, hi = self.limits
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in self.limits
        ):
            raise TypeError(
                f"ActuatorMapping {self.name!r}: limits must be finite reals"
            )
        if not isfinite(float(lo)) or not isfinite(float(hi)):
            raise ValueError(
                f"ActuatorMapping {self.name!r}: limits must be finite reals"
            )
        if float(lo) > float(hi):
            raise ValueError(
                f"ActuatorMapping {self.name!r}: limits require lower <= upper"
            )
        if self.rate_limit_per_step is None:
            return
        if isinstance(self.rate_limit_per_step, bool) or not isinstance(
            self.rate_limit_per_step, Real
        ):
            raise TypeError(
                f"ActuatorMapping {self.name!r}: rate_limit_per_step must be a "
                "finite non-negative real"
            )
        if (
            not isfinite(float(self.rate_limit_per_step))
            or float(self.rate_limit_per_step) < 0.0
        ):
            raise ValueError(
                f"ActuatorMapping {self.name!r}: rate_limit_per_step must be "
                "finite and non-negative"
            )


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
    params: dict[str, Any]


@dataclass(frozen=True)
class ProtocolTransitionSpec:
    """One transition in the Petri net protocol specification."""

    name: str
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
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


STANDARD_CHANNELS = frozenset({"P", "I", "S"})
VALID_CHANNELS = STANDARD_CHANNELS
_CHANNEL_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
VALID_SEVERITIES = frozenset({"soft", "hard"})
VALID_KNOBS = frozenset({"K", "alpha", "zeta", "Psi"})
VALID_SAFETY_TIERS = frozenset({"research", "clinical", "consumer", "production"})

# Domainpack validation posture — how much external evidence a binding scaffold
# carries, distinct from ``safety_tier`` (the deployment risk class). Grounded in
# the README §Evidence status: a binding is a reusable scaffold, not a validated
# detector, so ``scaffold`` is the honest default. A pack is promoted to
# ``partial`` (some validation evidence, not fully external) or
# ``externally_validated`` (clears an independent-reference test on real data)
# only with a citable evidence trail.
VALIDATION_TIER_SCAFFOLD = "scaffold"
VALIDATION_TIER_PARTIAL = "partial"
VALIDATION_TIER_EXTERNALLY_VALIDATED = "externally_validated"
VALID_VALIDATION_TIERS = frozenset(
    {
        VALIDATION_TIER_SCAFFOLD,
        VALIDATION_TIER_PARTIAL,
        VALIDATION_TIER_EXTERNALLY_VALIDATED,
    }
)
DEFAULT_VALIDATION_TIER = VALIDATION_TIER_SCAFFOLD

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


def is_valid_channel_id(channel: str) -> bool:
    """Return True when *channel* is a valid binding channel identifier.

    Parameters
    ----------
    channel : str
        Candidate channel identifier to validate.

    Returns
    -------
    bool
        ``True`` if *channel* matches the binding channel-id grammar.
    """
    return bool(_CHANNEL_ID_RE.fullmatch(channel))


def resolve_extractor_type(raw: str) -> str:
    """Map alias to algorithm name; pass algorithm names through unchanged.

    Parameters
    ----------
    raw : str
        An extractor alias or canonical algorithm name.

    Returns
    -------
    str
        The canonical extractor algorithm name; unknown values pass through
        unchanged.
    """
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
    #: Honest external-validation posture of this binding scaffold, one of
    #: :data:`VALID_VALIDATION_TIERS`. Defaults to
    #: :data:`DEFAULT_VALIDATION_TIER` (``scaffold``) so an undeclared spec is
    #: treated as unvalidated rather than silently trusted.
    validation_tier: str = DEFAULT_VALIDATION_TIER
    imprint_model: ImprintSpec | None = None
    geometry_prior: GeometrySpec | None = None
    protocol_net: ProtocolNetSpec | None = None
    amplitude: AmplitudeSpec | None = None
    channels: dict[str, ChannelSpec] = field(default_factory=dict)
    channel_groups: dict[str, ChannelGroupSpec] = field(default_factory=dict)
    cross_channel_couplings: list[CrossChannelCouplingSpec] = field(
        default_factory=list
    )
    value_alignment: dict[str, Any] = field(default_factory=dict)

    def get_omegas(self) -> list[float]:
        """Collect natural frequencies from all layers.

        Falls back to 1.0 rad/s per oscillator when a layer defines no omegas.

        Returns
        -------
        list[float]
            Natural frequencies in rad/s, concatenated in layer order, one per
            oscillator.

        Raises
        ------
        ValueError
            If a layer defines an ``omegas`` list whose length differs from its
            oscillator count.
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

    def used_channels(self) -> set[str]:
        """Return channels referenced by families, drivers, and algebra.

        Returns
        -------
        set[str]
            The set of channel identifiers referenced by oscillator families,
            configured drivers, and cross-channel coupling declarations.
        """
        used = {family.channel for family in self.oscillator_families.values()}
        used.update(self.drivers.all_channel_configs())
        used.update(self.channels)
        for channel in self.channels.values():
            used.update(channel.derived_from)
        for group in self.channel_groups.values():
            used.update(group.channels)
        for coupling in self.cross_channel_couplings:
            used.add(coupling.source)
            used.add(coupling.target)
        return used
