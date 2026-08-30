# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic-profile registry

"""Versioned ownership and verified-ingress profiles for reactor semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .registry import DEFAULT_REACTOR_REGISTRY
from .vocabulary import require_enum, require_identifier, require_semver, require_sha256

REACTOR_SEMANTIC_PROFILE_REGISTRY_VERSION = "1.0.0"
REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256 = (
    "fee9bbece259f26d7a1792f1072d5c92249366f1cecf377061b6fcb68db4e730"
)
REACTOR_FAMILY_ASSIGNMENT_STANDARD = (
    "agentic-shared/SCPN_REACTOR_FAMILY_REPOSITORY_STANDARD.md"
)


class SemanticIngressState(StrEnum):
    """Evidence-backed availability of a producer-to-SPO adapter."""

    VERIFIED_REVIEW_ADAPTER = "verified_review_adapter"
    NOT_DECLARED = "not_declared"


@dataclass(frozen=True, slots=True)
class ReactorSemanticProfile:
    """SPO binding for one reactor configuration.

    ``device_project`` records scientific ownership. Producer and adapter
    fields are populated only when an exercised, versioned ingress exists.
    Absence is explicit and cannot be interpreted as an inherited generic
    reactor adapter.
    """

    configuration: str
    device_project: str
    ingress_state: SemanticIngressState
    producer_project: str | None = None
    source_schema: str | None = None
    adapter_api: str | None = None
    handoff_schema: str | None = None
    semantic_profile: str | None = None
    semantic_profile_version: str | None = None
    control_adapter_contract: str | None = None
    control_intent_profile: str | None = None
    authority: str = "review_only"
    actionable: bool = False
    machine_protection_final_veto: bool = True

    def __post_init__(self) -> None:
        """Validate ownership, ingress completeness, and review authority."""
        require_identifier(self.configuration, field="configuration")
        require_identifier(self.device_project, field="device_project")
        require_enum(
            self.ingress_state,
            SemanticIngressState,
            field="ingress_state",
        )
        if self.configuration not in DEFAULT_REACTOR_REGISTRY.configurations:
            raise ValueError("semantic profile configuration is not built into SPO")
        if self.authority != "review_only" or self.actionable is not False:
            raise ValueError("reactor semantic profiles must remain review-only")
        if self.machine_protection_final_veto is not True:
            raise ValueError("machine protection must retain the final veto")
        ingress_fields = (
            self.producer_project,
            self.source_schema,
            self.adapter_api,
            self.handoff_schema,
            self.semantic_profile,
            self.semantic_profile_version,
        )
        if self.ingress_state is SemanticIngressState.VERIFIED_REVIEW_ADAPTER:
            if any(item is None for item in ingress_fields):
                raise ValueError("verified ingress requires a complete adapter binding")
            require_identifier(self.producer_project, field="producer_project")
            require_identifier(self.source_schema, field="source_schema")
            require_identifier(self.adapter_api, field="adapter_api")
            require_identifier(self.handoff_schema, field="handoff_schema")
            require_identifier(self.semantic_profile, field="semantic_profile")
            require_semver(
                self.semantic_profile_version,
                field="semantic_profile_version",
            )
        elif any(item is not None for item in ingress_fields):
            raise ValueError("undeclared ingress cannot advertise adapter fields")
        if self.control_adapter_contract is not None:
            require_identifier(
                self.control_adapter_contract,
                field="control_adapter_contract",
            )
        if self.control_intent_profile is not None:
            require_identifier(
                self.control_intent_profile,
                field="control_intent_profile",
            )

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic profile record.

        Returns
        -------
        dict[str, object]
            JSON-compatible ownership, ingress, and authority fields.
        """
        return {
            "actionable": self.actionable,
            "adapter_api": self.adapter_api,
            "authority": self.authority,
            "configuration": self.configuration,
            "control_adapter_contract": self.control_adapter_contract,
            "control_intent_profile": self.control_intent_profile,
            "device_project": self.device_project,
            "handoff_schema": self.handoff_schema,
            "ingress_state": self.ingress_state.value,
            "machine_protection_final_veto": self.machine_protection_final_veto,
            "producer_project": self.producer_project,
            "semantic_profile": self.semantic_profile,
            "semantic_profile_version": self.semantic_profile_version,
            "source_schema": self.source_schema,
        }


@dataclass(frozen=True, slots=True)
class ReactorSemanticProfileRegistry:
    """Immutable 32-configuration semantic ownership and ingress registry."""

    version: str
    reactor_registry_version: str
    reactor_registry_digest: str
    assignment_map_sha256: str
    profiles: Mapping[str, ReactorSemanticProfile]

    def __post_init__(self) -> None:
        """Validate complete coverage and exact reactor-registry binding."""
        version = require_semver(self.version, field="profile registry version")
        reactor_version = require_semver(
            self.reactor_registry_version,
            field="reactor registry version",
        )
        reactor_digest = require_sha256(
            self.reactor_registry_digest,
            field="reactor registry digest",
        )
        assignment_digest = require_sha256(
            self.assignment_map_sha256,
            field="assignment map digest",
        )
        profiles = dict(self.profiles)
        expected = set(DEFAULT_REACTOR_REGISTRY.configurations)
        if set(profiles) != expected:
            missing = sorted(expected - set(profiles))
            extra = sorted(set(profiles) - expected)
            raise ValueError(
                "semantic profile registry coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        for configuration, profile in profiles.items():
            if configuration != profile.configuration:
                raise ValueError("profile key must equal its configuration")
        if (
            reactor_version != DEFAULT_REACTOR_REGISTRY.version
            or reactor_digest != DEFAULT_REACTOR_REGISTRY.digest
        ):
            raise ValueError(
                "semantic profiles must bind the exact SPO reactor registry"
            )
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "reactor_registry_version", reactor_version)
        object.__setattr__(self, "reactor_registry_digest", reactor_digest)
        object.__setattr__(self, "assignment_map_sha256", assignment_digest)
        object.__setattr__(self, "profiles", MappingProxyType(profiles))

    def resolve(self, configuration: str) -> ReactorSemanticProfile:
        """Resolve a canonical configuration or registry alias.

        Parameters
        ----------
        configuration : str
            Canonical reactor configuration identifier or registered alias.

        Returns
        -------
        ReactorSemanticProfile
            Semantic ownership and ingress profile for the configuration.
        """
        canonical = DEFAULT_REACTOR_REGISTRY.resolve(configuration).identifier
        return self.profiles[canonical]

    def to_record(self) -> dict[str, object]:
        """Return a canonical JSON-compatible registry record.

        Returns
        -------
        dict[str, object]
            Registry identity and all configuration profiles.
        """
        return {
            "assignment_map_sha256": self.assignment_map_sha256,
            "assignment_standard": REACTOR_FAMILY_ASSIGNMENT_STANDARD,
            "profiles": [
                self.profiles[key].to_record() for key in sorted(self.profiles)
            ],
            "reactor_registry_digest": self.reactor_registry_digest,
            "reactor_registry_version": self.reactor_registry_version,
            "version": self.version,
        }

    @property
    def digest(self) -> str:
        """Return the SHA-256 of the canonical registry record."""
        payload = json.dumps(
            self.to_record(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


_DEVICE_ASSIGNMENTS = {
    "beam_target": "SCPN-BEAM-TARGET-CORE",
    "colliding_beam": "SCPN-BEAM-TARGET-CORE",
    "conventional_tokamak": "SCPN-TOKAMAK-CORE",
    "cusp": "SCPN-MAGNETIC-CUSP-CORE",
    "dense_plasma_focus": "SCPN-DENSE-PLASMA-FOCUS-CORE",
    "field_reversed_configuration": "SCPN-FRC-CORE",
    "frc_compression_mif": "SCPN-MIF-CORE",
    "fusion_fission_hybrid": "SCPN-FUSION-FISSION-HYBRID-CORE",
    "gas_dynamic_mirror": "SCPN-MIRROR-CORE",
    "gridded_iec": "SCPN-IEC-CORE",
    "heliotron": "SCPN-STELLARATOR-CORE",
    "ion_beam_icf": "SCPN-ICF-BEAM-CORE",
    "laser_icf_direct_drive": "SCPN-ICF-LASER-CORE",
    "laser_icf_fast_or_shock_ignition": "SCPN-ICF-LASER-CORE",
    "laser_icf_indirect_drive": "SCPN-ICF-LASER-CORE",
    "levitated_dipole": "SCPN-LEVITATED-DIPOLE-CORE",
    "maglif": "SCPN-MIF-MAGLIF-CORE",
    "mechanical_or_liquid_liner_mif": "SCPN-MIF-LINER-CORE",
    "plasma_jet_mif": "SCPN-MIF-PLASMA-JET-CORE",
    "polywell": "SCPN-IEC-CORE",
    "projectile_or_impact_icf": "SCPN-ICF-IMPACT-CORE",
    "pulsed_electron_beam_icf": "SCPN-ICF-BEAM-CORE",
    "reversed_field_pinch": "SCPN-RFP-CORE",
    "sheared_flow_z_pinch": "SCPN-Z-PINCH-CORE",
    "simple_magnetic_mirror": "SCPN-MIRROR-CORE",
    "spheromak": "SCPN-SPHEROMAK-CORE",
    "spherical_tokamak": "SCPN-TOKAMAK-CORE",
    "stellarator": "SCPN-STELLARATOR-CORE",
    "tandem_mirror": "SCPN-MIRROR-CORE",
    "theta_pinch": "SCPN-THETA-PINCH-CORE",
    "torsatron": "SCPN-STELLARATOR-CORE",
    "z_pinch": "SCPN-Z-PINCH-CORE",
}


def _unallocated_profile(configuration: str, project: str) -> ReactorSemanticProfile:
    """Build a profile with no declared semantic ingress."""
    return ReactorSemanticProfile(
        configuration=configuration,
        device_project=project,
        ingress_state=SemanticIngressState.NOT_DECLARED,
    )


_PROFILES = {
    configuration: _unallocated_profile(configuration, project)
    for configuration, project in _DEVICE_ASSIGNMENTS.items()
}
_PROFILES["conventional_tokamak"] = ReactorSemanticProfile(
    configuration="conventional_tokamak",
    device_project="SCPN-TOKAMAK-CORE",
    ingress_state=SemanticIngressState.VERIFIED_REVIEW_ADAPTER,
    producer_project="SCPN-FUSION-CORE",
    source_schema="scpn-fusion-core.torax-runtime-review-envelope.v1",
    adapter_api=(
        "scpn_phase_orchestrator.reactor_semantics."
        "coupled_transport_handoff_from_fusion_bytes"
    ),
    handoff_schema="scpn-phase-orchestrator.reactor-semantic-handoff.v1",
    semantic_profile=(
        "spo.reactor.conventional_tokamak.coupled_transport.nonphase_review.v1"
    ),
    semantic_profile_version="1.0.0",
)
_PROFILES["frc_compression_mif"] = ReactorSemanticProfile(
    configuration="frc_compression_mif",
    device_project="SCPN-MIF-CORE",
    ingress_state=SemanticIngressState.VERIFIED_REVIEW_ADAPTER,
    producer_project="SCPN-MIF-CORE",
    source_schema="scpn-mif-core.merge-compression-observation.v1",
    adapter_api=(
        "scpn_phase_orchestrator.reactor_semantics."
        "mif_merge_compression_handoff_from_mif_bytes"
    ),
    handoff_schema="scpn-phase-orchestrator.mif-merge-compression-handoff.v1",
    semantic_profile="spo.reactor.frc_compression_mif.merge_compression_review.v1",
    semantic_profile_version="1.0.0",
)

DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY = ReactorSemanticProfileRegistry(
    version=REACTOR_SEMANTIC_PROFILE_REGISTRY_VERSION,
    reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
    reactor_registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
    assignment_map_sha256=REACTOR_FAMILY_ASSIGNMENT_MAP_SHA256,
    profiles=_PROFILES,
)
