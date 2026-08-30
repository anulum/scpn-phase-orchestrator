# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic public facade

"""Reactor-family-neutral phase semantics and regime evidence.

This facade is non-actuating. It describes reactor context, observations,
semantic carriers, compatible relations, and review-only regime estimates.
"""

from .contracts import (
    JsonValue,
    ObservableDescriptor,
    PhaseRelation,
    PhaseSemanticRecord,
    ReactorContext,
    RegimeAxis,
    RegimeEstimate,
    build_phase_relation,
    validate_observable_sequence,
)
from .evidence import (
    CalibrationReference,
    ClockReference,
    ProvenanceRecord,
    QualityAssessment,
    Uncertainty,
    ValidityWindow,
)
from .handoff import (
    HANDOFF_SCHEMA,
    HANDOFF_SCHEMA_VERSION,
    MAX_HANDOFF_JSON_BYTES,
    MAX_SOURCE_ENVELOPE_BYTES,
    ReactorSemanticHandoff,
    canonicalize_source_envelope,
    handoff_digest,
    handoff_from_bytes,
    handoff_from_json,
    handoff_from_record,
    handoff_to_bytes,
    handoff_to_json,
    handoff_to_record,
)
from .reference_portfolio import (
    ReactorReferenceSlice,
    build_reactor_reference_portfolio,
)
from .registry import (
    DEFAULT_REACTOR_REGISTRY,
    REACTOR_REGISTRY_VERSION,
    ReactorConfiguration,
    ReactorConfigurationRegistry,
)
from .serialization import (
    ReactorSemanticContract,
    canonical_json,
    contract_digest,
    contract_from_json,
    contract_from_record,
    contract_to_record,
)
from .vocabulary import (
    ACTION_OWNER,
    PLANT_TRUTH_OWNERS,
    REVIEW_ONLY_AUTHORITY,
    SEMANTIC_OWNER,
    U0_SCHEMA_VERSION,
    ClockKind,
    ConfinementFamily,
    ConversionKind,
    DriverKind,
    EvidenceClass,
    OperatingCadence,
    PhaseRelationType,
    QualityState,
    ReactionKind,
    RegimeState,
    RelationInterpretation,
    SemanticCarrier,
    ValidityState,
)

__all__ = [
    "ACTION_OWNER",
    "DEFAULT_REACTOR_REGISTRY",
    "HANDOFF_SCHEMA",
    "HANDOFF_SCHEMA_VERSION",
    "MAX_HANDOFF_JSON_BYTES",
    "MAX_SOURCE_ENVELOPE_BYTES",
    "PLANT_TRUTH_OWNERS",
    "REACTOR_REGISTRY_VERSION",
    "REVIEW_ONLY_AUTHORITY",
    "SEMANTIC_OWNER",
    "U0_SCHEMA_VERSION",
    "CalibrationReference",
    "ClockKind",
    "ClockReference",
    "ConfinementFamily",
    "ConversionKind",
    "DriverKind",
    "EvidenceClass",
    "JsonValue",
    "ObservableDescriptor",
    "OperatingCadence",
    "PhaseRelation",
    "PhaseRelationType",
    "PhaseSemanticRecord",
    "ProvenanceRecord",
    "QualityAssessment",
    "QualityState",
    "ReactionKind",
    "ReactorConfiguration",
    "ReactorConfigurationRegistry",
    "ReactorContext",
    "ReactorReferenceSlice",
    "ReactorSemanticContract",
    "ReactorSemanticHandoff",
    "RegimeAxis",
    "RegimeEstimate",
    "RegimeState",
    "RelationInterpretation",
    "SemanticCarrier",
    "Uncertainty",
    "ValidityState",
    "ValidityWindow",
    "build_phase_relation",
    "build_reactor_reference_portfolio",
    "canonical_json",
    "canonicalize_source_envelope",
    "contract_digest",
    "contract_from_json",
    "contract_from_record",
    "contract_to_record",
    "handoff_digest",
    "handoff_from_bytes",
    "handoff_from_json",
    "handoff_from_record",
    "handoff_to_bytes",
    "handoff_to_json",
    "handoff_to_record",
    "validate_observable_sequence",
]
