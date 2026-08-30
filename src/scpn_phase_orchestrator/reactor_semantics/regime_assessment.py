# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Portable reactor regime assessment

"""Digest-sealed, review-only identity for a complete reactor regime vector.

The codec validates an already supplied eight-axis assessment.  It does not
run a classifier, infer a regime, admit evidence for control, or actuate a
device.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from .observability_profiles import DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
from .regime_ontology import (
    DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY,
    AxisApplicability,
)
from .registry import DEFAULT_REACTOR_REGISTRY
from .semantic_profiles import DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
from .vocabulary import (
    PLANT_TRUTH_OWNERS,
    REVIEW_ONLY_AUTHORITY,
    ClockKind,
    EvidenceClass,
    QualityState,
    ValidityState,
    non_negative_integer,
    non_negative_real,
    probability,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_semver,
    require_sha256,
)

REACTOR_REGIME_ASSESSMENT_SCHEMA = (
    "scpn-phase-orchestrator.reactor-regime-assessment.v1"
)
REACTOR_REGIME_ASSESSMENT_VERSION = "1.0.0"
MAX_REACTOR_REGIME_ASSESSMENT_BYTES = 1024 * 1024
SPO_PROJECT = "SCPN-PHASE-ORCHESTRATOR"

_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CLASSIFIER_IDENTITY_FIELDS = (
    "classifier_id",
    "classifier_version",
    "classifier_sha256",
)
_THRESHOLD_IDENTITY_FIELDS = (
    "threshold_policy_id",
    "threshold_policy_version",
    "threshold_policy_sha256",
    "hysteresis_policy_id",
    "hysteresis_policy_version",
    "hysteresis_policy_sha256",
)
_CLASSIFIER_FIELDS = (*_CLASSIFIER_IDENTITY_FIELDS, *_THRESHOLD_IDENTITY_FIELDS)


class ReactorRegimeAxisDisposition(StrEnum):
    """Classification disposition, separate from static applicability."""

    CLASSIFIED = "classified"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class ReactorRegimeEvidenceBinding:
    """Bind one ontology-required evidence role to a referenced artifact."""

    role_id: str
    reference_id: str

    def __post_init__(self) -> None:
        """Validate and normalise the evidence binding identifiers."""
        object.__setattr__(
            self,
            "role_id",
            require_identifier(self.role_id, field="evidence role_id"),
        )
        object.__setattr__(
            self,
            "reference_id",
            require_identifier(self.reference_id, field="evidence reference_id"),
        )

    def to_record(self) -> dict[str, str]:
        """Return one deterministic evidence-role binding.

        Returns
        -------
        dict[str, str]
            JSON-compatible role and reference identifiers.
        """
        return {"reference_id": self.reference_id, "role_id": self.role_id}

    @classmethod
    def from_record(cls, raw: object) -> ReactorRegimeEvidenceBinding:
        """Decode one strict evidence-role binding.

        Parameters
        ----------
        raw : object
            Candidate serialized evidence binding.

        Returns
        -------
        ReactorRegimeEvidenceBinding
            Validated immutable role-to-reference binding.
        """
        record = require_exact_keys(
            raw,
            required=frozenset({"reference_id", "role_id"}),
            field="regime evidence binding",
        )
        return cls(
            role_id=cast(str, record["role_id"]),
            reference_id=cast(str, record["reference_id"]),
        )


@dataclass(frozen=True, slots=True)
class ReactorRegimeAxisAssessment:
    """One evidence-bound row of a complete regime assessment."""

    axis_id: str
    static_applicability: AxisApplicability
    disposition: ReactorRegimeAxisDisposition
    label: str | None
    confidence: float
    observability: float
    uncertainty_probability: float
    uncertainty_basis_id: str | None
    evidence_ids: tuple[str, ...]
    evidence_bindings: tuple[ReactorRegimeEvidenceBinding, ...]
    evidence_class: EvidenceClass
    validity: ValidityState
    quality: QualityState
    validity_id: str
    quality_id: str
    provenance_id: str
    applicability_basis: tuple[str, ...]
    unknown_reason_id: str | None
    classifier_id: str | None
    classifier_version: str | None
    classifier_sha256: str | None
    threshold_policy_id: str | None
    threshold_policy_version: str | None
    threshold_policy_sha256: str | None
    hysteresis_policy_id: str | None
    hysteresis_policy_version: str | None
    hysteresis_policy_sha256: str | None
    dwell_samples: int | None
    authority: str = REVIEW_ONLY_AUTHORITY
    actionable: bool = False

    def __post_init__(self) -> None:
        """Validate and normalise one ontology-bound axis assessment."""
        axis_id = require_identifier(self.axis_id, field="axis_id")
        definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(axis_id)
        static_applicability = require_enum(
            self.static_applicability,
            AxisApplicability,
            field="axis static_applicability",
        )
        if static_applicability is AxisApplicability.UNKNOWN:
            raise ValueError("static axis applicability cannot be unknown")
        disposition = require_enum(
            self.disposition,
            ReactorRegimeAxisDisposition,
            field="axis disposition",
        )
        confidence = probability(self.confidence, field="axis confidence")
        observability = probability(
            self.observability,
            field="axis observability",
        )
        uncertainty = probability(
            self.uncertainty_probability,
            field="axis uncertainty_probability",
        )
        evidence = _identifiers(self.evidence_ids, field="evidence_id")
        bindings = tuple(self.evidence_bindings)
        binding_keys = tuple((item.role_id, item.reference_id) for item in bindings)
        binding_roles = tuple(item.role_id for item in bindings)
        if tuple(sorted(set(binding_keys))) != binding_keys or len(
            set(binding_roles)
        ) != len(binding_roles):
            raise ValueError("evidence bindings must be unique and sorted")
        basis = _identifiers(
            self.applicability_basis,
            field="applicability_basis",
        )
        evidence_class = require_enum(
            self.evidence_class,
            EvidenceClass,
            field="axis evidence_class",
        )
        validity = require_enum(self.validity, ValidityState, field="axis validity")
        quality = require_enum(self.quality, QualityState, field="axis quality")
        for field_name in ("validity_id", "quality_id", "provenance_id"):
            object.__setattr__(
                self,
                field_name,
                require_identifier(getattr(self, field_name), field=field_name),
            )
        if self.authority != REVIEW_ONLY_AUTHORITY or self.actionable is not False:
            raise ValueError("regime assessment axes must remain review-only")

        if disposition is ReactorRegimeAxisDisposition.CLASSIFIED:
            if static_applicability is not AxisApplicability.APPLICABLE:
                raise ValueError("classified axis must be statically applicable")
            self._validate_classified(
                definition.labels,
                definition.required_evidence,
                confidence,
                observability,
                bindings,
            )
        elif disposition is ReactorRegimeAxisDisposition.UNKNOWN:
            if static_applicability is not AxisApplicability.APPLICABLE:
                raise ValueError("unknown axis must be statically applicable")
            self._validate_unknown(confidence, uncertainty, bindings)
        else:
            if static_applicability is not AxisApplicability.NOT_APPLICABLE:
                raise ValueError("not-applicable disposition requires static support")
            self._validate_not_applicable(
                confidence,
                observability,
                uncertainty,
                evidence,
                basis,
                bindings,
            )

        object.__setattr__(self, "axis_id", axis_id)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "observability", observability)
        object.__setattr__(self, "uncertainty_probability", uncertainty)
        object.__setattr__(self, "evidence_ids", evidence)
        object.__setattr__(self, "evidence_bindings", bindings)
        object.__setattr__(self, "applicability_basis", basis)
        object.__setattr__(self, "evidence_class", evidence_class)
        object.__setattr__(self, "validity", validity)
        object.__setattr__(self, "quality", quality)

    def _validate_classified(
        self,
        labels: tuple[str, ...],
        required_evidence: tuple[str, ...],
        confidence: float,
        observability: float,
        bindings: tuple[ReactorRegimeEvidenceBinding, ...],
    ) -> None:
        """Validate evidence and policy identity for a classified axis."""
        if self.label is None or self.label == "unknown":
            raise ValueError("classified assessment requires a classified label")
        label = require_identifier(self.label, field="axis label")
        if label not in labels:
            raise ValueError("axis label is not defined by the ontology")
        if confidence == 0.0 or observability == 0.0:
            raise ValueError(
                "classified assessment requires non-zero confidence and observability"
            )
        if not self.evidence_ids:
            raise ValueError("classified assessment requires evidence")
        if self.unknown_reason_id is not None:
            raise ValueError("classified assessment forbids unknown_reason_id")
        if self.uncertainty_basis_id is None:
            raise ValueError("classified assessment requires uncertainty_basis_id")
        object.__setattr__(
            self,
            "uncertainty_basis_id",
            require_identifier(
                self.uncertainty_basis_id,
                field="uncertainty_basis_id",
            ),
        )
        roles = tuple(item.role_id for item in bindings)
        if roles != tuple(sorted(required_evidence)):
            raise ValueError(
                "classified assessment requires every ontology evidence role"
            )
        if self.validity not in {ValidityState.VALID, ValidityState.DEGRADED}:
            raise ValueError("classified assessment requires usable validity")
        if self.quality not in {QualityState.VALID, QualityState.DEGRADED}:
            raise ValueError("classified assessment requires usable quality")
        if self.evidence_class in {
            EvidenceClass.CONCEPT,
            EvidenceClass.SCAFFOLD,
            EvidenceClass.REVIEW_HYPOTHESIS,
            EvidenceClass.UNKNOWN,
        }:
            raise ValueError("classified assessment requires qualified evidence class")
        needs_classifier = "classifier" in required_evidence
        needs_threshold = "threshold_provenance" in required_evidence
        self._validate_classifier_policies(needs_classifier, needs_threshold)
        object.__setattr__(self, "label", label)

    def _validate_classifier_policies(
        self,
        needs_classifier: bool,
        needs_threshold: bool,
    ) -> None:
        """Require only the classifier policies mandated by the ontology."""
        required_names: tuple[str, ...] = ()
        if needs_classifier:
            required_names += _CLASSIFIER_IDENTITY_FIELDS
        if needs_threshold:
            required_names += _THRESHOLD_IDENTITY_FIELDS
        forbidden_names = tuple(
            name for name in _CLASSIFIER_FIELDS if name not in required_names
        )
        for field_name in required_names:
            value = getattr(self, field_name)
            if value is None:
                raise ValueError("classified assessment requires classifier policies")
            validator = (
                require_sha256
                if field_name.endswith("sha256")
                else (
                    require_semver
                    if field_name.endswith("version")
                    else require_identifier
                )
            )
            object.__setattr__(self, field_name, validator(value, field=field_name))
        if any(getattr(self, name) is not None for name in forbidden_names):
            raise ValueError(
                "axis supplies classifier policies not required by ontology"
            )
        if needs_threshold:
            dwell = non_negative_integer(self.dwell_samples, field="dwell_samples")
            if dwell == 0:
                raise ValueError(
                    "classified assessment requires positive dwell_samples"
                )
            object.__setattr__(self, "dwell_samples", dwell)
        elif self.dwell_samples is not None:
            raise ValueError("axis supplies dwell_samples without threshold provenance")

    def _validate_unknown(
        self,
        confidence: float,
        uncertainty: float,
        bindings: tuple[ReactorRegimeEvidenceBinding, ...],
    ) -> None:
        """Validate uncertainty and evidence state for an unknown axis."""
        if self.label is not None:
            raise ValueError("unknown assessment forbids a physics label")
        if confidence != 0.0:
            raise ValueError("unknown assessment confidence must be zero")
        if self.unknown_reason_id is None:
            raise ValueError("unknown assessment requires unknown_reason_id")
        if uncertainty != 1.0:
            raise ValueError("unknown assessment uncertainty_probability must be one")
        if self.uncertainty_basis_id is None:
            raise ValueError("unknown assessment requires uncertainty_basis_id")
        if not self.evidence_ids:
            no_evidence_state = (
                self.observability == 0.0
                and self.evidence_class is EvidenceClass.UNKNOWN
                and self.validity is ValidityState.UNKNOWN
                and self.quality is QualityState.UNKNOWN
                and not bindings
            )
            if not no_evidence_state:
                raise ValueError(
                    "unknown assessment without evidence requires explicit "
                    "unknown states"
                )
        object.__setattr__(
            self,
            "unknown_reason_id",
            require_identifier(self.unknown_reason_id, field="unknown_reason_id"),
        )
        object.__setattr__(
            self,
            "uncertainty_basis_id",
            require_identifier(
                self.uncertainty_basis_id,
                field="uncertainty_basis_id",
            ),
        )
        definition = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.resolve_axis(self.axis_id)
        if not {item.role_id for item in bindings}.issubset(
            definition.required_evidence
        ):
            raise ValueError("unknown assessment contains an undefined evidence role")
        self._forbid_classifier_result("unknown")

    def _validate_not_applicable(
        self,
        confidence: float,
        observability: float,
        uncertainty: float,
        evidence: tuple[str, ...],
        basis: tuple[str, ...],
        bindings: tuple[ReactorRegimeEvidenceBinding, ...],
    ) -> None:
        """Validate an ontology-supported not-applicable axis."""
        if self.label is not None:
            raise ValueError("not-applicable assessment forbids a physics label")
        if confidence != 0.0 or observability != 0.0 or uncertainty != 0.0:
            raise ValueError("not-applicable assessment metrics must be zero")
        if evidence:
            raise ValueError("not-applicable assessment forbids evidence")
        if not basis:
            raise ValueError("not-applicable assessment requires applicability basis")
        if bindings:
            raise ValueError("not-applicable assessment forbids evidence bindings")
        if self.unknown_reason_id is not None:
            raise ValueError("not-applicable assessment forbids unknown_reason_id")
        if self.uncertainty_basis_id is not None:
            raise ValueError("not-applicable assessment forbids uncertainty_basis_id")
        if (
            self.evidence_class is not EvidenceClass.REVIEW_HYPOTHESIS
            or self.validity is not ValidityState.VALID
            or self.quality is not QualityState.VALID
        ):
            raise ValueError(
                "not-applicable assessment requires valid ontology review identity"
            )
        self._forbid_classifier_result("not-applicable")

    def _forbid_classifier_result(self, disposition: str) -> None:
        """Reject classifier outputs on an unclassified axis."""
        has_classifier_field = any(
            getattr(self, field_name) is not None for field_name in _CLASSIFIER_FIELDS
        )
        if has_classifier_field:
            raise ValueError(f"{disposition} assessment forbids classifier policies")
        if self.dwell_samples is not None:
            raise ValueError(f"{disposition} assessment forbids dwell_samples")

    def to_record(self) -> dict[str, object]:
        """Return a complete deterministic axis row.

        Returns
        -------
        dict[str, object]
            JSON-compatible axis disposition and evidence fields.
        """
        return {
            "actionable": self.actionable,
            "disposition": self.disposition.value,
            "static_applicability": self.static_applicability.value,
            "applicability_basis": list(self.applicability_basis),
            "authority": self.authority,
            "axis_id": self.axis_id,
            "classifier_id": self.classifier_id,
            "classifier_sha256": self.classifier_sha256,
            "classifier_version": self.classifier_version,
            "confidence": self.confidence,
            "dwell_samples": self.dwell_samples,
            "evidence_class": self.evidence_class.value,
            "evidence_bindings": [item.to_record() for item in self.evidence_bindings],
            "evidence_ids": list(self.evidence_ids),
            "hysteresis_policy_id": self.hysteresis_policy_id,
            "hysteresis_policy_sha256": self.hysteresis_policy_sha256,
            "hysteresis_policy_version": self.hysteresis_policy_version,
            "label": self.label,
            "observability": self.observability,
            "provenance_id": self.provenance_id,
            "quality": self.quality.value,
            "quality_id": self.quality_id,
            "threshold_policy_id": self.threshold_policy_id,
            "threshold_policy_sha256": self.threshold_policy_sha256,
            "threshold_policy_version": self.threshold_policy_version,
            "uncertainty_basis_id": self.uncertainty_basis_id,
            "uncertainty_probability": self.uncertainty_probability,
            "unknown_reason_id": self.unknown_reason_id,
            "validity": self.validity.value,
            "validity_id": self.validity_id,
        }

    @classmethod
    def from_record(cls, raw: object) -> ReactorRegimeAxisAssessment:
        """Decode one strict axis row.

        Parameters
        ----------
        raw : object
            Candidate serialized axis-assessment mapping.

        Returns
        -------
        ReactorRegimeAxisAssessment
            Validated immutable axis assessment.

        Raises
        ------
        ValueError
            If fields, enums, evidence bindings, or disposition invariants fail.
        """
        record = require_exact_keys(
            raw,
            required=_AXIS_FIELDS,
            field="reactor regime axis assessment",
        )
        try:
            static_applicability = AxisApplicability(
                cast(str, record["static_applicability"])
            )
            disposition = ReactorRegimeAxisDisposition(cast(str, record["disposition"]))
            evidence_class = EvidenceClass(cast(str, record["evidence_class"]))
            validity = ValidityState(cast(str, record["validity"]))
            quality = QualityState(cast(str, record["quality"]))
        except ValueError as exc:
            raise ValueError("unknown reactor regime axis enum value") from exc
        return cls(
            axis_id=cast(str, record["axis_id"]),
            static_applicability=static_applicability,
            disposition=disposition,
            label=cast(str | None, record["label"]),
            confidence=cast(float, record["confidence"]),
            observability=cast(float, record["observability"]),
            uncertainty_probability=cast(
                float,
                record["uncertainty_probability"],
            ),
            uncertainty_basis_id=cast(
                str | None,
                record["uncertainty_basis_id"],
            ),
            evidence_ids=_string_tuple(record["evidence_ids"], field="evidence_ids"),
            evidence_bindings=_binding_tuple(record["evidence_bindings"]),
            evidence_class=evidence_class,
            validity=validity,
            quality=quality,
            validity_id=cast(str, record["validity_id"]),
            quality_id=cast(str, record["quality_id"]),
            provenance_id=cast(str, record["provenance_id"]),
            applicability_basis=_string_tuple(
                record["applicability_basis"],
                field="applicability_basis",
            ),
            unknown_reason_id=cast(str | None, record["unknown_reason_id"]),
            classifier_id=cast(str | None, record["classifier_id"]),
            classifier_version=cast(str | None, record["classifier_version"]),
            classifier_sha256=cast(str | None, record["classifier_sha256"]),
            threshold_policy_id=cast(str | None, record["threshold_policy_id"]),
            threshold_policy_version=cast(
                str | None,
                record["threshold_policy_version"],
            ),
            threshold_policy_sha256=cast(
                str | None,
                record["threshold_policy_sha256"],
            ),
            hysteresis_policy_id=cast(
                str | None,
                record["hysteresis_policy_id"],
            ),
            hysteresis_policy_version=cast(
                str | None,
                record["hysteresis_policy_version"],
            ),
            hysteresis_policy_sha256=cast(
                str | None,
                record["hysteresis_policy_sha256"],
            ),
            dwell_samples=cast(int | None, record["dwell_samples"]),
            authority=cast(str, record["authority"]),
            actionable=cast(bool, record["actionable"]),
        )


_AXIS_FIELDS = frozenset(
    {
        "actionable",
        "applicability_basis",
        "authority",
        "axis_id",
        "classifier_id",
        "classifier_sha256",
        "classifier_version",
        "confidence",
        "disposition",
        "dwell_samples",
        "evidence_class",
        "evidence_bindings",
        "evidence_ids",
        "hysteresis_policy_id",
        "hysteresis_policy_sha256",
        "hysteresis_policy_version",
        "label",
        "observability",
        "provenance_id",
        "quality",
        "quality_id",
        "threshold_policy_id",
        "threshold_policy_sha256",
        "threshold_policy_version",
        "static_applicability",
        "uncertainty_basis_id",
        "uncertainty_probability",
        "unknown_reason_id",
        "validity",
        "validity_id",
    }
)


@dataclass(frozen=True, slots=True)
class ReactorRegimeAssessment:
    """Portable identity for a full eight-axis reactor regime assessment."""

    assessment_id: str
    reactor_context_id: str
    configuration: str
    event_id: str
    producer_project: str
    producer_revision: str
    producer_artifact_sha256: str
    source_project: str
    source_revision: str
    source_handoff_schema: str
    source_handoff_sha256: str
    source_semantic_ids: tuple[str, ...]
    clock_domain: str
    clock_kind: ClockKind
    clock_epoch: str
    clock_synchronization_id: str
    evidence_timestamp_ns: int
    assessed_at_ns: int
    valid_from_ns: int
    valid_until_ns: int
    sample_rate_hz: float
    latency_s: float
    timestamp_offset_ps: int
    axes: tuple[ReactorRegimeAxisAssessment, ...]
    reactor_registry_version: str
    reactor_registry_digest: str
    semantic_profile_registry_version: str
    semantic_profile_registry_digest: str
    observability_registry_version: str
    observability_registry_digest: str
    ontology_version: str
    ontology_digest: str
    classification_performed: bool = False
    authority: str = REVIEW_ONLY_AUTHORITY
    actionable: bool = False
    schema: str = REACTOR_REGIME_ASSESSMENT_SCHEMA
    schema_version: str = REACTOR_REGIME_ASSESSMENT_VERSION

    def __post_init__(self) -> None:
        """Validate and normalise a complete review-only assessment."""
        if self.schema != REACTOR_REGIME_ASSESSMENT_SCHEMA:
            raise ValueError("unsupported reactor regime assessment schema")
        if (
            require_semver(self.schema_version, field="assessment schema_version")
            != REACTOR_REGIME_ASSESSMENT_VERSION
        ):
            raise ValueError("unsupported reactor regime assessment version")
        for field_name in (
            "assessment_id",
            "reactor_context_id",
            "event_id",
            "source_handoff_schema",
            "clock_domain",
            "clock_epoch",
            "clock_synchronization_id",
        ):
            object.__setattr__(
                self,
                field_name,
                require_identifier(getattr(self, field_name), field=field_name),
            )
        configuration = DEFAULT_REACTOR_REGISTRY.resolve(self.configuration).identifier
        object.__setattr__(self, "configuration", configuration)
        if self.producer_project != SPO_PROJECT:
            raise ValueError("reactor regime assessment producer must be SPO")
        if self.source_project not in PLANT_TRUTH_OWNERS:
            raise ValueError("assessment source must be a reactor plant-truth owner")
        for field_name in ("producer_revision", "source_revision"):
            object.__setattr__(
                self,
                field_name,
                _git_revision(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "producer_artifact_sha256",
            "source_handoff_sha256",
            "reactor_registry_digest",
            "semantic_profile_registry_digest",
            "observability_registry_digest",
            "ontology_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        semantics = _identifiers(
            self.source_semantic_ids,
            field="source_semantic_id",
        )
        if not semantics:
            raise ValueError("assessment requires source semantic identifiers")
        object.__setattr__(self, "source_semantic_ids", semantics)
        self._validate_clock()
        self._validate_axes(configuration)
        self._validate_registry_bindings()
        if self.classification_performed is not False:
            raise ValueError("assessment codec cannot claim classifier execution")
        if self.authority != REVIEW_ONLY_AUTHORITY or self.actionable is not False:
            raise ValueError("reactor regime assessment must remain review-only")

    def _validate_clock(self) -> None:
        """Validate clock kind, time ordering, rate, latency, and offset."""
        clock_kind = require_enum(self.clock_kind, ClockKind, field="clock_kind")
        if clock_kind is ClockKind.UNKNOWN:
            raise ValueError("assessment requires a known clock kind")
        evidence_time = non_negative_integer(
            self.evidence_timestamp_ns,
            field="evidence_timestamp_ns",
        )
        assessed_at = non_negative_integer(self.assessed_at_ns, field="assessed_at_ns")
        valid_from = non_negative_integer(self.valid_from_ns, field="valid_from_ns")
        valid_until = non_negative_integer(self.valid_until_ns, field="valid_until_ns")
        if not valid_from <= evidence_time <= assessed_at <= valid_until:
            raise ValueError("assessment clock and validity times are inconsistent")
        sample_rate = non_negative_real(self.sample_rate_hz, field="sample_rate_hz")
        if sample_rate == 0.0:
            raise ValueError("assessment sample_rate_hz must be positive")
        latency = non_negative_real(self.latency_s, field="latency_s")
        offset = non_negative_integer(
            self.timestamp_offset_ps,
            field="timestamp_offset_ps",
        )
        if offset > 999:
            raise ValueError("timestamp_offset_ps must be in [0, 999]")
        object.__setattr__(self, "evidence_timestamp_ns", evidence_time)
        object.__setattr__(self, "assessed_at_ns", assessed_at)
        object.__setattr__(self, "valid_from_ns", valid_from)
        object.__setattr__(self, "valid_until_ns", valid_until)
        object.__setattr__(self, "sample_rate_hz", sample_rate)
        object.__setattr__(self, "latency_s", latency)
        object.__setattr__(self, "timestamp_offset_ps", offset)

    def _validate_axes(self, configuration: str) -> None:
        """Require eight canonical axes with configuration-bound applicability."""
        axes = tuple(self.axes)
        expected_ids = tuple(sorted(DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.axes))
        supplied_ids = tuple(axis.axis_id for axis in axes)
        if len(axes) != 8 or supplied_ids != expected_ids:
            raise ValueError(
                "assessment requires exactly eight unique axes in canonical order"
            )
        ontology = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY
        for axis in axes:
            static = ontology.applicability_for(axis.axis_id, configuration)
            if axis.static_applicability is not static:
                raise ValueError("assessment static applicability binding mismatch")
        object.__setattr__(self, "axes", axes)

    def _validate_registry_bindings(self) -> None:
        """Require exact bindings to the installed semantic registries."""
        bindings = (
            (
                self.reactor_registry_version,
                self.reactor_registry_digest,
                DEFAULT_REACTOR_REGISTRY.version,
                DEFAULT_REACTOR_REGISTRY.digest,
                "reactor registry",
            ),
            (
                self.semantic_profile_registry_version,
                self.semantic_profile_registry_digest,
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.version,
                DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest,
                "semantic profile registry",
            ),
            (
                self.observability_registry_version,
                self.observability_registry_digest,
                DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version,
                DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest,
                "observability registry",
            ),
            (
                self.ontology_version,
                self.ontology_digest,
                DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.version,
                DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.digest,
                "ontology",
            ),
        )
        for version, digest, expected_version, expected_digest, name in bindings:
            if version != expected_version or digest != expected_digest:
                raise ValueError(f"assessment {name} binding mismatch")

    def to_record(self) -> dict[str, object]:
        """Return the complete deterministic assessment payload.

        Returns
        -------
        dict[str, object]
            JSON-compatible eight-axis assessment fields.
        """
        return {
            "actionable": self.actionable,
            "assessed_at_ns": self.assessed_at_ns,
            "assessment_id": self.assessment_id,
            "authority": self.authority,
            "axes": [axis.to_record() for axis in self.axes],
            "classification_performed": self.classification_performed,
            "clock_domain": self.clock_domain,
            "clock_epoch": self.clock_epoch,
            "clock_kind": self.clock_kind.value,
            "clock_synchronization_id": self.clock_synchronization_id,
            "configuration": self.configuration,
            "event_id": self.event_id,
            "evidence_timestamp_ns": self.evidence_timestamp_ns,
            "latency_s": self.latency_s,
            "observability_registry_digest": self.observability_registry_digest,
            "observability_registry_version": self.observability_registry_version,
            "ontology_digest": self.ontology_digest,
            "ontology_version": self.ontology_version,
            "producer_artifact_sha256": self.producer_artifact_sha256,
            "producer_project": self.producer_project,
            "producer_revision": self.producer_revision,
            "reactor_context_id": self.reactor_context_id,
            "reactor_registry_digest": self.reactor_registry_digest,
            "reactor_registry_version": self.reactor_registry_version,
            "sample_rate_hz": self.sample_rate_hz,
            "semantic_profile_registry_digest": (self.semantic_profile_registry_digest),
            "semantic_profile_registry_version": (
                self.semantic_profile_registry_version
            ),
            "source_handoff_schema": self.source_handoff_schema,
            "source_handoff_sha256": self.source_handoff_sha256,
            "source_project": self.source_project,
            "source_revision": self.source_revision,
            "source_semantic_ids": list(self.source_semantic_ids),
            "timestamp_offset_ps": self.timestamp_offset_ps,
            "valid_from_ns": self.valid_from_ns,
            "valid_until_ns": self.valid_until_ns,
        }


_ASSESSMENT_FIELDS = frozenset(
    {
        "actionable",
        "assessed_at_ns",
        "assessment_id",
        "authority",
        "axes",
        "classification_performed",
        "clock_domain",
        "clock_epoch",
        "clock_kind",
        "clock_synchronization_id",
        "configuration",
        "event_id",
        "evidence_timestamp_ns",
        "latency_s",
        "observability_registry_digest",
        "observability_registry_version",
        "ontology_digest",
        "ontology_version",
        "producer_artifact_sha256",
        "producer_project",
        "producer_revision",
        "reactor_context_id",
        "reactor_registry_digest",
        "reactor_registry_version",
        "sample_rate_hz",
        "semantic_profile_registry_digest",
        "semantic_profile_registry_version",
        "source_handoff_schema",
        "source_handoff_sha256",
        "source_project",
        "source_revision",
        "source_semantic_ids",
        "timestamp_offset_ps",
        "valid_from_ns",
        "valid_until_ns",
    }
)


def regime_assessment_to_record(
    assessment: ReactorRegimeAssessment,
) -> dict[str, object]:
    """Return a digest-sealed portable assessment envelope.

    Parameters
    ----------
    assessment : ReactorRegimeAssessment
        Validated eight-axis assessment to serialize.

    Returns
    -------
    dict[str, object]
        Envelope containing the assessment payload and payload digest.
    """
    payload = assessment.to_record()
    return {
        "payload": payload,
        "payload_sha256": _canonical_digest(payload),
        "schema": assessment.schema,
        "schema_version": assessment.schema_version,
    }


def regime_assessment_from_record(raw: object) -> ReactorRegimeAssessment:
    """Decode and verify one strict assessment record.

    Parameters
    ----------
    raw : object
        Candidate serialized assessment envelope.

    Returns
    -------
    ReactorRegimeAssessment
        Validated review-only eight-axis assessment.

    Raises
    ------
    ValueError
        If schema, digest, enum, axis, clock, or registry validation fails.
    """
    envelope = require_exact_keys(
        raw,
        required=frozenset({"payload", "payload_sha256", "schema", "schema_version"}),
        field="reactor regime assessment envelope",
    )
    if envelope["schema"] != REACTOR_REGIME_ASSESSMENT_SCHEMA:
        raise ValueError("unsupported reactor regime assessment schema")
    if envelope["schema_version"] != REACTOR_REGIME_ASSESSMENT_VERSION:
        raise ValueError("unsupported reactor regime assessment version")
    payload = require_exact_keys(
        envelope["payload"],
        required=_ASSESSMENT_FIELDS,
        field="reactor regime assessment payload",
    )
    supplied_digest = require_sha256(
        envelope["payload_sha256"],
        field="payload_sha256",
    )
    if supplied_digest != _canonical_digest(payload):
        raise ValueError("reactor regime assessment payload digest mismatch")
    try:
        clock_kind = ClockKind(cast(str, payload["clock_kind"]))
    except ValueError as exc:
        raise ValueError("unknown reactor regime assessment enum value") from exc
    raw_axes = payload["axes"]
    if not isinstance(raw_axes, list):
        raise ValueError("assessment axes must be a list")
    axes = tuple(ReactorRegimeAxisAssessment.from_record(item) for item in raw_axes)
    return ReactorRegimeAssessment(
        assessment_id=cast(str, payload["assessment_id"]),
        reactor_context_id=cast(str, payload["reactor_context_id"]),
        configuration=cast(str, payload["configuration"]),
        event_id=cast(str, payload["event_id"]),
        producer_project=cast(str, payload["producer_project"]),
        producer_revision=cast(str, payload["producer_revision"]),
        producer_artifact_sha256=cast(str, payload["producer_artifact_sha256"]),
        source_project=cast(str, payload["source_project"]),
        source_revision=cast(str, payload["source_revision"]),
        source_handoff_schema=cast(str, payload["source_handoff_schema"]),
        source_handoff_sha256=cast(str, payload["source_handoff_sha256"]),
        source_semantic_ids=_string_tuple(
            payload["source_semantic_ids"],
            field="source_semantic_ids",
        ),
        clock_domain=cast(str, payload["clock_domain"]),
        clock_kind=clock_kind,
        clock_epoch=cast(str, payload["clock_epoch"]),
        clock_synchronization_id=cast(str, payload["clock_synchronization_id"]),
        evidence_timestamp_ns=cast(int, payload["evidence_timestamp_ns"]),
        assessed_at_ns=cast(int, payload["assessed_at_ns"]),
        valid_from_ns=cast(int, payload["valid_from_ns"]),
        valid_until_ns=cast(int, payload["valid_until_ns"]),
        sample_rate_hz=cast(float, payload["sample_rate_hz"]),
        latency_s=cast(float, payload["latency_s"]),
        timestamp_offset_ps=cast(int, payload["timestamp_offset_ps"]),
        axes=axes,
        reactor_registry_version=cast(str, payload["reactor_registry_version"]),
        reactor_registry_digest=cast(str, payload["reactor_registry_digest"]),
        semantic_profile_registry_version=cast(
            str,
            payload["semantic_profile_registry_version"],
        ),
        semantic_profile_registry_digest=cast(
            str,
            payload["semantic_profile_registry_digest"],
        ),
        observability_registry_version=cast(
            str,
            payload["observability_registry_version"],
        ),
        observability_registry_digest=cast(
            str,
            payload["observability_registry_digest"],
        ),
        ontology_version=cast(str, payload["ontology_version"]),
        ontology_digest=cast(str, payload["ontology_digest"]),
        classification_performed=cast(bool, payload["classification_performed"]),
        authority=cast(str, payload["authority"]),
        actionable=cast(bool, payload["actionable"]),
    )


def regime_assessment_to_bytes(assessment: ReactorRegimeAssessment) -> bytes:
    """Encode canonical UTF-8 JSON bytes.

    Parameters
    ----------
    assessment : ReactorRegimeAssessment
        Validated assessment to encode.

    Returns
    -------
    bytes
        Canonical compact JSON bytes.
    """
    return _canonical_bytes(regime_assessment_to_record(assessment))


def regime_assessment_from_bytes(data: bytes) -> ReactorRegimeAssessment:
    """Decode bounded canonical bytes with duplicate-key refusal.

    Parameters
    ----------
    data : bytes
        Candidate canonical assessment bytes.

    Returns
    -------
    ReactorRegimeAssessment
        Validated assessment reconstructed from the bytes.

    Raises
    ------
    ValueError
        If the input is malformed, duplicated, oversized, or noncanonical.
    """
    if not isinstance(data, bytes):
        raise ValueError("reactor regime assessment input must be bytes")
    if not data or len(data) > MAX_REACTOR_REGIME_ASSESSMENT_BYTES:
        raise ValueError("reactor regime assessment input size is invalid")
    try:
        text = data.decode("utf-8")
        raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid reactor regime assessment JSON") from exc
    assessment = regime_assessment_from_record(raw)
    if regime_assessment_to_bytes(assessment) != data:
        raise ValueError("reactor regime assessment bytes are not canonical")
    return assessment


def regime_assessment_digest(assessment: ReactorRegimeAssessment) -> str:
    """Return SHA-256 of the complete canonical envelope bytes.

    Parameters
    ----------
    assessment : ReactorRegimeAssessment
        Validated assessment whose canonical bytes are hashed.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(regime_assessment_to_bytes(assessment)).hexdigest()


def _git_revision(value: object, *, field: str) -> str:
    """Validate and return a lowercase 40-character Git revision."""
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value) is None:
        raise ValueError(f"{field} must be a 40-character Git revision")
    return value


def _identifiers(values: Iterable[object], *, field: str) -> tuple[str, ...]:
    """Validate a canonically sorted tuple of unique identifiers."""
    result = tuple(require_identifier(value, field=field) for value in values)
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{field} values must be unique and sorted")
    return result


def _string_tuple(raw: object, *, field: str) -> tuple[str, ...]:
    """Decode a list of strings without coercion."""
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(raw)


def _binding_tuple(raw: object) -> tuple[ReactorRegimeEvidenceBinding, ...]:
    """Decode a list of strict evidence-role bindings."""
    if not isinstance(raw, list):
        raise ValueError("evidence_bindings must be a list")
    return tuple(ReactorRegimeEvidenceBinding.from_record(item) for item in raw)


def _canonical_bytes(value: object) -> bytes:
    """Encode a value as canonical UTF-8 JSON bytes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    """Return SHA-256 of canonical JSON bytes."""
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object while rejecting duplicate keys."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
