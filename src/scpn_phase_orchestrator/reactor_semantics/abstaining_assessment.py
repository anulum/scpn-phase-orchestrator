# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Abstaining reactor regime assessment builder

"""Fail-closed handoff projection without reactor-regime classification."""

from __future__ import annotations

import hashlib

from .evidence import ClockReference
from .handoff import ReactorSemanticHandoff, handoff_to_bytes
from .mif_merge_compression import (
    MIFMergeCompressionHandoff,
    mif_merge_compression_handoff_to_bytes,
)
from .observability_profiles import DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY
from .regime_assessment import (
    SPO_PROJECT,
    ReactorRegimeAssessment,
    ReactorRegimeAxisAssessment,
    ReactorRegimeAxisDisposition,
)
from .regime_ontology import (
    DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY,
    AxisApplicability,
)
from .registry import DEFAULT_REACTOR_REGISTRY, resolve_reactor_registry_release
from .semantic_profiles import DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY
from .vocabulary import EvidenceClass, QualityState, ValidityState


def build_abstaining_regime_assessment(
    handoff: ReactorSemanticHandoff | MIFMergeCompressionHandoff,
    *,
    producer_revision: str,
    producer_artifact_sha256: str,
) -> ReactorRegimeAssessment:
    """Project a verified semantic handoff into an unclassified regime vector.

    The builder derives static applicability from the installed ontology. It
    emits only ``not_applicable`` or explicit ``unknown`` axis dispositions;
    no handoff value is interpreted as classifier evidence or a physics label.
    Source bytes retain the handoff's exact allowlisted registry release;
    the new assessment uses the installed assessment registries and ontology.

    Parameters
    ----------
    handoff : ReactorSemanticHandoff | MIFMergeCompressionHandoff
        Already-validated FUSION or MIF semantic handoff.
    producer_revision : str
        Exact 40-character SPO revision producing the assessment.
    producer_artifact_sha256 : str
        SHA-256 identity of the SPO producer artifact.

    Returns
    -------
    ReactorRegimeAssessment
        Complete deterministic eight-axis review-only assessment.

    Raises
    ------
    ValueError
        If the handoff type, registry release, clocks, common validity, or
        identities are invalid.
    """
    handoff_bytes = _canonical_handoff_bytes(handoff)
    handoff_digest = hashlib.sha256(handoff_bytes).hexdigest()
    clock = _common_handoff_clock(handoff)
    evidence_timestamp_ns = max(item.clock.timestamp_ns for item in handoff.observables)
    valid_from_ns = max(item.validity.valid_from_ns for item in handoff.observables)
    valid_until_ns = min(item.validity.valid_until_ns for item in handoff.observables)
    if not valid_from_ns <= evidence_timestamp_ns <= valid_until_ns:
        raise ValueError(
            "handoff observables have no common validity at the assessment time"
        )

    identity = handoff_digest[:24]
    axes = tuple(
        _abstaining_axis(
            axis_id,
            configuration=handoff.context.configuration,
            identity=identity,
        )
        for axis_id in sorted(DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.axes)
    )
    synchronization_id = clock.synchronized_to
    if synchronization_id is None:
        synchronization_id = f"spo.clock.correlation.unavailable.{identity}"
    return ReactorRegimeAssessment(
        assessment_id=f"spo.assessment.abstaining.{identity}",
        reactor_context_id=handoff.context.context_id,
        configuration=handoff.context.configuration,
        event_id=handoff.event_id,
        producer_project=SPO_PROJECT,
        producer_revision=producer_revision,
        producer_artifact_sha256=producer_artifact_sha256,
        source_project=handoff.source_project,
        source_revision=handoff.source_revision,
        source_handoff_schema=handoff.schema,
        source_handoff_sha256=handoff_digest,
        source_semantic_ids=tuple(
            sorted(semantic.phase_id for semantic in handoff.semantics)
        ),
        clock_domain=clock.domain,
        clock_kind=clock.kind,
        clock_epoch=clock.epoch,
        clock_synchronization_id=synchronization_id,
        evidence_timestamp_ns=evidence_timestamp_ns,
        assessed_at_ns=evidence_timestamp_ns,
        valid_from_ns=valid_from_ns,
        valid_until_ns=valid_until_ns,
        sample_rate_hz=clock.sample_rate_hz,
        latency_s=clock.latency_s,
        timestamp_offset_ps=clock.picosecond_offset,
        axes=axes,
        reactor_registry_version=DEFAULT_REACTOR_REGISTRY.version,
        reactor_registry_digest=DEFAULT_REACTOR_REGISTRY.digest,
        semantic_profile_registry_version=(
            DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.version
        ),
        semantic_profile_registry_digest=(
            DEFAULT_REACTOR_SEMANTIC_PROFILE_REGISTRY.digest
        ),
        observability_registry_version=(
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.version
        ),
        observability_registry_digest=(
            DEFAULT_REACTOR_OBSERVABILITY_PROFILE_REGISTRY.digest
        ),
        ontology_version=DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.version,
        ontology_digest=DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.digest,
    )


def _canonical_handoff_bytes(
    handoff: ReactorSemanticHandoff | MIFMergeCompressionHandoff,
) -> bytes:
    """Encode one supported, already-validated handoff canonically."""
    if not isinstance(handoff, (MIFMergeCompressionHandoff, ReactorSemanticHandoff)):
        raise ValueError("unsupported reactor semantic handoff type")
    registry = resolve_reactor_registry_release(
        handoff.context.registry_version,
        handoff.context.registry_digest,
    )
    if isinstance(handoff, MIFMergeCompressionHandoff):
        return mif_merge_compression_handoff_to_bytes(handoff, registry=registry)
    return handoff_to_bytes(handoff, registry=registry)


def _common_handoff_clock(
    handoff: ReactorSemanticHandoff | MIFMergeCompressionHandoff,
) -> ClockReference:
    """Return one clock only when every handoff observable agrees exactly."""
    first = handoff.observables[0].clock
    identity = (
        first.domain,
        first.kind,
        first.epoch,
        first.sample_rate_hz,
        first.latency_s,
        first.picosecond_offset,
        first.synchronized_to,
    )
    for observable in handoff.observables[1:]:
        clock = observable.clock
        if (
            clock.domain,
            clock.kind,
            clock.epoch,
            clock.sample_rate_hz,
            clock.latency_s,
            clock.picosecond_offset,
            clock.synchronized_to,
        ) != identity:
            raise ValueError(
                "handoff observables require identical assessment clock metadata"
            )
    return first


def _abstaining_axis(
    axis_id: str,
    *,
    configuration: str,
    identity: str,
) -> ReactorRegimeAxisAssessment:
    """Build one ontology-only not-applicable or explicit unknown axis row."""
    static = DEFAULT_REACTOR_REGIME_MODE_ONTOLOGY.applicability_for(
        axis_id,
        configuration,
    )
    if static is AxisApplicability.NOT_APPLICABLE:
        return _new_abstaining_axis(
            axis_id,
            static=static,
            identity=identity,
            disposition=ReactorRegimeAxisDisposition.NOT_APPLICABLE,
            uncertainty_probability=0.0,
            uncertainty_basis_id=None,
            evidence_class=EvidenceClass.REVIEW_HYPOTHESIS,
            validity=ValidityState.VALID,
            quality=QualityState.VALID,
            applicability_basis=(
                f"spo.ontology.{configuration}.{axis_id}.not-applicable",
            ),
            unknown_reason_id=None,
        )
    return _new_abstaining_axis(
        axis_id,
        static=static,
        identity=identity,
        disposition=ReactorRegimeAxisDisposition.UNKNOWN,
        uncertainty_probability=1.0,
        uncertainty_basis_id=f"spo.assessment.{axis_id}.no-classifier-evidence",
        evidence_class=EvidenceClass.UNKNOWN,
        validity=ValidityState.UNKNOWN,
        quality=QualityState.UNKNOWN,
        applicability_basis=(),
        unknown_reason_id=f"spo.assessment.{axis_id}.classification-not-performed",
    )


def _new_abstaining_axis(
    axis_id: str,
    *,
    static: AxisApplicability,
    identity: str,
    disposition: ReactorRegimeAxisDisposition,
    uncertainty_probability: float,
    uncertainty_basis_id: str | None,
    evidence_class: EvidenceClass,
    validity: ValidityState,
    quality: QualityState,
    applicability_basis: tuple[str, ...],
    unknown_reason_id: str | None,
) -> ReactorRegimeAxisAssessment:
    """Construct one abstaining row with every classifier field absent."""
    return ReactorRegimeAxisAssessment(
        axis_id=axis_id,
        static_applicability=static,
        disposition=disposition,
        label=None,
        confidence=0.0,
        observability=0.0,
        uncertainty_probability=uncertainty_probability,
        uncertainty_basis_id=uncertainty_basis_id,
        evidence_ids=(),
        evidence_bindings=(),
        evidence_class=evidence_class,
        validity=validity,
        quality=quality,
        validity_id=f"spo.assessment.{identity}.{axis_id}.validity",
        quality_id=f"spo.assessment.{identity}.{axis_id}.quality",
        provenance_id=f"spo.assessment.{identity}.{axis_id}.source-handoff",
        applicability_basis=applicability_basis,
        unknown_reason_id=unknown_reason_id,
        classifier_id=None,
        classifier_version=None,
        classifier_sha256=None,
        threshold_policy_id=None,
        threshold_policy_version=None,
        threshold_policy_sha256=None,
        hysteresis_policy_id=None,
        hysteresis_policy_version=None,
        hysteresis_policy_sha256=None,
        dwell_samples=None,
    )
