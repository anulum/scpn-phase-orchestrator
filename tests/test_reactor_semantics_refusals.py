# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor-semantic refusal matrix

"""Invalid public inputs exercise every meaningful U0 refusal boundary."""

from __future__ import annotations

from dataclasses import replace

import pytest

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    ClockKind,
    ClockReference,
    ConfinementFamily,
    EvidenceClass,
    OperatingCadence,
    PhaseRelation,
    PhaseRelationType,
    PhaseSemanticRecord,
    ProvenanceRecord,
    QualityAssessment,
    QualityState,
    ReactorConfiguration,
    ReactorConfigurationRegistry,
    ReactorContext,
    RegimeEstimate,
    RelationInterpretation,
    ValidityState,
    ValidityWindow,
    build_phase_relation,
    build_reactor_reference_portfolio,
    canonical_json,
    contract_from_record,
    contract_to_record,
)
from scpn_phase_orchestrator.reactor_semantics.vocabulary import (
    finite_real,
    non_negative_integer,
    non_negative_real,
    probability,
    require_enum,
    require_exact_keys,
    require_identifier,
    require_semver,
    require_sha256,
    require_text,
    require_u0_schema,
)


def test_context_refusal_matrix_and_registry_identity() -> None:
    context = build_reactor_reference_portfolio()[0].context
    driver = context.drivers[0]

    for kwargs, match in (
        ({"drivers": ()}, "at least one driver"),
        ({"drivers": (driver, driver)}, "drivers must be unique"),
        (
            {"cadence": OperatingCadence.PULSED_SHOT, "event_id": None},
            "requires event_id",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            replace(context, **kwargs)

    with pytest.raises(ValueError, match="registry_version"):
        replace(context, registry_version="9.0.0").validate_registry()
    with pytest.raises(ValueError, match="confinement_family"):
        replace(
            context,
            confinement_family=ConfinementFamily.INERTIAL,
        ).validate_registry()
    with pytest.raises(ValueError, match="topology"):
        replace(context, topology="contradiction").validate_registry()

    for field, value, match in (
        ("drivers", "not-a-list", "drivers must be a list"),
        ("operating_point", [], "operating_point must be an object"),
    ):
        record = context.to_record()
        record[field] = value
        with pytest.raises(ValueError, match=match):
            ReactorContext.from_record(record)


def test_observable_evidence_and_json_refusal_matrix() -> None:
    observable = build_reactor_reference_portfolio()[0].observable

    with pytest.raises(ValueError, match="source owner"):
        replace(
            observable,
            provenance=replace(observable.provenance, source_project="external"),
        )
    with pytest.raises(ValueError, match="postdate"):
        replace(
            observable,
            calibration=replace(
                observable.calibration,
                calibrated_at_ns=observable.clock.timestamp_ns + 1,
            ),
        )
    with pytest.raises(ValueError, match="outside its validity"):
        replace(
            observable,
            validity=ValidityWindow(
                ValidityState.VALID,
                valid_from_ns=observable.clock.timestamp_ns + 1,
                valid_until_ns=observable.clock.timestamp_ns + 2,
            ),
        )
    with pytest.raises(ValueError, match="usable validity"):
        replace(
            observable,
            quality=QualityAssessment(QualityState.UNKNOWN, ("missing",)),
        )

    nested = replace(
        observable.reactor_context,
        operating_point={"array": [1, {"nested": True}]},
    )
    assert "nested" in canonical_json(nested)
    with pytest.raises(ValueError, match="JSON-compatible"):
        replace(observable, value=object())
    with pytest.raises(ValueError, match="non-empty strings"):
        replace(observable.reactor_context, operating_point={"": 1})


def test_phase_record_refusal_matrix() -> None:
    phase = build_reactor_reference_portfolio()[0].semantics[0]

    for kwargs, match in (
        ({"observable_ids": ()}, "at least one observable"),
        (
            {"observable_ids": (phase.observable_ids[0], phase.observable_ids[0])},
            "must be unique",
        ),
        ({"observability_threshold": 0.0}, "must be positive"),
        ({"phase_rad": 7.0}, r"\[0, 2\*pi\)"),
        (
            {
                "validity": ValidityWindow(
                    ValidityState.STALE,
                    0,
                    2_000_000,
                    ("stale",),
                )
            },
            "non-usable phase record",
        ),
        ({"phase_rad": None}, "usable phase carrier requires"),
        ({"amplitude": None}, "complex_mode requires"),
        ({"phase_origin": None}, "phase_rad requires"),
        ({"mode_harmonic": (1,)}, "two integers"),
        ({"mode_harmonic": (True, 1)}, "two integers"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(phase, **kwargs)

    record = phase.to_record()
    record["observable_ids"] = "bad"
    with pytest.raises(ValueError, match="must be a list"):
        PhaseSemanticRecord.from_record(record)

    protocol = build_reactor_reference_portfolio()[2].semantics[1]
    assert PhaseSemanticRecord.from_record(protocol.to_record()) == protocol


def test_relation_refusal_matrix_and_explicit_transforms() -> None:
    source = build_reactor_reference_portfolio()[0].semantics[0]
    target = replace(source, phase_id="u0.a1.relation_target", phase_rad=0.8)
    relation = build_phase_relation(
        source,
        target,
        relation_id="u0.a1.relation",
        relation_type=PhaseRelationType.SAME_MODE,
        interpretation=RelationInterpretation.AMBIGUOUS,
        identification_method="review",
        evidence_class=EvidenceClass.SCAFFOLD,
    )

    with pytest.raises(ValueError, match="distinct"):
        replace(relation, target_phase_id=relation.source_phase_id)
    with pytest.raises(ValueError, match="positive integers"):
        replace(relation, harmonic_ratio=(0, 1))

    transformed = build_phase_relation(
        source,
        replace(
            target,
            reference_frame="u0.a1.transformed_frame",
            clock_kind=ClockKind.SHOT_RELATIVE,
        ),
        relation_id="u0.a1.transformed_relation",
        relation_type=PhaseRelationType.SAME_MODE,
        interpretation=RelationInterpretation.CONTEXT_DEPENDENT,
        identification_method="declared_transform",
        evidence_class=EvidenceClass.SCAFFOLD,
        reference_transform="u0.a1.frame_transform",
        clock_transform_id="u0.a1.clock_transform",
        causal_direction="source_to_target",
    )
    assert PhaseRelation.from_record(transformed.to_record()) == transformed

    bounded = build_reactor_reference_portfolio()[3].semantics[0]
    with pytest.raises(ValueError, match="source carrier"):
        build_phase_relation(
            bounded,
            target,
            relation_id="u0.bad_source",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    with pytest.raises(ValueError, match="target carrier"):
        build_phase_relation(
            source,
            replace(bounded, phase_id="u0.bad_target"),
            relation_id="u0.bad_target_relation",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    unusable = replace(source, confidence=0.0)
    with pytest.raises(ValueError, match="usable source"):
        build_phase_relation(
            unusable,
            target,
            relation_id="u0.unusable",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    with pytest.raises(ValueError, match="cross-context"):
        build_phase_relation(
            source,
            replace(target, reactor_context_id="u0.other.context"),
            relation_id="u0.cross_context",
            relation_type=PhaseRelationType.SAME_MODE,
            interpretation=RelationInterpretation.AMBIGUOUS,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )
    with pytest.raises(ValueError, match="explicit ratio"):
        build_phase_relation(
            source,
            replace(target, mode_harmonic=(3, 2)),
            relation_id="u0.harmonic_default",
            relation_type=PhaseRelationType.HARMONIC,
            interpretation=RelationInterpretation.CONTEXT_DEPENDENT,
            identification_method="review",
            evidence_class=EvidenceClass.SCAFFOLD,
        )


def test_regime_refusal_matrix() -> None:
    regime = build_reactor_reference_portfolio()[0].regime
    duplicate_axis = (regime.axes[0], regime.axes[0])
    for kwargs, match in (
        ({"axes": ()}, "at least one axis"),
        ({"axes": duplicate_axis}, "axis names must be unique"),
        ({"evidence_ids": ()}, "requires evidence_ids"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(regime, **kwargs)

    for field, value, match in (
        ("axes", {}, "axes must be a list"),
        ("evidence_ids", {}, "evidence_ids must be a list"),
        ("threshold_provenance", {}, "threshold_provenance must be a list"),
    ):
        record = regime.to_record()
        record[field] = value
        with pytest.raises(ValueError, match=match):
            RegimeEstimate.from_record(record)


def test_evidence_primitive_refusal_matrix() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ClockReference("clock", ClockKind.UNKNOWN, "epoch", 0, 0.0, 0.0)
    synchronized = ClockReference(
        "clock",
        ClockKind.FACILITY_SYNCHRONIZED,
        "epoch",
        1,
        1.0,
        0.0,
        picosecond_offset=250,
        synchronized_to="facility.master",
    )
    assert ClockReference.from_record(synchronized.to_record()) == synchronized
    bad_clock = synchronized.to_record()
    bad_clock["synchronized_to"] = 1
    with pytest.raises(ValueError, match="string or null"):
        ClockReference.from_record(bad_clock)
    with pytest.raises(ValueError, match=r"\[0, 999\]"):
        replace(synchronized, picosecond_offset=1_000)

    with pytest.raises(ValueError, match="unique"):
        QualityAssessment(QualityState.DEGRADED, ("fault", "fault"))
    quality = QualityAssessment(QualityState.DEGRADED, ("fault",), 2.0)
    assert QualityAssessment.from_record(quality.to_record()) == quality
    with pytest.raises(ValueError, match="valid quality"):
        QualityAssessment(QualityState.VALID, ("fault",))
    bad_quality = quality.to_record()
    bad_quality["flags"] = "fault"
    with pytest.raises(ValueError, match="list of strings"):
        QualityAssessment.from_record(bad_quality)

    with pytest.raises(ValueError, match=">="):
        ValidityWindow(ValidityState.VALID, 2, 1)
    with pytest.raises(ValueError, match="valid state"):
        ValidityWindow(ValidityState.VALID, 1, 2, ("reason",))
    with pytest.raises(ValueError, match="requires at least one reason"):
        ValidityWindow(ValidityState.STALE, 1, 2)
    bad_validity = {
        "state": "stale",
        "valid_from_ns": 1,
        "valid_until_ns": 2,
        "reasons": 1,
    }
    with pytest.raises(ValueError, match="list of strings"):
        ValidityWindow.from_record(bad_validity)

    provenance = ProvenanceRecord(
        "SCPN-FUSION-CORE",
        "component",
        "symbol",
        "artifact://fixture",
        "0" * 64,
        (("runtime", "python"),),
    )
    assert ProvenanceRecord.from_record(provenance.to_record()) == provenance
    with pytest.raises(ValueError, match="64 lowercase"):
        replace(provenance, sha256="bad")
    with pytest.raises(ValueError, match="unique"):
        replace(provenance, attributes=(("runtime", "python"), ("runtime", "rust")))
    bad_provenance = provenance.to_record()
    bad_provenance["attributes"] = []
    with pytest.raises(ValueError, match="string mapping"):
        ProvenanceRecord.from_record(bad_provenance)


def test_registry_and_codec_defensive_refusals() -> None:
    configuration = ReactorConfiguration(
        "laboratory.example:item",
        ConfinementFamily.EXTENSION,
        "extension",
    )
    with pytest.raises(ValueError, match="at least one"):
        ReactorConfigurationRegistry("1.0.0", {}, {})
    with pytest.raises(ValueError, match="collides"):
        ReactorConfigurationRegistry(
            "1.0.0",
            {configuration.identifier: configuration},
            {configuration.identifier: configuration.identifier},
        )
    with pytest.raises(ValueError, match="already registered"):
        DEFAULT_REACTOR_REGISTRY.register(
            ReactorConfiguration(
                "laboratory.example:item",
                ConfinementFamily.EXTENSION,
                "extension",
            )
        ).register(configuration)

    with pytest.raises(TypeError, match="unsupported"):
        contract_to_record(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload must be an object"):
        contract_from_record(
            {
                "contract_type": "reactor_context",
                "payload": [],
                "schema_version": "1.0.0",
            }
        )


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: require_text("", field="value"), "non-empty"),
        (
            lambda: require_enum("valid", QualityState, field="value"),
            "QualityState member",
        ),
        (lambda: require_identifier("bad value", field="value"), "identifier"),
        (lambda: require_semver("1", field="value"), "MAJOR.MINOR.PATCH"),
        (lambda: require_u0_schema("1.1.0"), "forward schema"),
        (lambda: require_u0_schema("1.0.1"), "historical schema"),
        (lambda: require_u0_schema("0.9.0"), "schema major"),
        (lambda: require_sha256("A" * 64, field="digest"), "lowercase"),
        (lambda: finite_real(True, field="value"), "finite real"),
        (lambda: finite_real(float("inf"), field="value"), "finite real"),
        (lambda: non_negative_real(-1.0, field="value"), "non-negative"),
        (lambda: probability(2.0, field="value"), r"\[0, 1\]"),
        (lambda: non_negative_integer(True, field="value"), "non-negative integer"),
        (
            lambda: require_exact_keys([], required=frozenset(), field="value"),
            "object with string keys",
        ),
    ],
)
def test_vocabulary_refusal_matrix(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
