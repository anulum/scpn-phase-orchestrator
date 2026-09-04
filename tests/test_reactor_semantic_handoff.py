# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Reactor semantic handoff tests

"""Public review-only handoff and tamper-refusal tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from scpn_phase_orchestrator.reactor_semantics import (
    DEFAULT_REACTOR_REGISTRY,
    HANDOFF_SCHEMA,
    HANDOFF_SCHEMA_VERSION,
    REACTOR_REGISTRY_V1_0_0,
    ClockKind,
    ClockReference,
    EvidenceClass,
    PhaseRelationType,
    PhaseSemanticRecord,
    QualityAssessment,
    QualityState,
    ReactorConfigurationRegistry,
    ReactorSemanticHandoff,
    RegimeState,
    RelationInterpretation,
    SemanticCarrier,
    ValidityState,
    ValidityWindow,
    build_phase_relation,
    build_reactor_reference_portfolio,
    canonicalize_source_envelope,
    handoff_digest,
    handoff_from_bytes,
    handoff_from_json,
    handoff_from_record,
    handoff_to_bytes,
    handoff_to_json,
    handoff_to_record,
)
from scpn_phase_orchestrator.reactor_semantics import handoff as handoff_module

SOURCE_REVISION = "7a416116f647b6640651f432c791a128bbe27f7a"
SOURCE_SCHEMA = "scpn-fusion-core.coupled-transport-semantic-envelope.v1"
HANDOFF_SCHEMA_PATH = Path("docs/specs/reactor_semantic_handoff.schema.json")
U0_SCHEMA_PATH = Path("docs/specs/reactor_semantics_u0.schema.json")


def _handoff() -> ReactorSemanticHandoff:
    item = build_reactor_reference_portfolio()[0]
    event_id = "fusion.torax.event.0001"
    context = replace(item.context, event_id=event_id)
    clock = ClockReference(
        domain="fusion.torax.simulation",
        kind=ClockKind.SIMULATION_MONOTONIC,
        epoch="fusion.torax.event.0001.start",
        timestamp_ns=20_000_000,
        sample_rate_hz=100.0,
        latency_s=0.0,
    )
    observable = replace(
        item.observable,
        observable_id="fusion.torax.ion_temperature",
        reactor_context=context,
        physical_quantity="ion temperature profile",
        units="keV",
        coordinate_frame="torax_rho_norm",
        spatial_support="rho_norm[0,1]",
        diagnostic="torax_coupled_transport",
        channel="ion_temperature_kev",
        value=(9.1, 8.2, 7.0),
        clock=clock,
        validity=ValidityWindow(
            ValidityState.VALID,
            valid_from_ns=clock.timestamp_ns,
            valid_until_ns=clock.timestamp_ns,
        ),
        provenance=replace(
            item.observable.provenance,
            source_project="SCPN-FUSION-CORE",
            component="scpn_fusion.integrations.torax",
            symbol="CoupledTransportSemanticEnvelope",
            artifact_uri="artifact:torax-coupled-transport-envelope",
        ),
    )
    semantic = PhaseSemanticRecord(
        phase_id="spo.transport.ion_temperature.bounded_feature",
        reactor_context_id=context.context_id,
        observable_ids=(observable.observable_id,),
        carrier_type=SemanticCarrier.BOUNDED_FEATURE,
        phenomenon="ion temperature transport profile",
        phase_rad=None,
        amplitude=None,
        frequency_hz=None,
        bandwidth_hz=None,
        mode_identity=None,
        mode_harmonic=None,
        phase_origin=None,
        orientation=None,
        reference_frame=observable.coordinate_frame,
        clock_domain=clock.domain,
        clock_kind=clock.kind,
        clock_epoch=clock.epoch,
        wrap_convention=None,
        reference_signal=None,
        extractor="spo.coupled_transport.bounded_feature",
        extractor_version="1.0.0",
        observation_operator=None,
        uncertainty=observable.uncertainty,
        confidence=0.0,
        observability=0.0,
        observability_threshold=1.0,
        validity=ValidityWindow(
            ValidityState.UNOBSERVABLE,
            valid_from_ns=clock.timestamp_ns,
            valid_until_ns=clock.timestamp_ns,
            reasons=("no cyclic phase observable declared by producer",),
        ),
        quality=QualityAssessment(
            QualityState.UNKNOWN,
            flags=("noncyclic_transport_evidence",),
        ),
        evidence_class=EvidenceClass.SIMULATION,
    )
    regime = replace(
        item.regime,
        reactor_context_id=context.context_id,
        state=RegimeState.UNKNOWN,
        evidence_ids=(observable.observable_id,),
        confidence=0.0,
        transition_reason="transport evidence does not identify a reactor regime",
        safety_effect="review only; no control consequence",
        validity=ValidityWindow(
            ValidityState.UNKNOWN,
            valid_from_ns=clock.timestamp_ns,
            valid_until_ns=clock.timestamp_ns,
            reasons=("no regime classifier declared by producer",),
        ),
    )
    source_json = canonicalize_source_envelope(
        json.dumps(
            {
                "event_id": event_id,
                "profiles": {"ion_temperature_kev": [9.1, 8.2, 7.0]},
                "schema": SOURCE_SCHEMA,
                "simulation_time_ns": 20_000_000,
                "source_revision": SOURCE_REVISION,
            }
        )
    )
    return ReactorSemanticHandoff(
        source_schema=SOURCE_SCHEMA,
        source_revision=SOURCE_REVISION,
        source_envelope_json=source_json,
        event_id=event_id,
        context=context,
        observables=(observable,),
        semantics=(semantic,),
        phase_relations=(),
        regime=regime,
    )


def _handoff_for_registry(
    registry: ReactorConfigurationRegistry,
) -> ReactorSemanticHandoff:
    handoff = _handoff()
    context = replace(
        handoff.context,
        registry_version=registry.version,
        registry_digest=registry.digest,
    )
    return replace(
        handoff,
        context=context,
        observables=tuple(
            replace(observable, reactor_context=context)
            for observable in handoff.observables
        ),
    )


def test_public_handoff_round_trip_is_byte_stable_and_non_actuating() -> None:
    handoff = _handoff()
    encoded = handoff_to_json(handoff)
    decoded = handoff_from_json(encoded)
    encoded_bytes = handoff_to_bytes(handoff)

    assert decoded == handoff
    assert handoff_to_json(decoded) == encoded
    assert handoff_from_bytes(encoded_bytes) == handoff
    assert handoff_digest(decoded) == handoff_digest(handoff)
    assert decoded.schema == HANDOFF_SCHEMA
    assert decoded.schema_version == HANDOFF_SCHEMA_VERSION
    assert decoded.authority == "review_only"
    assert decoded.actionable is False
    assert decoded.phase_relations == ()
    assert decoded.regime.state is RegimeState.UNKNOWN
    assert all(
        semantic.carrier_type is SemanticCarrier.BOUNDED_FEATURE
        for semantic in decoded.semantics
    )


def test_public_handoff_bytes_resolve_exact_historical_registry_release() -> None:
    handoff = _handoff_for_registry(REACTOR_REGISTRY_V1_0_0)
    encoded = handoff_to_bytes(handoff, registry=REACTOR_REGISTRY_V1_0_0)

    decoded = handoff_from_bytes(encoded)

    assert decoded == handoff
    assert handoff_to_bytes(decoded, registry=REACTOR_REGISTRY_V1_0_0) == encoded
    with pytest.raises(ValueError, match="registry version mismatch"):
        handoff_from_bytes(encoded, registry=DEFAULT_REACTOR_REGISTRY)

    digest_mismatch_registry = ReactorConfigurationRegistry(
        version=REACTOR_REGISTRY_V1_0_0.version,
        configurations=REACTOR_REGISTRY_V1_0_0.configurations,
        aliases={
            **REACTOR_REGISTRY_V1_0_0.aliases,
            "historical_digest_mismatch": "conventional_tokamak",
        },
    )
    with pytest.raises(ValueError, match="registry digest mismatch"):
        handoff_from_bytes(encoded, registry=digest_mismatch_registry)


def test_portable_schema_accepts_the_public_handoff_record() -> None:
    schema = json.loads(HANDOFF_SCHEMA_PATH.read_text(encoding="utf-8"))
    u0_schema = json.loads(U0_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    registry = Registry().with_resource(
        u0_schema["$id"],
        Resource.from_contents(u0_schema),
    )
    validator = Draft202012Validator(schema, registry=registry)

    validator.validate(handoff_to_record(_handoff()))


def test_public_handoff_refuses_payload_and_source_tampering() -> None:
    record = handoff_to_record(_handoff())
    payload = record["payload"]
    assert isinstance(payload, dict)
    payload["event_id"] = "fusion.torax.event.tampered"
    with pytest.raises(ValueError, match="payload digest mismatch"):
        handoff_from_record(record)

    record = handoff_to_record(_handoff())
    payload = record["payload"]
    assert isinstance(payload, dict)
    payload["source_envelope_json"] = "{}"
    record["payload_sha256"] = _record_payload_digest(payload)
    with pytest.raises(ValueError, match="source envelope digest mismatch"):
        handoff_from_record(record)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "scpn-fusion-core.other-envelope.v1"),
        ("source_revision", "f" * 40),
        ("event_id", "fusion.torax.event.other"),
    ],
)
def test_public_handoff_refuses_resealed_source_identity_drift(
    field: str,
    value: str,
) -> None:
    record = handoff_to_record(_handoff())
    payload = record["payload"]
    assert isinstance(payload, dict)
    source = json.loads(payload["source_envelope_json"])
    source[field] = value
    source_json = json.dumps(
        source,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    payload["source_envelope_json"] = source_json
    payload["source_envelope_sha256"] = hashlib.sha256(source_json.encode()).hexdigest()
    record["payload_sha256"] = _record_payload_digest(payload)

    with pytest.raises(ValueError, match=rf"source {field} does not match"):
        handoff_from_record(record)


def test_public_handoff_refuses_missing_embedded_source_identity() -> None:
    record = handoff_to_record(_handoff())
    payload = record["payload"]
    assert isinstance(payload, dict)
    source = json.loads(payload["source_envelope_json"])
    del source["event_id"]
    source_json = canonicalize_source_envelope(json.dumps(source))
    payload["source_envelope_json"] = source_json
    payload["source_envelope_sha256"] = hashlib.sha256(
        source_json.encode("utf-8")
    ).hexdigest()
    record["payload_sha256"] = _record_payload_digest(payload)

    with pytest.raises(
        ValueError,
        match="embedded FUSION source envelope lacks event_id",
    ):
        handoff_from_record(record)


def test_public_handoff_refuses_duplicate_keys_and_noncanonical_source() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key"):
        handoff_from_json('{"schema":"first","schema":"second"}')
    with pytest.raises(ValueError, match="canonical JSON"):
        replace(_handoff(), source_envelope_json='{ "schema": "bad" }')
    with pytest.raises(ValueError, match="source envelope JSON is invalid"):
        canonicalize_source_envelope("{")
    with pytest.raises(ValueError, match="must contain an object"):
        canonicalize_source_envelope("[]")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: replace(item, actionable=True), "must not be actionable"),
        (lambda item: replace(item, authority="actuate"), "must be review_only"),
        (lambda item: replace(item, source_project="SCPN-CONTROL"), "source owner"),
        (lambda item: replace(item, source_schema="external.v1"), "allocated"),
        (lambda item: replace(item, source_revision="deadbeef"), "Git revision"),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        carrier_type=SemanticCarrier.CATEGORICAL_STATE,
                    ),
                ),
            ),
            "only bounded_feature",
        ),
        (
            lambda item: replace(
                item,
                semantics=(replace(item.semantics[0], amplitude=1.0),),
            ),
            "cannot carry phase fields",
        ),
        (
            lambda item: replace(
                item,
                regime=replace(
                    item.regime,
                    state=RegimeState.NOMINAL,
                    validity=ValidityWindow(
                        ValidityState.VALID,
                        valid_from_ns=20_000_000,
                        valid_until_ns=20_000_000,
                    ),
                ),
            ),
            "cannot infer a reactor regime",
        ),
    ],
)
def test_handoff_graph_refuses_semantic_and_authority_escalation(
    mutation,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        mutation(_handoff())


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda item: replace(item, schema="unknown.v1"), "unsupported.*schema"),
        (lambda item: replace(item, schema_version="1.1.0"), "unsupported.*version"),
        (
            lambda item: replace(
                item,
                context=replace(
                    item.context,
                    event_id="fusion.torax.event.other",
                ),
            ),
            "handoff and reactor context event_id must match",
        ),
        (lambda item: replace(item, observables=()), "requires observables"),
        (
            lambda item: replace(
                item,
                observables=(item.observables[0], item.observables[0]),
            ),
            "observable_ids must be unique",
        ),
        (
            lambda item: replace(
                item,
                observables=(
                    replace(
                        item.observables[0],
                        reactor_context=replace(
                            item.context,
                            context_id="fusion.torax.other.context",
                        ),
                    ),
                ),
            ),
            "observable context must match",
        ),
        (
            lambda item: replace(
                item,
                observables=(
                    replace(
                        item.observables[0],
                        provenance=replace(
                            item.observables[0].provenance,
                            source_project="SCPN-PHASE-ORCHESTRATOR",
                        ),
                    ),
                ),
            ),
            "retain FUSION provenance",
        ),
        (lambda item: replace(item, semantics=()), "one semantic record"),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        reactor_context_id="fusion.torax.other.context",
                    ),
                ),
            ),
            "semantic context must match",
        ),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        observable_ids=(
                            item.observables[0].observable_id,
                            "fusion.torax.other.observable",
                        ),
                    ),
                ),
            ),
            "must name one observable",
        ),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(item.semantics[0], clock_domain="fusion.torax.other"),
                ),
            ),
            "clock domains must match",
        ),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        clock_kind=ClockKind.PLANT_MONOTONIC,
                    ),
                ),
            ),
            "clock kinds must match",
        ),
        (
            lambda item: replace(
                item,
                semantics=(replace(item.semantics[0], clock_epoch="other.epoch"),),
            ),
            "clock epochs must match",
        ),
        (
            lambda item: replace(
                item,
                semantics=(replace(item.semantics[0], confidence=0.1),),
            ),
            "zero phase observability",
        ),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        validity=ValidityWindow(
                            ValidityState.UNKNOWN,
                            valid_from_ns=20_000_000,
                            valid_until_ns=20_000_000,
                            reasons=("unknown",),
                        ),
                    ),
                ),
            ),
            "requires UNOBSERVABLE validity",
        ),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        quality=QualityAssessment(QualityState.VALID),
                    ),
                ),
            ),
            "unknown phase quality",
        ),
        (
            lambda item: replace(
                item,
                semantics=(
                    replace(
                        item.semantics[0],
                        observable_ids=("fusion.torax.other.observable",),
                    ),
                ),
            ),
            "cover every observable",
        ),
        (
            lambda item: replace(
                item,
                regime=replace(
                    item.regime,
                    reactor_context_id="fusion.torax.other.context",
                ),
            ),
            "regime context must match",
        ),
        (
            lambda item: replace(
                item,
                regime=replace(item.regime, confidence=0.1),
            ),
            "regime confidence must be zero",
        ),
        (
            lambda item: replace(
                item,
                regime=replace(
                    item.regime,
                    evidence_ids=("fusion.torax.other.observable",),
                ),
            ),
            "evidence_ids must cover",
        ),
    ],
)
def test_handoff_graph_refuses_identity_shape_and_clock_drift(
    mutation,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        mutation(_handoff())


def test_handoff_graph_refuses_non_simulation_and_mixed_clocks() -> None:
    base = _handoff()
    plant_clock = replace(base.observables[0].clock, kind=ClockKind.PLANT_MONOTONIC)
    with pytest.raises(ValueError, match="simulation-monotonic clock"):
        replace(
            base,
            observables=(replace(base.observables[0], clock=plant_clock),),
            semantics=(
                replace(base.semantics[0], clock_kind=ClockKind.PLANT_MONOTONIC),
            ),
        )

    second_observable = replace(
        base.observables[0],
        observable_id="fusion.torax.electron_temperature",
        physical_quantity="electron temperature profile",
        channel="electron_temperature_kev",
        clock=replace(base.observables[0].clock, epoch="fusion.torax.other.epoch"),
    )
    second_semantic = replace(
        base.semantics[0],
        phase_id="spo.transport.electron_temperature.bounded_feature",
        observable_ids=(second_observable.observable_id,),
        clock_epoch=second_observable.clock.epoch,
    )
    with pytest.raises(ValueError, match="one clock domain and epoch"):
        replace(
            base,
            observables=(base.observables[0], second_observable),
            semantics=(base.semantics[0], second_semantic),
            regime=replace(
                base.regime,
                evidence_ids=(
                    base.observables[0].observable_id,
                    second_observable.observable_id,
                ),
            ),
        )


def test_handoff_graph_refuses_duplicate_semantics_and_phase_relations() -> None:
    base = _handoff()
    second_observable = replace(
        base.observables[0],
        observable_id="fusion.torax.electron_temperature",
        physical_quantity="electron temperature profile",
        channel="electron_temperature_kev",
    )
    duplicate_semantic = replace(
        base.semantics[0],
        observable_ids=(second_observable.observable_id,),
    )
    with pytest.raises(ValueError, match="semantic identifiers must be unique"):
        replace(
            base,
            observables=(base.observables[0], second_observable),
            semantics=(base.semantics[0], duplicate_semantic),
            regime=replace(
                base.regime,
                evidence_ids=(
                    base.observables[0].observable_id,
                    second_observable.observable_id,
                ),
            ),
        )

    reference = build_reactor_reference_portfolio()[0].semantics[0]
    relation = build_phase_relation(
        reference,
        replace(reference, phase_id="u0.a1.relation.target", phase_rad=0.7),
        relation_id="u0.a1.relation",
        relation_type=PhaseRelationType.SAME_MODE,
        interpretation=RelationInterpretation.CONTEXT_DEPENDENT,
        identification_method="test_fixture",
        evidence_class=EvidenceClass.SIMULATION,
    )
    with pytest.raises(ValueError, match="cannot contain phase relations"):
        replace(base, phase_relations=(relation,))


def test_handoff_defensively_refuses_tampered_regime_owner_and_authority() -> None:
    base = _handoff()
    regime = replace(base.regime)
    object.__setattr__(regime, "action_owner", "external")
    with pytest.raises(ValueError, match="action owner"):
        replace(base, regime=regime)

    regime = replace(base.regime)
    object.__setattr__(regime, "authority", "actuate")
    with pytest.raises(ValueError, match="regime authority"):
        replace(base, regime=regime)

    semantic = replace(base.semantics[0])
    object.__setattr__(
        semantic,
        "validity",
        ValidityWindow(
            ValidityState.UNKNOWN,
            valid_from_ns=20_000_000,
            valid_until_ns=20_000_000,
            reasons=("tampered",),
        ),
    )
    with pytest.raises(ValueError, match="must be unobservable"):
        replace(base, semantics=(semantic,))

    with pytest.raises(ValueError, match="unknown phase quality"):
        replace(
            base,
            semantics=(
                replace(
                    base.semantics[0],
                    quality=QualityAssessment(
                        QualityState.UNKNOWN,
                        flags=("different_unknown_reason",),
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "unknown.v1", "unsupported.*schema"),
        ("schema_version", "2.0.0", "unsupported.*version"),
        ("registry_version", "2.0.0", "unrecognised reactor registry release"),
        ("registry_digest", "0" * 64, "unrecognised reactor registry release"),
        ("u0_schema_version", "2.0.0", "U0 schema mismatch"),
    ],
)
def test_decoder_refuses_version_and_registry_drift(
    field: str,
    value: object,
    match: str,
) -> None:
    record = handoff_to_record(_handoff())
    if field in {"schema", "schema_version"}:
        record[field] = value
    else:
        payload = record["payload"]
        assert isinstance(payload, dict)
        payload[field] = value
        _reseal(record)
    with pytest.raises(ValueError, match=match):
        handoff_from_record(record)


@pytest.mark.parametrize("field", ["observables", "semantics", "phase_relations"])
def test_decoder_refuses_non_array_contract_collections(field: str) -> None:
    record = handoff_to_record(_handoff())
    payload = record["payload"]
    assert isinstance(payload, dict)
    payload[field] = {}
    _reseal(record)
    with pytest.raises(ValueError, match=f"{field} must be a list"):
        handoff_from_record(record)


def test_decoder_refuses_wrong_nested_contract_types() -> None:
    canonical = handoff_to_record(_handoff())
    canonical_payload = canonical["payload"]
    assert isinstance(canonical_payload, dict)
    context = canonical_payload["reactor_context"]
    regime = canonical_payload["regime"]

    for field, wrong, expected in (
        ("reactor_context", regime, "ReactorContext"),
        ("regime", context, "RegimeEstimate"),
        ("observables", [context], "ObservableDescriptor"),
        ("semantics", [context], "PhaseSemanticRecord"),
        ("phase_relations", [context], "PhaseRelation"),
    ):
        candidate = handoff_to_record(_handoff())
        payload = candidate["payload"]
        assert isinstance(payload, dict)
        payload[field] = wrong
        _reseal(candidate)
        with pytest.raises(ValueError, match=f"expected {expected}"):
            handoff_from_record(candidate)


def test_json_entrypoints_refuse_empty_invalid_duplicate_and_nonfinite_input() -> None:
    for value in ("", None):
        with pytest.raises(ValueError, match="non-empty string"):
            handoff_from_json(value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty string"):
            canonicalize_source_envelope(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="handoff JSON is invalid"):
        handoff_from_json("{")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        canonicalize_source_envelope('{"schema":"a","schema":"b"}')
    with pytest.raises(ValueError, match="JSON compliant"):
        canonicalize_source_envelope('{"value":NaN}')
    with pytest.raises(ValueError, match="source_envelope_json"):
        replace(_handoff(), source_envelope_json="")


def test_byte_entrypoint_refuses_noncanonical_non_utf8_and_wrong_types() -> None:
    canonical = handoff_to_bytes(_handoff())
    with pytest.raises(ValueError, match="canonical JSON"):
        handoff_from_bytes(canonical + b"\n")
    with pytest.raises(ValueError, match="strict UTF-8"):
        handoff_from_bytes(b"\xff")
    for value in (b"", "not-bytes"):
        with pytest.raises(ValueError, match="non-empty bytes"):
            handoff_from_bytes(value)  # type: ignore[arg-type]


def test_json_entrypoints_enforce_explicit_byte_bounds(monkeypatch) -> None:
    monkeypatch.setattr(handoff_module, "MAX_HANDOFF_JSON_BYTES", 1)
    with pytest.raises(ValueError, match="handoff JSON exceeds"):
        handoff_from_json("{}")
    with pytest.raises(ValueError, match="handoff bytes exceed"):
        handoff_from_bytes(b"{}")

    monkeypatch.setattr(handoff_module, "MAX_SOURCE_ENVELOPE_BYTES", 1)
    with pytest.raises(ValueError, match="source envelope JSON exceeds"):
        canonicalize_source_envelope("{}")


def _record_payload_digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _reseal(record: dict[str, object]) -> None:
    record["payload_sha256"] = _record_payload_digest(record["payload"])
