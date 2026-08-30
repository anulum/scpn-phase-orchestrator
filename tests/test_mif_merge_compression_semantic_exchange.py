# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Portable MIF semantic-handoff exchange tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import scpn_phase_orchestrator.reactor_semantics as rs

mmc = importlib.import_module(
    "scpn_phase_orchestrator.reactor_semantics.mif_merge_compression"
)

FIXTURE = (
    Path(__file__).parent
    / "fixtures/mif_merge_compression/mif_merge_compression_observation_v1.json"
)
SOURCE_BYTES = FIXTURE.read_bytes()


def _handoff() -> rs.MIFMergeCompressionHandoff:
    return rs.mif_merge_compression_handoff_from_mif_bytes(SOURCE_BYTES)


def _reseal(record: dict[str, object]) -> bytes:
    payload = record["payload"]
    canonical = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()
    record["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return json.dumps(
        record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode()


def test_handoff_round_trip_is_canonical_and_digest_sealed() -> None:
    handoff = _handoff()
    record = rs.mif_merge_compression_handoff_to_record(handoff)
    encoded = rs.mif_merge_compression_handoff_to_bytes(handoff)
    decoded = rs.mif_merge_compression_handoff_from_bytes(encoded)

    assert (
        encoded
        == json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
    )
    assert not encoded.endswith(b"\n")
    assert decoded == handoff
    assert rs.mif_merge_compression_handoff_from_record(record) == handoff
    assert (
        rs.mif_merge_compression_handoff_digest(handoff)
        == hashlib.sha256(encoded).hexdigest()
    )
    assert record["payload"]["source_envelope_sha256"] == handoff.source_envelope_sha256  # type: ignore[index]
    assert record["payload"]["phase_relations"] == []  # type: ignore[index]
    assert record["payload"]["actionable"] is False  # type: ignore[index]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema",), "scpn-phase-orchestrator.other.v1"),
        (("schema_version",), "2.0.0"),
        (("payload_sha256",), "f" * 64),
        (("payload", "registry_version"), "2.0.0"),
        (("payload", "registry_digest"), "f" * 64),
        (("payload", "u0_schema_version"), "2.0.0"),
        (("payload", "source_envelope_sha256"), "f" * 64),
        (("payload", "source_project"), "SCPN-FUSION-CORE"),
        (("payload", "source_schema"), "scpn-mif-core.other.v1"),
        (("payload", "source_revision"), "0" * 40),
        (("payload", "event_id"), "another_event"),
        (("payload", "authority"), "action"),
        (("payload", "actionable"), True),
        (("payload", "phase_relations"), [{}]),
        (("payload", "observables"), []),
        (("payload", "semantics"), []),
    ],
)
def test_handoff_decoder_refuses_digest_identity_and_authority_drift(
    path: tuple[str, ...], value: object
) -> None:
    record = rs.mif_merge_compression_handoff_to_record(_handoff())
    target = record
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    changed = (
        json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode()
        if path == ("payload_sha256",)
        else _reseal(record)
    )

    with pytest.raises(ValueError):
        rs.mif_merge_compression_handoff_from_bytes(changed)


@pytest.mark.parametrize("kind", ["empty", "whitespace", "duplicate", "utf8", "array"])
def test_handoff_decoder_refuses_noncanonical_bytes(kind: str) -> None:
    encoded = rs.mif_merge_compression_handoff_to_bytes(_handoff())
    if kind == "empty":
        changed = b""
    elif kind == "whitespace":
        changed = encoded + b"\n"
    elif kind == "duplicate":
        changed = encoded.replace(b'{"payload":', b'{"payload":{},"payload":', 1)
    elif kind == "utf8":
        changed = b"\xff"
    else:
        changed = b"[]"
    with pytest.raises(ValueError):
        rs.mif_merge_compression_handoff_from_bytes(changed)


def test_handoff_graph_refuses_semantic_and_context_escalation() -> None:
    handoff = _handoff()
    nonphase = next(
        item
        for item in handoff.semantics
        if item.carrier_type is rs.SemanticCarrier.BOUNDED_FEATURE
    )
    promoted = replace(
        nonphase,
        carrier_type=rs.SemanticCarrier.CYCLIC_PHASE,
        validity=rs.ValidityWindow(rs.ValidityState.VALID, 0, 0),
        quality=rs.QualityAssessment(rs.QualityState.VALID),
        phase_rad=0.1,
        phase_origin="invented",
        orientation="invented",
        wrap_convention="[0,2pi)",
        reference_signal="invented",
        observability=1.0,
        confidence=1.0,
    )
    semantics = tuple(
        promoted if item is nonphase else item for item in handoff.semantics
    )
    with pytest.raises(ValueError, match="only numerical phase or nonphase"):
        replace(handoff, semantics=semantics)
    with pytest.raises(ValueError):
        replace(handoff, context=replace(handoff.context, configuration="tokamak"))
    with pytest.raises(ValueError, match="review-only"):
        replace(handoff, actionable=True)


def test_runtime_module_has_no_mif_or_control_dependency() -> None:
    module = Path(rs.__file__).with_name("mif_merge_compression.py").read_text()
    assert "import scpn_mif_core" not in module
    assert "import scpn_control" not in module
    assert "ControlAction" not in module


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "scpn-phase-orchestrator.other.v1"),
        ("schema_version", "2.0.0"),
        ("source_revision", "main"),
    ],
)
def test_handoff_object_refuses_outer_identity_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(_handoff(), **{field: value})


def test_handoff_object_refuses_complete_graph_drift() -> None:
    handoff = _handoff()
    context = handoff.context
    first_observable = handoff.observables[0]
    first_semantic = handoff.semantics[0]

    frc = rs.DEFAULT_REACTOR_REGISTRY.resolve("field_reversed_configuration")
    other_context = replace(
        context,
        configuration=frc.identifier,
        confinement_family=frc.confinement_family,
        topology=frc.topology,
    )
    with pytest.raises(ValueError, match="frc_compression_mif"):
        replace(handoff, context=other_context)

    event_context = replace(context, event_id="another_event")
    with pytest.raises(ValueError, match="context event_id"):
        replace(handoff, context=event_context)

    duplicate_observables = (
        first_observable,
        first_observable,
        *handoff.observables[2:],
    )
    with pytest.raises(ValueError, match="observable_ids must be unique"):
        replace(handoff, observables=duplicate_observables)

    foreign_context_observable = replace(
        first_observable, reactor_context=event_context
    )
    with pytest.raises(ValueError, match="observable context"):
        replace(
            handoff,
            observables=(foreign_context_observable, *handoff.observables[1:]),
        )

    fusion_provenance = replace(
        first_observable.provenance, source_project="SCPN-FUSION-CORE"
    )
    fusion_observable = replace(first_observable, provenance=fusion_provenance)
    with pytest.raises(ValueError, match="retain MIF provenance"):
        replace(handoff, observables=(fusion_observable, *handoff.observables[1:]))

    shifted_clock = replace(first_observable.clock, timestamp_ns=1)
    shifted_observable = replace(first_observable, clock=shifted_clock)
    with pytest.raises(ValueError, match="one exact clock"):
        replace(handoff, observables=(shifted_observable, *handoff.observables[1:]))

    model_clock = replace(first_observable.clock, kind=rs.ClockKind.MODEL_TICK)
    model_observables = tuple(
        replace(item, clock=model_clock) for item in handoff.observables
    )
    with pytest.raises(ValueError, match="simulation-monotonic"):
        replace(handoff, observables=model_observables)

    duplicate_semantics = (first_semantic, first_semantic, *handoff.semantics[2:])
    with pytest.raises(ValueError, match="semantic identifiers"):
        replace(handoff, semantics=duplicate_semantics)

    foreign_semantic = replace(first_semantic, reactor_context_id="spo.other.context")
    with pytest.raises(ValueError, match="semantic context"):
        replace(handoff, semantics=(foreign_semantic, *handoff.semantics[1:]))

    multi_source_semantic = replace(
        first_semantic,
        observable_ids=(
            first_semantic.observable_ids[0],
            handoff.observables[1].observable_id,
        ),
    )
    with pytest.raises(ValueError, match="name one observable"):
        replace(handoff, semantics=(multi_source_semantic, *handoff.semantics[1:]))

    shifted_semantic = replace(first_semantic, clock_domain="another_clock")
    with pytest.raises(ValueError, match="clocks must match"):
        replace(handoff, semantics=(shifted_semantic, *handoff.semantics[1:]))

    wrong_source_semantic = replace(
        first_semantic, observable_ids=("mif.unmapped.observable",)
    )
    with pytest.raises(ValueError, match="cover every observable"):
        replace(handoff, semantics=(wrong_source_semantic, *handoff.semantics[1:]))

    with pytest.raises(ValueError, match="phase relations"):
        replace(handoff, phase_relations=((),))  # type: ignore[arg-type]

    foreign_regime = replace(handoff.regime, reactor_context_id="spo.other.context")
    with pytest.raises(ValueError, match="regime context"):
        replace(handoff, regime=foreign_regime)

    inferred_regime = replace(
        handoff.regime,
        state=rs.RegimeState.NOMINAL,
        confidence=1.0,
        validity=rs.ValidityWindow(rs.ValidityState.VALID, 0, 0),
    )
    with pytest.raises(ValueError, match="remain UNKNOWN"):
        replace(handoff, regime=inferred_regime)

    incomplete_regime = replace(
        handoff.regime, evidence_ids=handoff.regime.evidence_ids[:-1]
    )
    with pytest.raises(ValueError, match="cover every observable"):
        replace(handoff, regime=incomplete_regime)


def test_handoff_object_refuses_mutated_nested_authority_and_semantics() -> None:
    handoff = _handoff()
    regime = replace(handoff.regime)
    object.__setattr__(regime, "action_owner", "SCPN-MIF-CORE")
    with pytest.raises(ValueError, match="action owner"):
        replace(handoff, regime=regime)

    regime = replace(handoff.regime)
    object.__setattr__(regime, "authority", "action")
    with pytest.raises(ValueError, match="authority"):
        replace(handoff, regime=regime)

    numerical = replace(handoff.semantics[0])
    object.__setattr__(numerical, "evidence_class", rs.EvidenceClass.OBSERVED)
    with pytest.raises(ValueError, match="simulation evidence"):
        replace(handoff, semantics=(numerical, *handoff.semantics[1:]))

    numerical = replace(handoff.semantics[0], orientation="negative")
    with pytest.raises(ValueError, match="reference semantics"):
        replace(handoff, semantics=(numerical, *handoff.semantics[1:]))

    index = next(
        i
        for i, item in enumerate(handoff.semantics)
        if item.carrier_type is rs.SemanticCarrier.BOUNDED_FEATURE
    )
    nonphase = replace(handoff.semantics[index])
    object.__setattr__(nonphase, "phase_rad", 0.1)
    semantics = list(handoff.semantics)
    semantics[index] = nonphase
    with pytest.raises(ValueError, match="cannot carry phase fields"):
        replace(handoff, semantics=tuple(semantics))

    nonphase = replace(handoff.semantics[index], observability=1.0, confidence=1.0)
    semantics[index] = nonphase
    with pytest.raises(ValueError, match="zero phase observability"):
        replace(handoff, semantics=tuple(semantics))

    nonphase = replace(handoff.semantics[index])
    object.__setattr__(
        nonphase,
        "validity",
        rs.ValidityWindow(rs.ValidityState.VALID, 0, 0),
    )
    semantics[index] = nonphase
    with pytest.raises(ValueError, match="UNOBSERVABLE"):
        replace(handoff, semantics=tuple(semantics))


def test_handoff_decoder_refuses_size_list_and_contract_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = rs.mif_merge_compression_handoff_to_bytes(_handoff())
    monkeypatch.setattr(mmc, "MAX_MIF_MERGE_COMPRESSION_HANDOFF_BYTES", 1)
    with pytest.raises(ValueError, match="maximum byte size"):
        rs.mif_merge_compression_handoff_from_bytes(encoded)
    monkeypatch.undo()

    record = rs.mif_merge_compression_handoff_to_record(_handoff())
    record["payload"]["observables"] = {}  # type: ignore[index]
    with pytest.raises(ValueError, match="observables must be a list"):
        rs.mif_merge_compression_handoff_from_bytes(_reseal(record))

    record = rs.mif_merge_compression_handoff_to_record(_handoff())
    payload = record["payload"]
    payload["reactor_context"] = payload["observables"][0]  # type: ignore[index]
    with pytest.raises(ValueError, match="expected ReactorContext"):
        rs.mif_merge_compression_handoff_from_bytes(_reseal(record))
