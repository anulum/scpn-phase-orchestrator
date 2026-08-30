# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Public MIF source-adapter and U0 semantic tests."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
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
FIXTURE_SHA256 = "c780706abd5a0b185a95e85767e623248388664da61126d196fcb3d528b0c0ca"
SOURCE_BYTES = FIXTURE.read_bytes()


def _record() -> dict[str, object]:
    return json.loads(SOURCE_BYTES)


def _reseal(record: dict[str, object]) -> bytes:
    body = record["payload"]
    body_bytes = (
        json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    record["payload_sha256"] = hashlib.sha256(body_bytes).hexdigest()
    return (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _change(
    path: tuple[str, ...], value: object
) -> Callable[[dict[str, object]], None]:
    def mutate(record: dict[str, object]) -> None:
        target = record
        for key in path[:-1]:
            target = target[key]  # type: ignore[assignment]
        target[path[-1]] = value

    return mutate


def _delete(path: tuple[str, ...]) -> Callable[[dict[str, object]], None]:
    def mutate(record: dict[str, object]) -> None:
        target = record
        for key in path[:-1]:
            target = target[key]  # type: ignore[assignment]
        del target[path[-1]]

    return mutate


def test_fixture_maps_to_numeric_phase_without_physical_overclaim() -> None:
    source = SOURCE_BYTES
    handoff = rs.mif_merge_compression_handoff_from_mif_bytes(
        source, expected_sha256=FIXTURE_SHA256
    )

    assert len(source) == 2_475
    assert hashlib.sha256(source).hexdigest() == FIXTURE_SHA256
    assert handoff.source_envelope_sha256 == FIXTURE_SHA256
    assert handoff.context.configuration == "frc_compression_mif"
    assert handoff.context.confinement_family is rs.ConfinementFamily.MAGNETO_INERTIAL
    assert handoff.context.topology == "compressed field-reversed configuration"
    assert handoff.context.cadence is rs.OperatingCadence.PULSED_SHOT
    assert handoff.context.evidence_class is rs.EvidenceClass.SIMULATION
    assert handoff.authority == "review_only"
    assert handoff.actionable is False
    assert handoff.phase_relations == ()
    assert handoff.regime.state is rs.RegimeState.UNKNOWN
    assert handoff.regime.confidence == 0.0
    assert len(handoff.observables) == len(handoff.semantics) == 25

    numerical = [
        item
        for item in handoff.semantics
        if item.carrier_type is rs.SemanticCarrier.NUMERICAL_PHASE
    ]
    assert [item.phase_rad for item in numerical] == [0.0, 0.001]
    assert all(item.evidence_class is rs.EvidenceClass.SIMULATION for item in numerical)
    assert all(item.phase_origin == "event_start" for item in numerical)
    assert all(item.orientation == "positive_model_evolution" for item in numerical)
    assert all(item.wrap_convention == "[0,2pi)" for item in numerical)
    assert all(item.is_usable for item in numerical)

    nonphase = [item for item in handoff.semantics if item not in numerical]
    assert {item.carrier_type for item in nonphase} == {
        rs.SemanticCarrier.BOUNDED_FEATURE,
        rs.SemanticCarrier.CATEGORICAL_STATE,
    }
    assert all(item.phase_rad is None for item in nonphase)
    assert all(item.observability == item.confidence == 0.0 for item in nonphase)
    assert all(
        item.validity.state is rs.ValidityState.UNOBSERVABLE for item in nonphase
    )


def test_public_facade_exposes_distinct_mif_contract() -> None:
    assert (
        rs.MIF_MERGE_COMPRESSION_SOURCE_SCHEMA
        == "scpn-mif-core.merge-compression-observation.v1"
    )
    assert (
        rs.MIF_MERGE_COMPRESSION_HANDOFF_SCHEMA
        == "scpn-phase-orchestrator.mif-merge-compression-handoff.v1"
    )
    assert rs.MIFMergeCompressionHandoff.__module__.endswith("mif_merge_compression")


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        ("missing root", _delete(("event_id",))),
        ("unknown root", _change(("command",), "compress")),
        ("schema", _change(("schema",), "scpn-mif-core.other.v1")),
        ("version", _change(("schema_version",), "2.0.0")),
        ("project", _change(("source_project",), "SCPN-FUSION-CORE")),
        ("revision", _change(("source_revision",), "main")),
        ("event", _change(("event_id",), "not portable")),
        ("authority", _change(("payload", "authority", "actionable"), True)),
        ("configuration", _change(("payload", "reactor", "configuration"), "frc")),
        ("cadence", _change(("payload", "reactor", "cadence"), "steady")),
        ("frame", _change(("payload", "reactor", "coordinate_frame"), "bad frame")),
        ("reaction", _change(("payload", "reactor", "reaction"), "unknown")),
        ("conversion", _change(("payload", "reactor", "conversion"), "unknown")),
        (
            "drivers",
            _change(
                ("payload", "reactor", "drivers"),
                ["pulsed_power", "external_magnetic_coils"],
            ),
        ),
        ("clock kind", _change(("payload", "clock", "kind"), "wall_clock")),
        ("clock domain", _change(("payload", "clock", "domain"), "bad clock")),
        ("timestamp", _change(("payload", "clock", "timestamp_ns"), True)),
        ("period", _change(("payload", "clock", "sample_period_ns"), 0)),
        ("picosecond", _change(("payload", "clock", "picosecond_offset"), 1000)),
        ("latency", _change(("payload", "clock", "latency_s"), "-1")),
        ("rate", _change(("payload", "clock", "sample_rate_hz"), "999")),
        ("evidence class", _change(("payload", "evidence", "class"), "observed")),
        ("calibration", _change(("payload", "evidence", "calibration_id"), "physical")),
        (
            "backend version",
            _change(("payload", "evidence", "backend_version"), "main"),
        ),
        ("calibration time", _change(("payload", "evidence", "calibrated_at_ns"), 1)),
        ("input digest", _change(("payload", "evidence", "input_sha256"), ["bad"])),
        ("quality", _change(("payload", "evidence", "quality"), "unknown")),
        (
            "quality flags",
            _change(("payload", "evidence", "quality_flags"), ["clipped"]),
        ),
        ("phase list", _change(("payload", "kinematics", "phases_rad"), [])),
        ("phase range", _change(("payload", "kinematics", "phases_rad"), ["0", "7"])),
        ("velocity shape", _change(("payload", "kinematics", "velocities_m_s"), ["0"])),
        ("float", _change(("payload", "kinematics", "local_error_estimate"), 0.0)),
        ("nan", _change(("payload", "kinematics", "local_error_estimate"), "NaN")),
        ("order", _change(("payload", "kinematics", "order_parameter"), "2")),
        ("separation", _change(("payload", "kinematics", "separation_m"), "0")),
        ("phase lock", _change(("payload", "kinematics", "phase_lock_error_rad"), "0")),
        ("merge bool", _change(("payload", "merge_window", "candidate_lock"), 1)),
        (
            "merge tolerance",
            _change(("payload", "merge_window", "phase_tolerance_rad"), "0"),
        ),
        (
            "merge candidate",
            _change(("payload", "merge_window", "candidate_lock"), False),
        ),
        ("merge lock", _change(("payload", "merge_window", "lock_achieved"), False)),
        ("decision", _change(("payload", "trigger", "decision"), "compress")),
        ("trigger bool", _change(("payload", "trigger", "armed"), 1)),
        ("sample index", _change(("payload", "trigger", "sample_index"), -1)),
        (
            "fire timestamp",
            _change(("payload", "trigger", "first_fire_timestamp_ns"), None),
        ),
        ("safety slack", _change(("payload", "trigger", "safety_slack_m"), "-1")),
    ],
)
def test_source_adapter_refuses_semantic_drift(
    label: str, mutate: Callable[[dict[str, object]], None]
) -> None:
    record = _record()
    mutate(record)

    with pytest.raises((ValueError, KeyError), match=".+"):
        rs.mif_merge_compression_handoff_from_mif_bytes(_reseal(record))


@pytest.mark.parametrize(
    "kind", ["empty", "noncanonical", "duplicate", "utf8", "digest"]
)
def test_source_adapter_refuses_byte_custody_drift(kind: str) -> None:
    source = SOURCE_BYTES
    if kind == "empty":
        changed = b""
    elif kind == "noncanonical":
        changed = source + b"\n"
    elif kind == "duplicate":
        changed = source.replace(b'{"event_id":', b'{"event_id":"x","event_id":', 1)
    elif kind == "utf8":
        changed = b"\xff"
    else:
        changed = source

    with pytest.raises(ValueError):
        rs.mif_merge_compression_handoff_from_mif_bytes(
            changed,
            expected_sha256="f" * 64 if kind == "digest" else None,
        )


def test_source_adapter_refuses_payload_digest_and_container_types() -> None:
    record = _record()
    record["payload"]["trigger"]["sample_index"] = 7  # type: ignore[index]
    changed = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()
    with pytest.raises(ValueError, match="payload digest mismatch"):
        rs.mif_merge_compression_handoff_from_mif_bytes(changed)

    with pytest.raises(ValueError, match="must be an object"):
        rs.mif_merge_compression_handoff_from_mif_bytes(b"[]\n")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("payload", "evidence", "input_sha256"), []),
        (("payload", "evidence", "input_sha256"), ["0" * 64, "0" * 64]),
        (("payload", "evidence", "quality_flags"), ["x", "x"]),
        (("payload", "reactor", "drivers"), "pulsed_power"),
        (("payload", "reactor", "drivers"), [1]),
        (("payload", "kinematics", "local_error_estimate"), "bad"),
        (("payload", "kinematics", "local_error_estimate"), 0),
        (("payload", "trigger", "first_violation_index"), -1),
        (("payload", "trigger", "decision"), "hold_no_lock"),
    ],
)
def test_source_adapter_refuses_remaining_scalar_and_list_drift(
    path: tuple[str, ...], value: object
) -> None:
    record = _record()
    target = record
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        rs.mif_merge_compression_handoff_from_mif_bytes(_reseal(record))


@pytest.mark.parametrize(
    ("quality", "flags", "usable"),
    [
        ("degraded", ["model_warning"], True),
        ("invalid", ["model_invalid"], False),
    ],
)
def test_source_quality_maps_without_physical_phase_overclaim(
    quality: str, flags: list[str], usable: bool
) -> None:
    record = _record()
    evidence = record["payload"]["evidence"]  # type: ignore[index]
    evidence["quality"] = quality
    evidence["quality_flags"] = flags
    handoff = rs.mif_merge_compression_handoff_from_mif_bytes(_reseal(record))
    numerical = handoff.semantics[:2]
    assert [item.is_usable for item in numerical] == [usable, usable]
    assert all(item.evidence_class is rs.EvidenceClass.SIMULATION for item in numerical)
    if not usable:
        assert all(item.phase_rad is None for item in numerical)


def test_valid_nonfire_source_remains_categorical_not_actuation() -> None:
    record = _record()
    trigger = record["payload"]["trigger"]  # type: ignore[index]
    trigger["decision"] = "hold_no_lock"
    trigger["armed"] = False
    trigger["first_fire_timestamp_ns"] = None
    handoff = rs.mif_merge_compression_handoff_from_mif_bytes(_reseal(record))
    decision = next(
        item
        for item in handoff.semantics
        if item.phase_id.endswith("trigger.decision.categorical_state")
    )
    assert decision.carrier_type is rs.SemanticCarrier.CATEGORICAL_STATE
    assert decision.phase_rad is None


def test_source_size_limit_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mmc, "MAX_SOURCE_ENVELOPE_BYTES", 1)
    with pytest.raises(ValueError, match="maximum byte size"):
        rs.mif_merge_compression_handoff_from_mif_bytes(SOURCE_BYTES)
