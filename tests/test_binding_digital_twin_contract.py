# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin binding contract tests

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints

import pytest

from scpn_phase_orchestrator.binding import (
    DigitalTwinAdapterManifest,
    DigitalTwinBindingContract,
    DigitalTwinOperatorEvidence,
    DigitalTwinSyncEnvelope,
    DigitalTwinSyncGrpcAdapter,
    DigitalTwinSyncHardwareAdapter,
    DigitalTwinSyncKafkaAdapter,
    DigitalTwinSyncMemoryAdapter,
    DigitalTwinSyncRestAdapter,
    build_digital_twin_adapter_manifest,
    build_digital_twin_binding_contract,
    build_digital_twin_operator_evidence,
    build_digital_twin_sync_envelope,
    load_binding_spec,
    read_digital_twin_sync_jsonl,
    validate_digital_twin_sync_envelope,
    write_digital_twin_sync_jsonl,
)
from scpn_phase_orchestrator.binding.types import BindingSpec


def test_digital_twin_contract_builder_is_typed() -> None:
    hints = get_type_hints(build_digital_twin_binding_contract)

    assert hints["spec"] is BindingSpec
    assert hints["return"] is DigitalTwinBindingContract


def test_digital_twin_contract_serialises_binding_and_channel_contract() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")

    contract = build_digital_twin_binding_contract(spec)
    record = contract.to_audit_record()

    assert record["contract_version"] == "spo-digital-twin-binding/v1"
    assert record["binding"] == {
        "name": "digital_twin_nchannel",
        "version": "0.1.0",
        "safety_tier": "research",
    }
    assert record["timing"] == {"sample_period_s": 0.01, "control_period_s": 0.1}
    assert [layer["name"] for layer in record["layers"]] == [
        "machine_cells",
        "process_line",
        "twin_supervisor",
    ]
    assert [actuator["name"] for actuator in record["actuators"]] == [
        "line_phase_lag",
        "twin_coupling",
    ]
    channel_algebra = record["channel_algebra"]
    assert channel_algebra["declared_channels"] == [
        "I",
        "P",
        "Quality",
        "S",
        "Thermal",
        "TwinResidual",
    ]
    assert channel_algebra["derived_channels"] == ["TwinResidual"]
    assert "TwinResidual" not in channel_algebra["coupling_participating_channels"]
    assert {capability["name"] for capability in record["sync_capabilities"]} == {
        "state_snapshot",
        "phase_observation",
        "control_action_proposal",
        "audit_replay",
    }
    assert len(record["contract_hash"]) == 64


def test_digital_twin_contract_hash_is_deterministic_and_sensitive() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")

    default_contract = build_digital_twin_binding_contract(spec)
    repeated_contract = build_digital_twin_binding_contract(spec)
    custom_contract = build_digital_twin_binding_contract(
        spec,
        sync_capabilities=("state_snapshot",),
    )

    assert default_contract.contract_hash == repeated_contract.contract_hash
    assert default_contract.to_json() == repeated_contract.to_json()
    assert default_contract.contract_hash != custom_contract.contract_hash
    assert custom_contract.sync_capabilities[0].direction == "twin_to_spo"
    assert custom_contract.sync_capabilities[0].payload == "state_vector"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"contract_version": ""}, "contract_version must be a non-empty string"),
        (
            {"sync_capabilities": ()},
            "sync_capabilities must contain at least one capability",
        ),
        (
            {"sync_capabilities": ("",)},
            "sync capability must be a non-empty string",
        ),
    ],
)
def test_digital_twin_contract_rejects_invalid_contract_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    spec = load_binding_spec(
        Path("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    )

    with pytest.raises(ValueError, match=message):
        build_digital_twin_binding_contract(spec, **kwargs)  # type: ignore[arg-type]


def test_digital_twin_sync_envelope_accepts_declared_capability() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=7,
        payload={"layer": "machine_cells", "R": 0.91},
    )

    validation = validate_digital_twin_sync_envelope(contract, envelope)

    assert validation.accepted is True
    assert validation.reason == "accepted"
    assert envelope.to_audit_record()["contract_hash"] == contract.contract_hash
    assert envelope.to_json() == (
        '{"capability":"state_snapshot",'
        f'"contract_hash":"{contract.contract_hash}",'
        '"direction":"twin_to_spo",'
        '"payload":{"R":0.91,"layer":"machine_cells"},'
        '"sequence":7}'
    )


@pytest.mark.parametrize(
    ("envelope_kwargs", "reason"),
    [
        (
            {
                "contract_hash": "wrong",
                "capability": "state_snapshot",
                "direction": "twin_to_spo",
                "sequence": 1,
                "payload": {"R": 0.9},
            },
            "contract_hash_mismatch",
        ),
        (
            {
                "capability": "unknown_capability",
                "direction": "twin_to_spo",
                "sequence": 1,
                "payload": {"R": 0.9},
            },
            "capability_not_declared",
        ),
        (
            {
                "capability": "control_action_proposal",
                "direction": "twin_to_spo",
                "sequence": 1,
                "payload": {"knob": "K"},
            },
            "direction_not_allowed",
        ),
        (
            {
                "capability": "audit_replay",
                "direction": "spo_to_twin",
                "sequence": 1,
                "payload": {},
            },
            "payload_empty",
        ),
    ],
)
def test_digital_twin_sync_envelope_rejects_invalid_transport_payloads(
    envelope_kwargs: dict[str, object],
    reason: str,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    envelope = DigitalTwinSyncEnvelope(
        contract_hash=str(envelope_kwargs.get("contract_hash", contract.contract_hash)),
        capability=str(envelope_kwargs["capability"]),
        direction=str(envelope_kwargs["direction"]),
        sequence=int(envelope_kwargs["sequence"]),
        payload=envelope_kwargs["payload"],  # type: ignore[arg-type]
    )

    validation = validate_digital_twin_sync_envelope(contract, envelope)

    assert validation.accepted is False
    assert validation.reason == reason
    assert validation.to_audit_record()["envelope"]["capability"] == envelope.capability


def test_digital_twin_sync_envelope_rejects_negative_sequence() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)

    with pytest.raises(ValueError, match="sequence must be >= 0"):
        build_digital_twin_sync_envelope(
            contract,
            capability="state_snapshot",
            direction="twin_to_spo",
            sequence=-1,
            payload={"R": 0.9},
        )


def test_digital_twin_jsonl_adapter_round_trips_valid_envelopes(tmp_path) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    path = tmp_path / "sync.jsonl"
    first = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=1,
        payload={"R": 0.91},
    )
    second = build_digital_twin_sync_envelope(
        contract,
        capability="audit_replay",
        direction="spo_to_twin",
        sequence=2,
        payload={"event": "accepted"},
    )

    write_report = write_digital_twin_sync_jsonl(path, (first, second))
    read_report = read_digital_twin_sync_jsonl(contract, path)

    assert write_report.written == 2
    assert path.read_text(encoding="utf-8").splitlines() == [
        first.to_json(),
        second.to_json(),
    ]
    assert [validation.envelope.sequence for validation in read_report.accepted] == [
        1,
        2,
    ]
    assert read_report.rejected == ()
    audit = read_report.to_audit_record()
    assert audit["accepted_count"] == 2
    assert audit["rejected_count"] == 0


def test_digital_twin_jsonl_adapter_reports_rejected_lines(tmp_path) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    path = tmp_path / "sync.jsonl"
    valid = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=1,
        payload={"R": 0.91},
    )
    wrong_direction = build_digital_twin_sync_envelope(
        contract,
        capability="control_action_proposal",
        direction="twin_to_spo",
        sequence=2,
        payload={"knob": "K"},
    )
    path.write_text(
        "\n".join(
            [
                valid.to_json(),
                "{not-json}",
                '{"capability":"state_snapshot","sequence":"bad"}',
                wrong_direction.to_json(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = read_digital_twin_sync_jsonl(contract, path)

    assert [validation.envelope.sequence for validation in report.accepted] == [1]
    assert report.rejected == (
        {"line_number": 2, "reason": "malformed_json"},
        {"line_number": 3, "reason": "invalid_envelope"},
        {"line_number": 4, "reason": "direction_not_allowed"},
    )


def test_digital_twin_memory_adapter_queues_only_accepted_payloads() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncMemoryAdapter.for_contract(contract)
    accepted = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=3,
        payload={"R": 0.88},
    )
    rejected = build_digital_twin_sync_envelope(
        contract,
        capability="control_action_proposal",
        direction="twin_to_spo",
        sequence=4,
        payload={"knob": "K"},
    )

    accepted_validation = adapter.submit(accepted)
    rejected_validation = adapter.submit(rejected)

    assert accepted_validation.accepted is True
    assert rejected_validation.accepted is False
    assert rejected_validation.reason == "direction_not_allowed"
    assert adapter.to_audit_record() == {
        "contract_hash": contract.contract_hash,
        "queued_count": 1,
        "queued_sequences": [3],
    }
    assert adapter.drain() == (accepted,)
    assert adapter.drain() == ()


def test_digital_twin_adapter_manifest_reports_compatible_offline_adapter() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)

    compatibility = build_digital_twin_adapter_manifest(
        contract,
        name="jsonl-review",
        transport="jsonl",
        sync_capabilities=("state_snapshot", "audit_replay"),
        supports_replay=True,
        requires_auth=False,
        notes="offline replay adapter",
    )

    assert compatibility.compatible is True
    assert compatibility.reasons == ()
    assert compatibility.to_audit_record()["manifest"] == {
        "name": "jsonl-review",
        "transport": "jsonl",
        "sync_capabilities": ["state_snapshot", "audit_replay"],
        "supports_replay": True,
        "requires_auth": False,
        "notes": "offline replay adapter",
    }


def test_digital_twin_adapter_manifest_flags_live_transport_gates() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)

    compatibility = build_digital_twin_adapter_manifest(
        contract,
        name="grpc-live",
        transport="grpc",
        sync_capabilities=("state_snapshot", "not_declared"),
        supports_replay=True,
        requires_auth=False,
    )

    assert compatibility.compatible is False
    assert compatibility.reasons == (
        "capability_not_declared:not_declared",
        "live_transport_requires_auth",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "name": "",
                "transport": "memory",
                "sync_capabilities": ("state_snapshot",),
                "supports_replay": True,
                "requires_auth": False,
            },
            "adapter name must be a non-empty string",
        ),
        (
            {
                "name": "bad",
                "transport": "smtp",
                "sync_capabilities": ("state_snapshot",),
                "supports_replay": True,
                "requires_auth": False,
            },
            "adapter transport must be one of",
        ),
        (
            {
                "name": "bad",
                "transport": "memory",
                "sync_capabilities": (),
                "supports_replay": True,
                "requires_auth": False,
            },
            "adapter sync_capabilities must not be empty",
        ),
    ],
)
def test_digital_twin_adapter_manifest_rejects_invalid_manifest_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DigitalTwinAdapterManifest(**kwargs)  # type: ignore[arg-type]


def test_digital_twin_rest_adapter_accepts_authorised_contract_payloads() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncRestAdapter.for_contract(
        contract,
        sync_capabilities=("state_snapshot", "audit_replay"),
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=8,
        payload={"layer": "machine_cells", "R": 0.93},
    )

    response = adapter.handle_post(
        envelope.to_audit_record(),
        headers={"Authorization": "Bearer test-token"},
    )

    assert response.status_code == 202
    assert response.accepted is True
    assert response.reason == "accepted"
    assert response.body == {
        "capability": "state_snapshot",
        "sequence": 8,
        "contract_hash": contract.contract_hash,
    }
    assert adapter.to_audit_record()["queued_sequences"] == [8]
    assert adapter.drain() == (envelope,)
    assert adapter.drain() == ()


def test_digital_twin_rest_adapter_blocks_unauthorised_live_posts() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncRestAdapter.for_contract(contract)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=9,
        payload={"R": 0.89},
    )

    response = adapter.handle_post(envelope.to_audit_record(), headers={})

    assert response.status_code == 401
    assert response.accepted is False
    assert response.reason == "auth_required"
    assert adapter.drain() == ()


@pytest.mark.parametrize(
    ("body", "status_code", "reason"),
    [
        (
            {"capability": "state_snapshot", "sequence": "bad"},
            400,
            "invalid_envelope",
        ),
        (
            {
                "contract_hash": "wrong",
                "capability": "state_snapshot",
                "direction": "twin_to_spo",
                "sequence": 10,
                "payload": {"R": 0.9},
            },
            422,
            "contract_hash_mismatch",
        ),
        (
            {
                "capability": "control_action_proposal",
                "direction": "twin_to_spo",
                "sequence": 11,
                "payload": {"knob": "K"},
            },
            422,
            "direction_not_allowed",
        ),
    ],
)
def test_digital_twin_rest_adapter_reports_shape_and_contract_rejections(
    body: dict[str, object],
    status_code: int,
    reason: str,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    if "contract_hash" not in body:
        body = {"contract_hash": contract.contract_hash, **body}
    adapter = DigitalTwinSyncRestAdapter.for_contract(contract)

    response = adapter.handle_post(
        body,
        headers={"authorization": "Bearer test-token"},
    )

    assert response.status_code == status_code
    assert response.accepted is False
    assert response.reason == reason
    assert adapter.drain() == ()


def test_digital_twin_rest_adapter_refuses_incompatible_manifest() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncRestAdapter.for_contract(
        contract,
        sync_capabilities=("state_snapshot", "unknown"),
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=12,
        payload={"R": 0.92},
    )

    response = adapter.handle_post(
        envelope.to_audit_record(),
        headers={"authorization": "Bearer test-token"},
    )

    assert response.status_code == 503
    assert response.reason == "adapter_incompatible"
    assert response.body["reasons"] == ["capability_not_declared:unknown"]
    assert adapter.drain() == ()


def test_digital_twin_grpc_adapter_accepts_authorised_contract_requests() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncGrpcAdapter.for_contract(
        contract,
        sync_capabilities=("state_snapshot", "audit_replay"),
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="audit_replay",
        direction="spo_to_twin",
        sequence=13,
        payload={"event": "accepted", "source": "audit"},
    )

    response = adapter.handle_unary(
        envelope.to_audit_record(),
        metadata={"authorization": "Bearer grpc-token"},
    )

    assert response.status_code == "OK"
    assert response.accepted is True
    assert response.reason == "accepted"
    assert response.message == {
        "capability": "audit_replay",
        "sequence": 13,
        "contract_hash": contract.contract_hash,
    }
    assert adapter.to_audit_record()["queued_sequences"] == [13]
    assert adapter.drain() == (envelope,)
    assert adapter.drain() == ()


def test_digital_twin_grpc_adapter_blocks_unauthenticated_metadata() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncGrpcAdapter.for_contract(contract)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=14,
        payload={"R": 0.86},
    )

    response = adapter.handle_unary(envelope.to_audit_record(), metadata={})

    assert response.status_code == "UNAUTHENTICATED"
    assert response.accepted is False
    assert response.reason == "auth_required"
    assert response.message == {"contract_hash": contract.contract_hash}
    assert adapter.drain() == ()


@pytest.mark.parametrize(
    ("request_body", "status_code", "reason"),
    [
        (
            {"contract_hash": "wrong", "capability": "state_snapshot"},
            "INVALID_ARGUMENT",
            "invalid_envelope",
        ),
        (
            {
                "contract_hash": "wrong",
                "capability": "state_snapshot",
                "direction": "twin_to_spo",
                "sequence": 15,
                "payload": {"R": 0.82},
            },
            "FAILED_PRECONDITION",
            "contract_hash_mismatch",
        ),
        (
            {
                "capability": "control_action_proposal",
                "direction": "twin_to_spo",
                "sequence": 16,
                "payload": {"knob": "K"},
            },
            "FAILED_PRECONDITION",
            "direction_not_allowed",
        ),
    ],
)
def test_digital_twin_grpc_adapter_maps_request_failures_to_status_codes(
    request_body: dict[str, object],
    status_code: str,
    reason: str,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    if "contract_hash" not in request_body:
        request_body = {"contract_hash": contract.contract_hash, **request_body}
    adapter = DigitalTwinSyncGrpcAdapter.for_contract(contract)

    response = adapter.handle_unary(
        request_body,
        metadata={"authorization": "Bearer grpc-token"},
    )

    assert response.status_code == status_code
    assert response.accepted is False
    assert response.reason == reason
    assert adapter.drain() == ()


def test_digital_twin_grpc_adapter_refuses_incompatible_manifest() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncGrpcAdapter.for_contract(
        contract,
        requires_auth=False,
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=17,
        payload={"R": 0.95},
    )

    response = adapter.handle_unary(
        envelope.to_audit_record(),
        metadata={"authorization": "Bearer grpc-token"},
    )

    assert response.status_code == "FAILED_PRECONDITION"
    assert response.reason == "adapter_incompatible"
    assert response.message["reasons"] == ["live_transport_requires_auth"]
    assert adapter.drain() == ()


def test_digital_twin_kafka_adapter_accepts_authorised_message_values() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncKafkaAdapter.for_contract(
        contract,
        topic="spo.digital_twin.test",
        sync_capabilities=("phase_observation", "audit_replay"),
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="phase_observation",
        direction="twin_to_spo",
        sequence=18,
        payload={"layer": "process_line", "R": 0.84},
    )

    response = adapter.handle_message(
        {
            "topic": "spo.digital_twin.test",
            "key": "process_line",
            "value": envelope.to_audit_record(),
            "partition": 0,
            "offset": 101,
        },
        headers={"authorization": "Bearer kafka-token"},
    )

    assert response.accepted is True
    assert response.retryable is False
    assert response.reason == "accepted"
    assert response.message == {
        "topic": "spo.digital_twin.test",
        "capability": "phase_observation",
        "sequence": 18,
        "contract_hash": contract.contract_hash,
    }
    assert adapter.to_audit_record()["queued_sequences"] == [18]
    assert adapter.drain() == (envelope,)
    assert adapter.drain() == ()


def test_digital_twin_kafka_adapter_rejects_wrong_topic_without_queueing() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncKafkaAdapter.for_contract(
        contract,
        topic="spo.digital_twin.expected",
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=19,
        payload={"R": 0.81},
    )

    response = adapter.handle_message(
        {
            "topic": "spo.digital_twin.other",
            "value": envelope.to_audit_record(),
        },
        headers={"authorization": "Bearer kafka-token"},
    )

    assert response.accepted is False
    assert response.retryable is False
    assert response.reason == "topic_mismatch"
    assert response.message == {
        "expected_topic": "spo.digital_twin.expected",
        "observed_topic": "spo.digital_twin.other",
    }
    assert adapter.drain() == ()


@pytest.mark.parametrize(
    ("message", "reason", "retryable"),
    [
        (
            {"topic": "spo.digital_twin.sync", "value": "not-a-record"},
            "invalid_message_value",
            False,
        ),
        (
            {
                "topic": "spo.digital_twin.sync",
                "value": {"contract_hash": "wrong", "capability": "state_snapshot"},
            },
            "invalid_envelope",
            False,
        ),
        (
            {
                "topic": "spo.digital_twin.sync",
                "value": {
                    "contract_hash": "wrong",
                    "capability": "state_snapshot",
                    "direction": "twin_to_spo",
                    "sequence": 20,
                    "payload": {"R": 0.8},
                },
            },
            "contract_hash_mismatch",
            False,
        ),
    ],
)
def test_digital_twin_kafka_adapter_reports_message_failures(
    message: dict[str, object],
    reason: str,
    retryable: bool,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncKafkaAdapter.for_contract(contract)

    response = adapter.handle_message(
        message,
        headers={"authorization": "Bearer kafka-token"},
    )

    assert response.accepted is False
    assert response.reason == reason
    assert response.retryable is retryable
    assert adapter.drain() == ()


def test_digital_twin_kafka_adapter_auth_and_manifest_failures_are_retryable() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=21,
        payload={"R": 0.91},
    )
    message = {
        "topic": "spo.digital_twin.sync",
        "value": envelope.to_audit_record(),
    }
    auth_adapter = DigitalTwinSyncKafkaAdapter.for_contract(contract)
    incompatible_adapter = DigitalTwinSyncKafkaAdapter.for_contract(
        contract,
        requires_auth=False,
    )

    auth_response = auth_adapter.handle_message(message, headers={})
    incompatible_response = incompatible_adapter.handle_message(
        message,
        headers={"authorization": "Bearer kafka-token"},
    )

    assert auth_response.accepted is False
    assert auth_response.reason == "auth_required"
    assert auth_response.retryable is True
    assert incompatible_response.accepted is False
    assert incompatible_response.reason == "adapter_incompatible"
    assert incompatible_response.retryable is True
    assert incompatible_response.message["reasons"] == ["live_transport_requires_auth"]
    assert auth_adapter.drain() == ()
    assert incompatible_adapter.drain() == ()


def test_digital_twin_hardware_adapter_accepts_interlocked_authorised_frame() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncHardwareAdapter.for_contract(
        contract,
        device_ids=("pynq-loopback-0",),
        sync_capabilities=("state_snapshot", "audit_replay"),
    )
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=22,
        payload={"R": 0.87, "source": "loopback"},
    )

    response = adapter.handle_frame(
        {
            "device_id": "pynq-loopback-0",
            "safety_interlock": True,
            "value": envelope.to_audit_record(),
        },
        headers={"authorization": "Bearer hardware-token"},
    )

    assert response.accepted is True
    assert response.reason == "accepted"
    assert response.hardware_write_permitted is False
    assert response.frame == {
        "device_id": "pynq-loopback-0",
        "capability": "state_snapshot",
        "sequence": 22,
        "contract_hash": contract.contract_hash,
    }
    audit = adapter.to_audit_record()
    assert audit["device_ids"] == ["pynq-loopback-0"]
    assert audit["queued_sequences"] == [22]
    assert audit["hardware_write_permitted"] is False
    assert adapter.drain() == (envelope,)
    assert adapter.drain() == ()


@pytest.mark.parametrize(
    ("frame", "reason"),
    [
        (
            {
                "device_id": "unknown-device",
                "safety_interlock": True,
                "value": {},
            },
            "device_not_registered",
        ),
        (
            {
                "device_id": "pynq-loopback-0",
                "safety_interlock": False,
                "value": {},
            },
            "safety_interlock_required",
        ),
        (
            {
                "device_id": "pynq-loopback-0",
                "safety_interlock": True,
                "value": "not-a-record",
            },
            "invalid_frame_value",
        ),
        (
            {
                "device_id": "pynq-loopback-0",
                "safety_interlock": True,
                "value": {"contract_hash": "wrong", "capability": "state_snapshot"},
            },
            "invalid_envelope",
        ),
    ],
)
def test_digital_twin_hardware_adapter_rejects_unsafe_or_malformed_frames(
    frame: dict[str, object],
    reason: str,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    adapter = DigitalTwinSyncHardwareAdapter.for_contract(
        contract,
        device_ids=("pynq-loopback-0",),
    )

    response = adapter.handle_frame(
        frame,
        headers={"authorization": "Bearer hardware-token"},
    )

    assert response.accepted is False
    assert response.reason == reason
    assert response.hardware_write_permitted is False
    assert adapter.drain() == ()


def test_digital_twin_hardware_adapter_blocks_auth_and_contract_failures() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    envelope = build_digital_twin_sync_envelope(
        contract,
        capability="control_action_proposal",
        direction="twin_to_spo",
        sequence=23,
        payload={"knob": "K"},
    )
    frame = {
        "device_id": "pynq-loopback-0",
        "safety_interlock": True,
        "value": envelope.to_audit_record(),
    }
    auth_adapter = DigitalTwinSyncHardwareAdapter.for_contract(
        contract,
        device_ids=("pynq-loopback-0",),
    )
    incompatible_adapter = DigitalTwinSyncHardwareAdapter.for_contract(
        contract,
        device_ids=("pynq-loopback-0",),
        requires_auth=False,
    )

    auth_response = auth_adapter.handle_frame(frame, headers={})
    incompatible_response = incompatible_adapter.handle_frame(
        frame,
        headers={"authorization": "Bearer hardware-token"},
    )
    contract_response = auth_adapter.handle_frame(
        frame,
        headers={"authorization": "Bearer hardware-token"},
    )

    assert auth_response.reason == "auth_required"
    assert incompatible_response.reason == "adapter_incompatible"
    assert incompatible_response.frame["reasons"] == ["live_transport_requires_auth"]
    assert contract_response.reason == "direction_not_allowed"
    assert auth_response.hardware_write_permitted is False
    assert incompatible_response.hardware_write_permitted is False
    assert contract_response.hardware_write_permitted is False
    assert auth_adapter.drain() == ()
    assert incompatible_adapter.drain() == ()


@pytest.mark.parametrize(
    ("device_ids", "message"),
    [
        ((), "hardware device_ids must not be empty"),
        (("",), "hardware device_id must be a non-empty string"),
    ],
)
def test_digital_twin_hardware_adapter_rejects_invalid_device_registry(
    device_ids: tuple[str, ...],
    message: str,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)

    with pytest.raises(ValueError, match=message):
        DigitalTwinSyncHardwareAdapter.for_contract(
            contract,
            device_ids=device_ids,
        )


def test_digital_twin_jsonl_write_returns_empty_report_for_empty_batch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sync.jsonl"

    report = write_digital_twin_sync_jsonl(path, ())

    assert report.written == 0
    assert report.accepted == ()
    assert report.rejected == ()
    assert report.to_audit_record() == {
        "path": str(path),
        "written": 0,
        "accepted_count": 0,
        "rejected_count": 0,
        "accepted": [],
        "rejected": [],
    }
    assert path.read_text(encoding="utf-8") == ""


def test_digital_twin_jsonl_read_records_boolean_and_payload_type_rejections(
    tmp_path: Path,
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    path = tmp_path / "sync.jsonl"
    valid = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=7,
        payload={"R": 0.91},
    )
    wrong_hash = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=9,
        payload={"R": 0.89},
    )

    path.write_text(
        "\n".join(
            [
                "",  # parser skip path
                valid.to_json(),
                "{not-json}",
                (
                    '{"contract_hash":"'
                    + contract.contract_hash
                    + '","capability":"state_snapshot","direction":"twin_to_spo",'
                    + '"sequence":true,"payload":{"R":0.91}}'
                ),
                (
                    '{"contract_hash":"'
                    + contract.contract_hash
                    + '","capability":"state_snapshot","direction":"twin_to_spo",'
                    + '"sequence":1,"payload":[1,2]}'
                ),
                wrong_hash.to_json().replace(
                    contract.contract_hash,
                    "mismatch-hash",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = read_digital_twin_sync_jsonl(contract, path)

    assert [validation.envelope.sequence for validation in report.accepted] == [7]
    assert report.rejected == (
        {"line_number": 3, "reason": "malformed_json"},
        {"line_number": 4, "reason": "invalid_envelope"},
        {"line_number": 5, "reason": "invalid_envelope"},
        {"line_number": 6, "reason": "contract_hash_mismatch"},
    )


def test_digital_twin_grpc_kafka_and_hardware_responses_expose_audit_records() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)

    grpc_adapter = DigitalTwinSyncGrpcAdapter.for_contract(contract)
    grpc_envelope = build_digital_twin_sync_envelope(
        contract,
        capability="audit_replay",
        direction="spo_to_twin",
        sequence=30,
        payload={"event": "accepted"},
    )
    grpc_response = grpc_adapter.handle_unary(
        grpc_envelope.to_audit_record(),
        metadata={"authorization": "Bearer grpc"},
    )
    assert grpc_response.to_audit_record() == {
        "status_code": "OK",
        "accepted": True,
        "reason": "accepted",
        "message": {
            "capability": "audit_replay",
            "sequence": 30,
            "contract_hash": contract.contract_hash,
        },
    }

    kafka_adapter = DigitalTwinSyncKafkaAdapter.for_contract(contract)
    kafka_response = kafka_adapter.handle_message(
        {"topic": "spo.digital_twin.sync"},
        headers={"authorization": "Bearer kafka"},
    )
    assert kafka_response.to_audit_record() == {
        "accepted": False,
        "reason": "invalid_message_value",
        "retryable": False,
        "message": {"contract_hash": contract.contract_hash},
    }

    hardware_adapter = DigitalTwinSyncHardwareAdapter.for_contract(
        contract,
        device_ids=("pynq-loopback-0",),
    )
    hardware_response = hardware_adapter.handle_frame(
        {
            "device_id": "pynq-loopback-0",
            "safety_interlock": True,
            "value": "not-a-record",
        },
        headers={"authorization": "Bearer hardware"},
    )
    assert hardware_response.to_audit_record() == {
        "accepted": False,
        "reason": "invalid_frame_value",
        "hardware_write_permitted": False,
        "frame": {
            "device_id": "pynq-loopback-0",
            "contract_hash": contract.contract_hash,
        },
    }


def test_digital_twin_operator_evidence_summarises_live_and_replay_health(
    tmp_path: Path,
) -> None:
    hints = get_type_hints(build_digital_twin_operator_evidence)
    assert hints["return"] is DigitalTwinOperatorEvidence

    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    path = tmp_path / "sync.jsonl"
    accepted = build_digital_twin_sync_envelope(
        contract,
        capability="phase_observation",
        direction="twin_to_spo",
        sequence=41,
        payload={
            "layer": "twin_supervisor",
            "R": 0.94,
            "TwinResidual": -0.031,
        },
    )
    rejected = build_digital_twin_sync_envelope(
        contract,
        capability="control_action_proposal",
        direction="twin_to_spo",
        sequence=42,
        payload={"knob": "K"},
    )
    path.write_text(
        accepted.to_json() + "\n" + rejected.to_json() + "\n",
        encoding="utf-8",
    )
    replay_report = read_digital_twin_sync_jsonl(contract, path)
    rest_adapter = DigitalTwinSyncRestAdapter.for_contract(
        contract,
        sync_capabilities=("phase_observation",),
    )

    evidence = build_digital_twin_operator_evidence(
        contract,
        replay_report.accepted,
        rejected=replay_report.rejected,
        adapter_records=(rest_adapter.to_audit_record(),),
    )

    assert evidence.to_audit_record() == {
        "contract_hash": contract.contract_hash,
        "accepted_count": 1,
        "rejected_count": 1,
        "adapter_count": 1,
        "unhealthy_adapter_count": 0,
        "latest_sequence": 41,
        "capability_counts": {
            "audit_replay": 0,
            "control_action_proposal": 0,
            "phase_observation": 1,
            "state_snapshot": 0,
        },
        "direction_counts": {"twin_to_spo": 1},
        "max_abs_twin_residual": 0.031,
        "mismatch_reasons": ["direction_not_allowed"],
        "status": "degraded",
    }


def test_digital_twin_operator_evidence_marks_residual_and_adapter_status() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    nominal = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=50,
        payload={"twin_residual": 0.07},
    )
    critical = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=51,
        payload={"twin_residual": 0.23},
    )
    incompatible_adapter = DigitalTwinSyncGrpcAdapter.for_contract(
        contract,
        requires_auth=False,
    )

    warning_evidence = build_digital_twin_operator_evidence(
        contract,
        (validate_digital_twin_sync_envelope(contract, nominal),),
    )
    critical_evidence = build_digital_twin_operator_evidence(
        contract,
        (validate_digital_twin_sync_envelope(contract, critical),),
        adapter_records=(incompatible_adapter.to_audit_record(),),
    )

    assert warning_evidence.status == "warning"
    assert warning_evidence.max_abs_twin_residual == 0.07
    assert critical_evidence.status == "critical"
    assert critical_evidence.unhealthy_adapter_count == 1
    assert critical_evidence.latest_sequence == 51


def test_digital_twin_operator_evidence_rejects_invalid_residual_inputs() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    invalid_residual = build_digital_twin_sync_envelope(
        contract,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=60,
        payload={"TwinResidual": "bad"},
    )

    with pytest.raises(ValueError, match="TwinResidual"):
        build_digital_twin_operator_evidence(
            contract,
            (validate_digital_twin_sync_envelope(contract, invalid_residual),),
        )

    with pytest.raises(ValueError, match="residual_warning_threshold"):
        build_digital_twin_operator_evidence(
            contract,
            (),
            residual_warning_threshold=0.3,
            residual_critical_threshold=0.2,
        )


def test_digital_twin_contract_unknown_capability_is_bidirectional_and_auditable(
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(
        spec,
        sync_capabilities=("state_snapshot", "probe"),
    )
    probe = next(cap for cap in contract.sync_capabilities if cap.name == "probe")

    assert probe.direction == "bidirectional"
    assert probe.payload == "json_object"
    for direction in ("twin_to_spo", "spo_to_twin"):
        envelope = build_digital_twin_sync_envelope(
            contract,
            capability="probe",
            direction=direction,
            sequence=77,
            payload={"signal": 0.11},
        )
        validation = validate_digital_twin_sync_envelope(contract, envelope)

        assert validation.accepted is True
        assert validation.reason == "accepted"


def test_digital_twin_operator_evidence_respects_discrepancy_boundaries() -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    equal_warning = build_digital_twin_sync_envelope(
        contract,
        capability="phase_observation",
        direction="twin_to_spo",
        sequence=91,
        payload={"TwinResidual": 0.05},
    )
    above_warning = build_digital_twin_sync_envelope(
        contract,
        capability="phase_observation",
        direction="twin_to_spo",
        sequence=92,
        payload={"TwinResidual": 0.15},
    )
    above_critical = build_digital_twin_sync_envelope(
        contract,
        capability="phase_observation",
        direction="twin_to_spo",
        sequence=93,
        payload={"TwinResidual": 0.35},
    )

    at_warning = build_digital_twin_operator_evidence(
        contract,
        (validate_digital_twin_sync_envelope(contract, equal_warning),),
        residual_warning_threshold=0.05,
        residual_critical_threshold=0.2,
    )
    warning = build_digital_twin_operator_evidence(
        contract,
        (validate_digital_twin_sync_envelope(contract, above_warning),),
        residual_warning_threshold=0.05,
        residual_critical_threshold=0.5,
    )
    critical = build_digital_twin_operator_evidence(
        contract,
        (validate_digital_twin_sync_envelope(contract, above_critical),),
        residual_warning_threshold=0.05,
        residual_critical_threshold=0.2,
    )

    assert at_warning.status == "healthy"
    assert at_warning.max_abs_twin_residual == 0.05
    assert warning.status == "warning"
    assert critical.status == "critical"


def test_digital_twin_operator_evidence_marks_contract_hash_mismatch_as_rejected(
) -> None:
    spec = load_binding_spec("domainpacks/digital_twin_nchannel/binding_spec.yaml")
    contract = build_digital_twin_binding_contract(spec)
    contract_alt = build_digital_twin_binding_contract(
        spec,
        contract_version="spo-digital-twin-binding/v2",
    )
    mismatched = build_digital_twin_sync_envelope(
        contract_alt,
        capability="state_snapshot",
        direction="twin_to_spo",
        sequence=101,
        payload={"R": 0.91},
    )
    evidence = build_digital_twin_operator_evidence(
        contract,
        (validate_digital_twin_sync_envelope(contract, mismatched),),
    )

    assert evidence.accepted_count == 0
    assert evidence.rejected_count == 1
    assert evidence.mismatch_reasons == ("contract_hash_mismatch",)
    assert evidence.status == "degraded"
