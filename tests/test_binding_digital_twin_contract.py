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
    DigitalTwinSyncEnvelope,
    DigitalTwinSyncMemoryAdapter,
    DigitalTwinSyncRestAdapter,
    build_digital_twin_adapter_manifest,
    build_digital_twin_binding_contract,
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
