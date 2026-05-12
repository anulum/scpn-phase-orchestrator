# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy adapter boundary tests

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    HierarchyAdapterResult,
    HierarchySyncEnvelope,
    HierarchyTransportRuntime,
    build_hierarchy_sync_envelope,
    handle_hierarchy_frame,
    handle_hierarchy_rest_payload,
    replay_hierarchy_jsonl,
)


def _envelope(
    source_node: str,
    sequence: int,
    *,
    r: float = 0.8,
) -> HierarchySyncEnvelope:
    return build_hierarchy_sync_envelope(
        ChildSupervisorSummary(
            name=f"{source_node}-summary",
            channel="grid",
            R=r,
            psi=0.0,
            metadata={"region": "north"},
        ),
        source_node=source_node,
        sequence=sequence,
    )


def _envelope_record(
    source_node: str,
    sequence: int,
    *,
    r: float = 0.8,
) -> dict[str, object]:
    return _envelope(source_node, sequence, r=r).to_audit_record()


def test_jsonl_replay_is_deterministic_and_reports_stale_sequences() -> None:
    first = json.dumps(_envelope_record("edge-a", 1, r=0.2), sort_keys=True)
    latest = json.dumps(_envelope_record("edge-a", 2, r=0.9), sort_keys=True)

    left = replay_hierarchy_jsonl((first, latest))
    right = replay_hierarchy_jsonl((latest, first))

    assert isinstance(left, HierarchyAdapterResult)
    assert left.to_audit_record() == right.to_audit_record()
    audit = left.to_audit_record()
    assert audit["boundary"] == "jsonl_replay"
    assert audit["accepted_count"] == 1
    assert audit["rejected_count"] == 1
    assert audit["watermarks"] == {"edge-a": 2}
    assert audit["ledger"]["rejected"][0]["reason"] == "stale_or_duplicate_sequence"
    assert audit["parent_plan"]["R"] == pytest.approx(0.9)


def test_jsonl_replay_accepts_decoded_envelopes_and_mappings() -> None:
    result = replay_hierarchy_jsonl(
        (
            _envelope("edge-a", 1, r=0.7),
            _envelope_record("edge-b", 3, r=0.5),
        )
    )

    audit = result.to_audit_record()
    assert audit["boundary"] == "jsonl_replay"
    assert audit["accepted_count"] == 2
    assert audit["rejected_count"] == 0
    assert audit["watermarks"] == {"edge-a": 1, "edge-b": 3}
    assert [item["source_node"] for item in audit["ledger"]["accepted"]] == [
        "edge-a",
        "edge-b",
    ]


def test_jsonl_replay_rejects_blank_and_malformed_lines() -> None:
    with pytest.raises(ValueError, match="JSONL line 1 must not be blank"):
        replay_hierarchy_jsonl((" ",))

    with pytest.raises(ValueError, match="JSONL line 1 must be valid JSON"):
        replay_hierarchy_jsonl(("{not json}",))

    with pytest.raises(ValueError, match="JSONL line 1 must be a string"):
        replay_hierarchy_jsonl(([{"not": "a line"}],))

    with pytest.raises(ValueError, match="JSONL line 1 must decode to a mapping"):
        replay_hierarchy_jsonl(("[1, 2, 3]",))


def test_rest_boundary_rejects_non_json_content_type() -> None:
    with pytest.raises(ValueError, match="content-type must be application/json"):
        handle_hierarchy_rest_payload(
            {"envelope": _envelope_record("edge-a", 1)},
            headers={"content-type": "text/plain"},
        )

    with pytest.raises(ValueError, match="headers must be a decoded mapping"):
        handle_hierarchy_rest_payload(
            {"envelope": _envelope_record("edge-a", 1)},
            headers=[],
        )

    with pytest.raises(ValueError, match="content-type must be application/json"):
        handle_hierarchy_rest_payload(
            {"envelope": _envelope_record("edge-a", 1)},
            headers={"x-request-id": "sync-1"},
        )


def test_rest_boundary_accepts_single_and_batch_payloads_with_watermarks() -> None:
    runtime = HierarchyTransportRuntime()
    single = handle_hierarchy_rest_payload(
        {"envelope": _envelope("edge-a", 1)},
        headers={"Content-Type": "application/json; charset=utf-8"},
        runtime=runtime,
    )
    batch = handle_hierarchy_rest_payload(
        {"envelopes": [_envelope_record("edge-a", 1), _envelope_record("edge-b", 2)]},
        headers={"content-type": "application/json"},
        runtime=runtime,
    )

    assert single.to_audit_record()["watermarks"] == {"edge-a": 1}
    assert batch.to_audit_record()["accepted_count"] == 1
    assert batch.to_audit_record()["rejected_count"] == 1
    assert batch.to_audit_record()["watermarks"] == {"edge-a": 1, "edge-b": 2}


def test_rest_boundary_rejects_ambiguous_or_unknown_payload_shape() -> None:
    headers = {"content-type": "application/json"}
    with pytest.raises(ValueError, match="exactly one of envelope or envelopes"):
        handle_hierarchy_rest_payload(
            {
                "envelope": _envelope_record("edge-a", 1),
                "envelopes": [_envelope_record("edge-b", 1)],
            },
            headers=headers,
        )

    with pytest.raises(ValueError, match="REST payload contains unknown keys"):
        handle_hierarchy_rest_payload(
            {"envelope": _envelope_record("edge-a", 1), "socket": "no"},
            headers=headers,
        )


def test_rest_boundary_rejects_malformed_payload_shapes() -> None:
    headers = {"content-type": "application/json"}

    with pytest.raises(ValueError, match="REST payload must be a decoded mapping"):
        handle_hierarchy_rest_payload(["not", "a", "mapping"], headers=headers)

    with pytest.raises(ValueError, match="REST payload.envelope must be"):
        handle_hierarchy_rest_payload({"envelope": "not-an-envelope"}, headers=headers)

    with pytest.raises(ValueError, match="REST payload.envelopes must be a sequence"):
        handle_hierarchy_rest_payload({"envelopes": "not-a-sequence"}, headers=headers)

    with pytest.raises(ValueError, match="REST payload.envelopes must contain"):
        handle_hierarchy_rest_payload({"envelopes": []}, headers=headers)


def test_frame_boundary_validates_kind_and_payload() -> None:
    with pytest.raises(ValueError, match="frame must be a decoded mapping"):
        handle_hierarchy_frame(["not", "a", "frame"])

    with pytest.raises(ValueError, match="frame kind must be hierarchy_sync"):
        handle_hierarchy_frame({"kind": "open_socket", "payload": {}})

    with pytest.raises(ValueError, match="payload must be provided"):
        handle_hierarchy_frame({"kind": "hierarchy_sync"})

    with pytest.raises(ValueError, match="frame must not contain both kind and type"):
        handle_hierarchy_frame(
            {
                "kind": "hierarchy_sync",
                "type": "hierarchy_sync_batch",
                "payload": _envelope_record("edge-a", 1),
            }
        )

    with pytest.raises(ValueError, match="frame kind must be hierarchy_sync"):
        handle_hierarchy_frame({"kind": " ", "payload": _envelope_record("edge-a", 1)})


def test_frame_boundary_ingests_single_and_batch_frames() -> None:
    runtime = HierarchyTransportRuntime()
    single = handle_hierarchy_frame(
        {"kind": "hierarchy_sync", "payload": _envelope_record("edge-a", 1)},
        runtime=runtime,
    )
    batch = handle_hierarchy_frame(
        {
            "kind": "hierarchy_sync_batch",
            "payload": {
                "envelopes": [
                    _envelope_record("edge-a", 1),
                    _envelope_record("edge-b", 1),
                ]
            },
        },
        runtime=runtime,
    )

    assert single.to_audit_record()["boundary"] == "websocket_frame"
    assert single.to_audit_record()["frame_kind"] == "hierarchy_sync"
    assert batch.to_audit_record()["accepted_count"] == 1
    assert batch.to_audit_record()["rejected_count"] == 1
    assert batch.to_audit_record()["watermarks"] == {"edge-a": 1, "edge-b": 1}


def test_frame_boundary_accepts_direct_batch_sequence_and_rejects_bad_items() -> None:
    runtime = HierarchyTransportRuntime()
    batch = handle_hierarchy_frame(
        {
            "type": "hierarchy_sync_batch",
            "payload": [
                _envelope_record("edge-a", 2),
                _envelope_record("edge-b", 4),
            ],
        },
        runtime=runtime,
    )

    audit = batch.to_audit_record()
    assert audit["frame_kind"] == "hierarchy_sync_batch"
    assert audit["accepted_count"] == 2
    assert audit["rejected_count"] == 0
    assert audit["watermarks"] == {"edge-a": 2, "edge-b": 4}

    with pytest.raises(ValueError, match=r"frame payload\[0\] must be"):
        handle_hierarchy_frame({"kind": "hierarchy_sync_batch", "payload": [None]})


def test_hierarchy_adapters_do_not_import_network_modules() -> None:
    import scpn_phase_orchestrator.supervisor.hierarchy_adapters as adapters

    assert "socket" not in adapters.__dict__
    assert "asyncio" not in adapters.__dict__
    assert "http" not in adapters.__dict__


def test_frame_boundary_rejects_direction_and_capability_fields_before_ingest() -> None:
    runtime = HierarchyTransportRuntime()

    with pytest.raises(ValueError, match="frame contains unknown keys: capability"):
        handle_hierarchy_frame(
            {
                "kind": "hierarchy_sync",
                "capability": "open_network_transport",
                "payload": _envelope_record("edge-a", 1),
            },
            runtime=runtime,
        )

    with pytest.raises(ValueError, match="frame contains unknown keys: direction"):
        handle_hierarchy_frame(
            {
                "kind": "hierarchy_sync_batch",
                "direction": "parent_to_child_actuation",
                "payload": [_envelope_record("edge-a", 1)],
            },
            runtime=runtime,
        )

    assert runtime.previous_sequences == {}


def test_rest_boundary_reports_accepted_and_rejected_envelope_queues() -> None:
    runtime = HierarchyTransportRuntime()
    result = handle_hierarchy_rest_payload(
        {
            "envelopes": [
                _envelope_record("edge-a", 1, r=0.2),
                _envelope_record("edge-a", 1, r=0.9),
                _envelope_record("edge-b", 2, r=0.8),
            ]
        },
        headers={"content-type": "application/json"},
        runtime=runtime,
    )

    audit = result.to_audit_record()

    assert audit["accepted_count"] == 1
    assert audit["rejected_count"] == 2
    assert audit["watermarks"] == {"edge-b": 2}
    assert [item["source_node"] for item in audit["ledger"]["accepted"]] == [
        "edge-b"
    ]
    assert [item["reason"] for item in audit["ledger"]["rejected"]] == [
        "duplicate_sequence_conflict",
        "duplicate_sequence_conflict",
    ]
