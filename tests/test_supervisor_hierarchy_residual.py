# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hierarchy residual boundary tests

from __future__ import annotations

import json
import math

import pytest

from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    HierarchyTransportRuntime,
    build_hierarchical_orchestration_plan,
    build_hierarchy_sync_envelope,
    handle_hierarchy_frame,
    handle_hierarchy_rest_payload,
    ingest_hierarchy_sync_envelopes,
    load_hierarchy_sync_envelope,
    replay_hierarchy_jsonl,
    simulate_hierarchy_gossip_consensus,
)


def _summary(name: str = "edge-a", *, r: float = 0.8) -> ChildSupervisorSummary:
    return ChildSupervisorSummary(
        name=name,
        channel="grid",
        R=r,
        psi=0.25,
        metadata={"site": "north", "tier": ["edge", "validated"]},
    )


def _record(source: str, sequence: int, *, r: float = 0.8) -> dict[str, object]:
    return build_hierarchy_sync_envelope(
        _summary(f"{source}-summary", r=r),
        source_node=source,
        sequence=sequence,
        monotonic_time_s=float(sequence),
    ).to_audit_record()


def test_hierarchy_summary_metadata_is_immutable_json_safe_and_audit_sorted() -> None:
    summary = _summary()

    with pytest.raises(TypeError):
        summary.metadata["site"] = "south"  # type: ignore[index]

    audit = summary.to_audit_record()
    assert audit["metadata"] == {"site": "north", "tier": ["edge", "validated"]}
    assert audit["weighted_R"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "a",
                "sequence": 1,
            },
            "summary must be",
        ),
        ({1: "bad", "summary": {}}, "keys must be strings"),
        (_record("edge-a", 1) | {"raw_phases": [0.1, 0.2]}, "raw child evidence"),
        (_record("edge-a", 1) | {"sequence": True}, "sequence must be an integer"),
        (_record("edge-a", 1) | {"monotonic_time_s": -0.1}, "monotonic_time_s"),
        (
            _record("edge-a", 1)
            | {"summary": _record("edge-a", 1)["summary"] | {"raw_events": []}},
            "raw child evidence",
        ),
    ],
)
def test_load_hierarchy_sync_envelope_rejects_malformed_or_raw_public_records(
    record: dict[object, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_hierarchy_sync_envelope(record)  # type: ignore[arg-type]


def test_hierarchy_sync_envelope_json_is_deterministic_and_strictly_reloaded() -> None:
    envelope = build_hierarchy_sync_envelope(
        _summary(),
        source_node="edge-a",
        sequence=3,
        monotonic_time_s=12.5,
    )

    first = envelope.to_json()
    second = envelope.to_json()
    reloaded = load_hierarchy_sync_envelope(first)

    assert first == second
    assert list(json.loads(first)) == [
        "monotonic_time_s",
        "protocol_version",
        "sequence",
        "source_node",
        "summary",
    ]
    assert reloaded.to_audit_record() == envelope.to_audit_record()


def test_ingest_rejects_conflicting_duplicate_sequences() -> None:
    same_source_conflict = [
        load_hierarchy_sync_envelope(_record("edge-a", 2, r=0.3)),
        load_hierarchy_sync_envelope(_record("edge-a", 2, r=0.9)),
        load_hierarchy_sync_envelope(_record("edge-b", 1, r=0.7)),
    ]

    ledger = ingest_hierarchy_sync_envelopes(same_source_conflict)

    assert [item.source_node for item in ledger.accepted] == ["edge-b"]
    assert [item["reason"] for item in ledger.rejected] == [
        "duplicate_sequence_conflict",
        "duplicate_sequence_conflict",
    ]
    assert ledger.plan.parent_R == pytest.approx(0.7)


def test_transport_runtime_validates_watermarks_and_reports_sorted_state() -> None:
    with pytest.raises(ValueError, match="previous_sequences must be a mapping"):
        HierarchyTransportRuntime(previous_sequences=[("edge-a", 1)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sequence must be >= 0"):
        HierarchyTransportRuntime(previous_sequences={"edge-a": -1})

    runtime = HierarchyTransportRuntime(previous_sequences={"edge-b": 4, "edge-a": 1})
    assert runtime.to_audit_record()["previous_sequences"] == {"edge-a": 1, "edge-b": 4}


def test_hierarchy_adapters_reject_unknown_frame_keys_and_sort_watermarks() -> None:
    with pytest.raises(ValueError, match="frame contains unknown keys: socket"):
        handle_hierarchy_frame(
            {
                "kind": "hierarchy_sync",
                "payload": _record("edge-a", 1),
                "socket": "not-owned-here",
            }
        )

    result = handle_hierarchy_rest_payload(
        {"envelopes": [_record("edge-b", 2), _record("edge-a", 1)]},
        headers={"content-type": "application/json"},
    )

    audit = result.to_audit_record()
    assert audit["watermarks"] == {"edge-a": 1, "edge-b": 2}
    assert [item["source_node"] for item in audit["ledger"]["accepted"]] == [
        "edge-a",
        "edge-b",
    ]


def test_jsonl_and_gossip_public_boundaries_reject_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="JSONL line 1 must decode to a mapping"):
        replay_hierarchy_jsonl((json.dumps([_record("edge-a", 1)]),))
    with pytest.raises(ValueError, match="rounds must be >= 1"):
        simulate_hierarchy_gossip_consensus(
            [load_hierarchy_sync_envelope(_record("edge-a", 1))],
            neighbour_map={"edge-a": []},
            rounds=0,
        )
    with pytest.raises(ValueError, match="self_weight must be finite"):
        simulate_hierarchy_gossip_consensus(
            [load_hierarchy_sync_envelope(_record("edge-a", 1))],
            neighbour_map={"edge-a": []},
            self_weight=math.nan,
        )


def test_hierarchical_plan_rejects_non_finite_thresholds_before_parent_state() -> None:
    with pytest.raises(ValueError, match="degraded_threshold must be finite"):
        build_hierarchical_orchestration_plan(
            [_summary()],
            degraded_threshold=math.nan,
        )
