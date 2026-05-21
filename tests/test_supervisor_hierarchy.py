from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping

import pytest

from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    HierarchySyncEnvelope,
    HierarchyTransportRuntime,
    build_hierarchical_orchestration_plan,
    build_hierarchy_sync_envelope,
    ingest_hierarchy_sync_envelopes,
    load_hierarchy_sync_envelope,
    simulate_hierarchy_gossip_consensus,
)
from scpn_phase_orchestrator.supervisor.hierarchy import (
    _child_escalations,
    _is_forbidden_hierarchy_key,
    _load_child_summary,
    _load_mapping_record,
    _metadata_to_audit_record,
    _normalise_metadata_value,
    _normalise_previous_sequences,
    _reject_raw_hierarchy_keys,
    _reject_raw_instance_attributes,
    _reject_unknown_keys,
)


def test_hierarchical_plan_composes_reduced_child_summaries_only() -> None:
    children = (
        ChildSupervisorSummary(
            name="edge-a",
            channel="power",
            R=0.9,
            psi=0.0,
            metadata={"region": "north"},
        ),
        ChildSupervisorSummary(
            name="edge-b",
            channel="thermal",
            R=0.8,
            psi=0.0,
            confidence=0.5,
        ),
    )

    plan = build_hierarchical_orchestration_plan(children)

    assert plan.parent_state.regime_id == "hierarchical_nominal"
    assert plan.parent_state.stability_proxy == pytest.approx((0.9 + 0.4) / 2)
    assert [layer.R for layer in plan.parent_state.layers] == pytest.approx([0.9, 0.4])
    assert plan.parent_R == pytest.approx(0.65)
    assert plan.parent_psi == pytest.approx(0.0)
    assert plan.escalations == ()

    audit = plan.to_audit_record()
    assert audit["audit_scope"] == "reduced_child_summaries_only"
    child_record = audit["children"][0]
    assert child_record == {
        "name": "edge-a",
        "channel": "power",
        "R": 0.9,
        "psi": 0.0,
        "regime": "nominal",
        "confidence": 1.0,
        "weighted_R": 0.9,
        "metadata": {"region": "north"},
    }
    assert "raw_phases" not in child_record
    assert "time_series" not in child_record
    assert "knm" not in child_record


def test_hierarchical_plan_accepts_single_pass_child_iterable() -> None:
    child = ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0)

    plan = build_hierarchical_orchestration_plan(iter((child,)))  # type: ignore[arg-type]

    assert [item.name for item in plan.children] == ["edge-a"]
    assert len(plan.parent_state.layers) == 1
    assert plan.parent_R == pytest.approx(0.9)


def test_hierarchical_plan_escalates_degraded_critical_and_low_confidence() -> None:
    children = (
        ChildSupervisorSummary(
            name="child-critical",
            channel="grid",
            R=0.2,
            psi=math.pi,
            regime="critical",
        ),
        ChildSupervisorSummary(
            name="child-degraded",
            channel="traffic",
            R=0.5,
            psi=0.5,
        ),
        ChildSupervisorSummary(
            name="child-uncertain",
            channel="cardiac",
            R=0.9,
            psi=0.1,
            confidence=0.2,
        ),
    )

    plan = build_hierarchical_orchestration_plan(
        children,
        degraded_threshold=0.65,
        critical_threshold=0.35,
        min_confidence=0.5,
    )

    assert plan.parent_state.regime_id == "hierarchical_critical"
    assert [(item.child, item.severity, item.reason) for item in plan.escalations] == [
        ("child-critical", "critical", "child_coherence_below_critical"),
        ("child-degraded", "degraded", "child_coherence_below_degraded"),
        ("child-uncertain", "degraded", "child_summary_below_min_confidence"),
    ]
    assert plan.to_audit_record()["escalations"][0]["child_regime"] == "critical"


def test_hierarchical_plan_escalates_declared_regime_even_when_coherent() -> None:
    children = (
        ChildSupervisorSummary(
            name="critical-reporter",
            channel="grid",
            R=0.9,
            psi=0.0,
            regime="critical-local-actuator-clamped",
        ),
        ChildSupervisorSummary(
            name="degraded-reporter",
            channel="thermal",
            R=0.8,
            psi=0.0,
            regime="degraded-observer-confidence",
        ),
    )

    plan = build_hierarchical_orchestration_plan(children)

    assert plan.parent_state.regime_id == "hierarchical_nominal"
    assert [(item.child, item.severity, item.reason) for item in plan.escalations] == [
        ("critical-reporter", "critical", "child_regime_escalation"),
        ("degraded-reporter", "degraded", "child_regime_escalation"),
    ]


def test_hierarchical_plan_zero_confidence_children_produce_zero_parent_order() -> None:
    plan = build_hierarchical_orchestration_plan(
        (
            ChildSupervisorSummary(
                "silent-a",
                "grid",
                R=0.9,
                psi=0.0,
                confidence=0.0,
            ),
            ChildSupervisorSummary(
                "silent-b",
                "thermal",
                R=0.8,
                psi=math.pi,
                confidence=0.0,
            ),
        )
    )

    assert [layer.R for layer in plan.parent_state.layers] == [0.0, 0.0]
    assert plan.parent_R == 0.0
    assert plan.parent_psi == 0.0
    assert plan.parent_state.regime_id == "hierarchical_critical"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"children": ()}, "children must contain at least one child summary"),
        (
            {
                "children": (ChildSupervisorSummary("child", "channel", 0.7, 0.0),),
                "critical_threshold": 0.8,
                "degraded_threshold": 0.6,
            },
            "critical_threshold must be <= degraded_threshold",
        ),
    ],
)
def test_hierarchical_plan_rejects_invalid_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_hierarchical_orchestration_plan(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"name": "", "channel": "grid", "R": 0.9, "psi": 0.0},
            "name must be a non-empty string",
        ),
        (
            {"name": "child", "channel": "grid", "R": 1.2, "psi": 0.0},
            "R must be finite and in \\[0, 1\\]",
        ),
        (
            {
                "name": "child",
                "channel": "grid",
                "R": 0.9,
                "psi": 0.0,
                "confidence": -0.1,
            },
            "confidence must be finite and in \\[0, 1\\]",
        ),
    ],
)
def test_child_summary_rejects_invalid_reduced_evidence(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ChildSupervisorSummary(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "raw_key",
    [
        "raw_phases",
        "graph",
        "event",
        "events",
        "raw_signal",
        "raw_signals",
        "actuator",
        "actuators",
        "raw_actuator",
        "raw_actuators",
        "raw_actuator_target",
        "raw_actuator_targets",
        "raw_evidence",
        "child_evidence",
        "raw_observation",
        "raw_observations",
        "raw_event",
        "raw_events",
        "raw_coupling_matrix",
        "raw_time_series",
        "raw_phase",
        "raw_phase_history",
        "raw_timeseries",
        "raw_graph",
        "raw_coupling",
        "coupling",
        "couplings",
        "evidence",
        "local_coupling_matrix",
        "signal",
        "signals",
    ],
)
def test_child_summary_rejects_raw_metadata(raw_key: str) -> None:
    with pytest.raises(ValueError, match=raw_key):
        ChildSupervisorSummary(
            "edge-a",
            "power",
            R=0.8,
            psi=0.0,
            metadata={raw_key: [0.1, 0.2]},
        )


@pytest.mark.parametrize(
    "raw_key",
    ["raw_phase_history", "raw_phase", "raw_timeseries"],
)
def test_child_summary_rejects_generic_raw_metadata_aliases(raw_key: str) -> None:
    with pytest.raises(ValueError, match=raw_key):
        ChildSupervisorSummary(
            "edge-a",
            "power",
            R=0.8,
            psi=0.0,
            metadata={raw_key: [0.1, 0.2]},
        )

    with pytest.raises(ValueError, match=raw_key):
        ChildSupervisorSummary(
            "edge-a",
            "power",
            R=0.8,
            psi=0.0,
            metadata={"nested": {raw_key: [0.1, 0.2]}},
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"site": object()},
        {"too_large": 2**60},
        {"too_negative": -(2**60)},
        {"huge": 10**100},
        {"first_unsafe": 9007199254740992},
    ],
)
def test_child_summary_rejects_non_json_safe_metadata(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ChildSupervisorSummary(
            "edge-a",
            "power",
            R=0.8,
            psi=0.0,
            metadata=metadata,
        )


def test_child_summary_metadata_is_defensively_normalised() -> None:
    metadata = {"site": "north", "nested": {"role": "edge"}}
    summary = ChildSupervisorSummary(
        "edge-a",
        "power",
        R=0.8,
        psi=0.0,
        metadata=metadata,
    )

    metadata["raw_phases"] = [0.1, 0.2]
    metadata["nested"]["time_series"] = [0.1, 0.2]  # type: ignore[index]

    audit = summary.to_audit_record()

    assert audit["metadata"] == {"site": "north", "nested": {"role": "edge"}}
    json.dumps(audit, allow_nan=False)


def test_hierarchy_sync_envelopes_ingest_deterministically() -> None:
    edge_b = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-b", "thermal", R=0.8, psi=0.0),
        source_node="node-b",
        sequence=3,
        monotonic_time_s=12.5,
    )
    edge_a = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=2,
    )

    ledger = ingest_hierarchy_sync_envelopes((edge_b, edge_a))

    assert [envelope.source_node for envelope in ledger.accepted] == [
        "node-a",
        "node-b",
    ]
    assert ledger.rejected == ()
    assert ledger.plan.parent_state.regime_id == "hierarchical_nominal"
    assert ledger.plan.to_audit_record()["audit_scope"] == (
        "reduced_child_summaries_only"
    )
    assert edge_b.to_json() == (
        '{"monotonic_time_s":12.5,'
        '"protocol_version":"spo-hierarchy-sync/v1",'
        '"sequence":3,'
        '"source_node":"node-b",'
        '"summary":{"R":0.8,'
        '"channel":"thermal",'
        '"confidence":1.0,'
        '"metadata":{},'
        '"name":"edge-b",'
        '"psi":0.0,'
        '"regime":"nominal",'
        '"weighted_R":0.8}}'
    )


def test_hierarchy_sync_ingestion_rejects_stale_and_wrong_protocol() -> None:
    stale = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=1,
    )
    wrong_protocol = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-b", "traffic", R=0.4, psi=1.0),
        source_node="node-b",
        sequence=5,
        protocol_version="spo-hierarchy-sync/v0",
    )
    accepted = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-c", "grid", R=0.2, psi=math.pi),
        source_node="node-c",
        sequence=6,
    )

    ledger = ingest_hierarchy_sync_envelopes(
        (stale, wrong_protocol, accepted),
        previous_sequences={"node-a": 1},
    )

    assert [record["reason"] for record in ledger.rejected] == [
        "stale_or_duplicate_sequence",
        "protocol_version_mismatch",
    ]
    assert [envelope.source_node for envelope in ledger.accepted] == ["node-c"]
    assert ledger.plan.escalations[0].reason == "child_coherence_below_critical"


def test_hierarchy_sync_ingestion_requires_one_accepted_envelope() -> None:
    stale = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=1,
    )

    with pytest.raises(
        ValueError,
        match="at least one hierarchy sync envelope must be accepted",
    ):
        ingest_hierarchy_sync_envelopes(
            (stale,),
            previous_sequences={"node-a": 1},
        )


def test_hierarchy_sync_ingestion_rejects_empty_batch_after_config_check() -> None:
    with pytest.raises(
        ValueError,
        match="at least one hierarchy sync envelope must be accepted",
    ):
        ingest_hierarchy_sync_envelopes(())


def test_hierarchy_sync_ingestion_accepts_latest_same_source() -> None:
    sequence_2 = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-seq-2", "power", R=0.2, psi=0.0),
        source_node="node-a",
        sequence=2,
    )
    sequence_3 = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-seq-3", "power", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=3,
    )

    forward = ingest_hierarchy_sync_envelopes((sequence_2, sequence_3))
    reverse = ingest_hierarchy_sync_envelopes((sequence_3, sequence_2))

    assert [envelope.sequence for envelope in forward.accepted] == [3]
    assert [envelope.sequence for envelope in reverse.accepted] == [3]
    assert (
        forward.rejected
        == reverse.rejected
        == (
            {
                "source_node": "node-a",
                "sequence": 2,
                "reason": "stale_or_duplicate_sequence",
                "protocol_version": "spo-hierarchy-sync/v1",
            },
        )
    )
    assert forward.plan.to_audit_record() == reverse.plan.to_audit_record()


def test_hierarchy_sync_ingestion_collapses_identical_duplicate_sequence_entries() -> (
    None
):
    duplicate_a = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.3, psi=0.0),
        source_node="node-a",
        sequence=7,
    )
    duplicate_b = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.3, psi=0.0),
        source_node="node-a",
        sequence=7,
    )

    ledger = ingest_hierarchy_sync_envelopes((duplicate_a, duplicate_b))

    assert [envelope.source_node for envelope in ledger.accepted] == ["node-a"]
    assert [record["reason"] for record in ledger.rejected] == [
        "stale_or_duplicate_sequence",
    ]


def test_hierarchy_sync_ingestion_rejects_equal_sequence_conflicts() -> None:
    conflict_low = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-conflict-low", "power", R=0.2, psi=0.0),
        source_node="node-a",
        sequence=5,
    )
    conflict_high = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-conflict-high", "power", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=5,
    )
    stable = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-stable", "grid", R=0.7, psi=0.0),
        source_node="node-b",
        sequence=1,
    )

    forward = ingest_hierarchy_sync_envelopes((conflict_low, conflict_high, stable))
    reverse = ingest_hierarchy_sync_envelopes((conflict_high, conflict_low, stable))

    assert [envelope.source_node for envelope in forward.accepted] == ["node-b"]
    assert [envelope.source_node for envelope in reverse.accepted] == ["node-b"]
    assert [record["reason"] for record in forward.rejected] == [
        "duplicate_sequence_conflict",
        "duplicate_sequence_conflict",
    ]
    assert forward.rejected == reverse.rejected
    assert forward.plan.to_audit_record() == reverse.plan.to_audit_record()


def test_hierarchy_transport_runtime_ingests_json_and_mapping_batch() -> None:
    runtime = HierarchyTransportRuntime()
    json_record = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-b", "thermal", R=0.8, psi=0.0),
        source_node="node-b",
        sequence=3,
        monotonic_time_s=12.5,
    ).to_json()
    mapping_record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 2,
        "summary": {
            "name": "edge-a",
            "channel": "power",
            "R": 0.2,
            "psi": math.pi,
            "regime": "critical",
            "confidence": 1.0,
            "metadata": {"region": "north"},
        },
    }

    ledger = runtime.ingest((json_record, mapping_record))

    assert [envelope.source_node for envelope in ledger.accepted] == [
        "node-a",
        "node-b",
    ]
    assert ledger.plan.hierarchy == "edge_cloud_summary_sync"
    assert ledger.plan.escalations[0].child == "edge-a"
    assert ledger.plan.escalations[0].reason == "child_coherence_below_critical"
    assert runtime.previous_sequences == {"node-a": 2, "node-b": 3}
    assert runtime.to_audit_record() == {
        "hierarchy": "edge_cloud_summary_sync",
        "protocol_version": "spo-hierarchy-sync/v1",
        "previous_sequences": {"node-a": 2, "node-b": 3},
        "audit_scope": "reduced_child_summaries_only",
    }


def test_hierarchy_transport_runtime_rejects_inverted_thresholds() -> None:
    with pytest.raises(
        ValueError,
        match="critical_threshold must be <= degraded_threshold",
    ):
        HierarchyTransportRuntime(
            critical_threshold=0.8,
            degraded_threshold=0.6,
        )


def test_hierarchy_transport_runtime_rejects_invalid_min_confidence() -> None:
    with pytest.raises(
        ValueError,
        match="min_confidence must be finite and in \\[0, 1\\]",
    ):
        HierarchyTransportRuntime(min_confidence=-0.1)

    with pytest.raises(
        ValueError,
        match="min_confidence must be finite and in \\[0, 1\\]",
    ):
        HierarchyTransportRuntime(min_confidence=1.5)


def test_hierarchy_transport_runtime_rejects_stale_followup_only() -> None:
    runtime = HierarchyTransportRuntime()
    runtime.ingest(
        (
            build_hierarchy_sync_envelope(
                ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0),
                source_node="node-a",
                sequence=2,
            ),
        )
    )

    ledger = runtime.ingest(
        (
            build_hierarchy_sync_envelope(
                ChildSupervisorSummary("edge-a", "power", R=0.1, psi=0.0),
                source_node="node-a",
                sequence=2,
            ),
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-b",
                "sequence": 1,
                "summary": {
                    "name": "edge-b",
                    "channel": "grid",
                    "R": 0.7,
                    "psi": 0.2,
                    "regime": "nominal",
                    "confidence": 0.9,
                    "metadata": {},
                },
            },
        )
    )

    assert [record["reason"] for record in ledger.rejected] == [
        "stale_or_duplicate_sequence"
    ]
    assert [envelope.source_node for envelope in ledger.accepted] == ["node-b"]
    assert runtime.previous_sequences == {"node-a": 2, "node-b": 1}


def test_hierarchy_transport_runtime_rejects_thresholds_outside_unit_interval() -> None:
    with pytest.raises(
        ValueError,
        match="degraded_threshold must be finite and in \\[0, 1\\]",
    ):
        HierarchyTransportRuntime(degraded_threshold=1.2)

    with pytest.raises(
        ValueError,
        match="critical_threshold must be finite and in \\[0, 1\\]",
    ):
        HierarchyTransportRuntime(critical_threshold=-0.2)


def test_load_mapping_record_parses_json_and_rejects_invalid_record_types() -> None:
    record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 1,
        "summary": {"name": "edge-a", "channel": "power", "R": 0.9, "psi": 0.0},
    }

    loaded = _load_mapping_record(json.dumps(record))
    assert loaded == record

    with pytest.raises(ValueError, match="hierarchy sync envelope must be valid JSON"):
        _load_mapping_record('{"protocol_version":}')

    with pytest.raises(
        ValueError,
        match="hierarchy sync envelope must be a mapping or JSON string",
    ):
        _load_mapping_record(1)


def test_child_escalations_reports_confidence_and_coherence_reasons() -> None:
    child = ChildSupervisorSummary(
        "edge-a",
        "grid",
        R=0.2,
        psi=0.0,
        confidence=0.2,
        regime="critical-observed",
    )

    escalations = _child_escalations(
        child,
        degraded_threshold=0.65,
        critical_threshold=0.35,
        min_confidence=0.5,
    )

    assert [item.reason for item in escalations] == [
        "child_summary_below_min_confidence",
        "child_coherence_below_critical",
    ]
    assert escalations[1].severity == "critical"


def test_child_escalations_reports_declared_degraded_regime() -> None:
    child = ChildSupervisorSummary(
        "edge-a",
        "grid",
        R=0.9,
        psi=0.0,
        confidence=1.0,
        regime="degraded-observer",
    )

    escalations = _child_escalations(
        child,
        degraded_threshold=0.65,
        critical_threshold=0.35,
        min_confidence=0.5,
    )

    assert [(item.severity, item.reason) for item in escalations] == [
        ("degraded", "child_regime_escalation"),
    ]


def test_load_child_summary_defaults_regime_and_confidence() -> None:
    summary = _load_child_summary(
        {
            "name": "edge-a",
            "channel": "grid",
            "R": 0.8,
            "psi": 0.1,
        },
    )
    assert summary.regime == "nominal"
    assert summary.confidence == 1.0
    assert summary.metadata == {}


def test_load_child_summary_rejects_unknown_fields() -> None:
    with pytest.raises(
        ValueError,
        match="hierarchy sync summary contains unknown fields",
    ):
        _load_child_summary(
            {
                "name": "edge-a",
                "channel": "grid",
                "R": 0.8,
                "psi": 0.0,
                "unknown": "bad",
            },
        )


def test_load_child_summary_rejects_json_unsafe_metadata() -> None:
    with pytest.raises(ValueError, match="contains raw child evidence: raw_phases"):
        _load_child_summary(
            {
                "name": "edge-a",
                "channel": "grid",
                "R": 0.8,
                "psi": 0.0,
                "metadata": {"raw_phases": [0.1, 0.2]},
            },
        )


def test_reject_raw_hierarchy_keys_enforces_child_evidence_filter() -> None:
    _reject_raw_hierarchy_keys({"name": "node-a", "channel": "grid"}, "summary")
    with pytest.raises(
        ValueError,
        match="contains raw child evidence: raw_phase_history",
    ):
        _reject_raw_hierarchy_keys({"raw_phase_history": [0.1]}, "summary")


def test_reject_unknown_keys_rejects_bad_key_types_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="keys must be strings"):
        _reject_unknown_keys({1: "x"}, allowed=frozenset({"name"}), location="summary")

    with pytest.raises(
        ValueError,
        match="summary contains unknown fields: extra",
    ):
        _reject_unknown_keys(
            {"extra": 1},
            allowed=frozenset({"name"}),
            location="summary",
        )


def test_normalise_metadata_value_converts_nested_containers() -> None:
    value = _normalise_metadata_value(
        {"site": {"region": ["north", "west"]}, "active": True},
        "metadata",
    )
    assert isinstance(value["site"], Mapping)
    assert isinstance(value["site"]["region"], tuple)


def test_normalise_metadata_value_rejects_json_unsafe_scalar() -> None:
    with pytest.raises(ValueError, match="must be JSON-safe metadata"):
        _normalise_metadata_value((2**60), "metadata")


def test_metadata_to_audit_record_returns_plain_python_containers() -> None:
    assert _metadata_to_audit_record(
        {"site": ("north", "south"), "samples": [1, (2, 3)]},
    ) == {
        "site": ["north", "south"],
        "samples": [1, [2, 3]],
    }


def test_is_forbidden_hierarchy_key_distinguishes_clean_keys() -> None:
    assert _is_forbidden_hierarchy_key("raw_phase_history")
    assert _is_forbidden_hierarchy_key("child_evidence")
    assert not _is_forbidden_hierarchy_key("coherence")
    assert not _is_forbidden_hierarchy_key("phase_count")


def test_normalise_previous_sequences_checks_mapping_contract() -> None:
    assert _normalise_previous_sequences({"node-a": 3}) == {"node-a": 3}

    with pytest.raises(ValueError, match="source_node must be a non-empty string"):
        _normalise_previous_sequences({"": 1})

    with pytest.raises(ValueError, match="sequence must be >= 0"):
        _normalise_previous_sequences({"node-a": -1})


@pytest.mark.parametrize(
    "record",
    [
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "raw_child_observations": [0.1, 0.2],
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1.0,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": True,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": True,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
                "metadata": ["not", "a", "mapping"],
            },
        },
    ],
)
def test_load_hierarchy_sync_envelope_rejects_malformed_payloads(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        load_hierarchy_sync_envelope(record)

    with pytest.raises(ValueError):
        load_hierarchy_sync_envelope(json.dumps(record))


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ('{"protocol_version":', "hierarchy sync envelope must be valid JSON"),
        ("[]", "hierarchy sync envelope JSON must decode to a mapping"),
        (object(), "hierarchy sync envelope must be a mapping or JSON string"),
    ],
)
def test_load_hierarchy_sync_envelope_rejects_non_mapping_transport_records(
    record: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_hierarchy_sync_envelope(record)


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-a",
                "sequence": 1,
                "monotonic_time_s": -0.1,
                "summary": {
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            },
            "monotonic_time_s must be finite and non-negative",
        ),
        (
            {
                "protocol_version": "",
                "source_node": "node-a",
                "sequence": 1,
                "summary": {
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            },
            "protocol_version must be a non-empty string",
        ),
        (
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "   ",
                "sequence": 1,
                "summary": {
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            },
            "source_node must be a non-empty string",
        ),
        (
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-a",
                "sequence": 1,
                "summary": {
                    "name": "",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            },
            "name must be a non-empty string",
        ),
    ],
)
def test_load_hierarchy_sync_envelope_rejects_invalid_sync_contract_fields(
    record: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_hierarchy_sync_envelope(record)


def test_load_hierarchy_sync_envelope_rejects_non_string_decoded_keys() -> None:
    with pytest.raises(
        ValueError,
        match="hierarchy sync envelope keys must be strings",
    ):
        load_hierarchy_sync_envelope(
            {
                1: "node-a",
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-a",
                "sequence": 1,
                "summary": {
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            }
        )

    with pytest.raises(ValueError, match="hierarchy sync summary keys must be strings"):
        load_hierarchy_sync_envelope(
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-a",
                "sequence": 1,
                "summary": {
                    1: "edge-a",
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            }
        )


def test_load_hierarchy_sync_envelope_rejects_adversarial_raw_transport_keys() -> None:
    record = _TwoPassRawEnvelopeMapping()

    with pytest.raises(ValueError, match="raw_phases"):
        load_hierarchy_sync_envelope(record)


@pytest.mark.parametrize(
    "raw_key",
    [
        "phase",
        "phases",
        "observation",
        "observations",
        "raw_child_observations",
        "raw_phases",
        "time_series",
        "actuator_target",
        "actuator_targets",
        "coupling_matrix",
        "coupling_matrices",
        "graph",
        "event",
        "events",
        "raw_signal",
        "raw_signals",
        "raw_observation",
        "raw_observations",
        "raw_event",
        "raw_events",
        "raw_coupling_matrix",
        "raw_time_series",
        "raw_phase",
        "raw_phase_history",
        "raw_timeseries",
        "raw_graph",
        "raw_coupling",
        "coupling",
        "couplings",
        "evidence",
        "local_coupling_matrix",
        "signal",
        "signals",
    ],
)
def test_load_hierarchy_sync_envelope_rejects_raw_metadata_keys(
    raw_key: str,
) -> None:
    record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 1,
        "summary": {
            "name": "edge-a",
            "channel": "power",
            "R": 0.8,
            "psi": 0.0,
            "metadata": {
                "site": "north",
                "role": "edge",
                "nested": {raw_key: [0.1, 0.2]},
            },
        },
    }

    with pytest.raises(ValueError, match=raw_key):
        load_hierarchy_sync_envelope(record)

    with pytest.raises(ValueError, match=raw_key):
        load_hierarchy_sync_envelope(json.dumps(record))


@pytest.mark.parametrize(
    "raw_key",
    ["raw_phase_history", "raw_phase", "raw_timeseries"],
)
def test_load_hierarchy_sync_envelope_rejects_generic_raw_metadata_aliases(
    raw_key: str,
) -> None:
    record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 1,
        "summary": {
            "name": "edge-a",
            "channel": "power",
            "R": 0.8,
            "psi": 0.0,
            "metadata": {
                "site": "north",
                "nested": {raw_key: [0.1, 0.2]},
            },
        },
    }

    with pytest.raises(ValueError, match=raw_key):
        load_hierarchy_sync_envelope(record)

    with pytest.raises(ValueError, match=raw_key):
        load_hierarchy_sync_envelope(json.dumps(record))


@pytest.mark.parametrize(
    "record",
    [
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "raw_phase_history": [0.1, 0.2],
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "unknown": "ignored-before",
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
                "phase_history": [0.1, 0.2],
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
                "unknown": "ignored-before",
            },
        },
    ],
)
def test_load_hierarchy_sync_envelope_rejects_unknown_decoded_fields(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        load_hierarchy_sync_envelope(record)

    with pytest.raises(ValueError):
        load_hierarchy_sync_envelope(json.dumps(record))


@pytest.mark.parametrize(
    "raw_key",
    [
        "graph",
        "event",
        "events",
        "raw_signal",
        "raw_signals",
        "raw_time_series",
        "raw_phase",
        "raw_phase_history",
        "raw_timeseries",
        "raw_graph",
        "raw_coupling",
        "coupling",
        "couplings",
        "evidence",
    ],
)
def test_hierarchy_sync_envelope_construction_rejects_raw_metadata_variants(
    raw_key: str,
) -> None:
    summary = _unchecked_child_summary({raw_key: {"sample": 1}})

    with pytest.raises(ValueError, match=raw_key):
        build_hierarchy_sync_envelope(
            summary,
            source_node="node-a",
            sequence=1,
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {1: "non-string-key"},
        {"nested": {1: "non-string-key"}},
        {"site": object()},
        {"too_large": 2**60},
        {"too_negative": -(2**60)},
        {"huge": 10**100},
        {"first_unsafe": 9007199254740992},
        {"huge": 10**400},
    ],
)
def test_load_hierarchy_sync_envelope_rejects_non_json_safe_metadata(
    metadata: dict[object, object],
) -> None:
    record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 1,
        "summary": {
            "name": "edge-a",
            "channel": "power",
            "R": 0.8,
            "psi": 0.0,
            "metadata": metadata,
        },
    }

    with pytest.raises(ValueError):
        load_hierarchy_sync_envelope(record)


def test_hierarchical_plan_rejects_raw_metadata() -> None:
    summary = _unchecked_child_summary({"raw_phases": [0.1, 0.2]})

    with pytest.raises(ValueError, match="raw_phases"):
        build_hierarchical_orchestration_plan((summary,))


def test_hierarchical_plan_rejects_non_json_safe_metadata() -> None:
    with pytest.raises(ValueError, match="JSON-safe"):
        build_hierarchical_orchestration_plan(
            (_unchecked_child_summary({"site": object()}),)
        )

    with pytest.raises(ValueError, match="JSON-safe"):
        build_hierarchical_orchestration_plan(
            (_unchecked_child_summary({"too_large": 2**60}),)
        )


@pytest.mark.parametrize(
    "attribute",
    [
        "raw_phases",
        "raw_phase",
        "raw_phase_history",
        "raw_timeseries",
        "time_series",
        "raw_signal",
        "actuator",
        "raw_evidence",
        "child_evidence",
    ],
)
def test_forged_child_summary_rejects_raw_evidence_attributes(
    attribute: str,
) -> None:
    summary = _unchecked_child_summary({}, extra_attrs={attribute: [0.1, 0.2]})
    envelope = _unchecked_hierarchy_sync_envelope(summary=summary)

    with pytest.raises(ValueError, match=attribute):
        build_hierarchical_orchestration_plan((summary,))

    with pytest.raises(ValueError, match=attribute):
        load_hierarchy_sync_envelope(envelope)

    with pytest.raises(ValueError, match=attribute):
        ingest_hierarchy_sync_envelopes((envelope,))


@pytest.mark.parametrize(
    "record",
    [
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 10**400,
                "psi": 0.0,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 10**400,
            },
        },
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "monotonic_time_s": 10**400,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
        },
    ],
)
def test_load_hierarchy_sync_envelope_rejects_oversized_numeric_fields(
    record: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        load_hierarchy_sync_envelope(record)


def test_load_hierarchy_sync_envelope_rejects_infinite_metadata_float() -> None:
    record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 1,
        "summary": {
            "name": "edge-a",
            "channel": "power",
            "R": 0.8,
            "psi": 0.0,
            "metadata": {"bad_weight": math.inf},
        },
    }

    with pytest.raises(ValueError, match="finite JSON-safe metadata"):
        load_hierarchy_sync_envelope(record)


def test_load_hierarchy_sync_envelope_rejects_compact_raw_metadata_prefix() -> None:
    record = {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "node-a",
        "sequence": 1,
        "summary": {
            "name": "edge-a",
            "channel": "power",
            "R": 0.8,
            "psi": 0.0,
            "metadata": {"rawphase": [0.1, 0.2]},
        },
    }

    with pytest.raises(ValueError, match="rawphase"):
        load_hierarchy_sync_envelope(record)


def test_hierarchy_sync_envelope_rejects_oversized_sequence() -> None:
    summary = ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0)

    with pytest.raises(ValueError, match="sequence"):
        HierarchySyncEnvelope(
            protocol_version="spo-hierarchy-sync/v1",
            source_node="node-a",
            sequence=10**10000,
            summary=summary,
        )

    with pytest.raises(ValueError, match="sequence"):
        HierarchySyncEnvelope(
            protocol_version="spo-hierarchy-sync/v1",
            source_node="node-a",
            sequence=-(2**60),
            summary=summary,
        )

    with pytest.raises(ValueError, match="sequence"):
        load_hierarchy_sync_envelope(
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-a",
                "sequence": 10**10000,
                "summary": {
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            }
        )

    with pytest.raises(ValueError, match="sequence"):
        load_hierarchy_sync_envelope(
            {
                "protocol_version": "spo-hierarchy-sync/v1",
                "source_node": "node-a",
                "sequence": -(2**60),
                "summary": {
                    "name": "edge-a",
                    "channel": "power",
                    "R": 0.8,
                    "psi": 0.0,
                },
            }
        )


def test_hierarchical_plan_rejects_oversized_threshold() -> None:
    with pytest.raises(ValueError):
        build_hierarchical_orchestration_plan(
            (ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0),),
            degraded_threshold=10**400,
        )


def test_load_hierarchy_sync_envelope_accepts_reduced_metadata() -> None:
    envelope = load_hierarchy_sync_envelope(
        {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
                "metadata": {
                    "site": "north",
                    "role": "edge",
                    "enabled": True,
                    "threshold": 0.875,
                    "optional": None,
                    "tags": ["primary", "calibrated", 2],
                    "nested": {"site_id": "north-1", "weight": 1.0},
                },
            },
        }
    )

    assert envelope.summary.to_audit_record()["metadata"] == {
        "site": "north",
        "role": "edge",
        "enabled": True,
        "threshold": 0.875,
        "optional": None,
        "tags": ["primary", "calibrated", 2],
        "nested": {"site_id": "north-1", "weight": 1.0},
    }
    json.dumps(envelope.to_audit_record(), allow_nan=False)


def test_hierarchy_transport_runtime_ingest_batch_rejects_direct_raw_metadata() -> None:
    runtime = HierarchyTransportRuntime()
    envelope = _unchecked_hierarchy_sync_envelope(
        {"raw_child_observations": [0.1, 0.2]}
    )

    with pytest.raises(ValueError, match="raw_child_observations"):
        runtime.ingest_batch((envelope,))

    assert runtime.previous_sequences == {}


def test_load_hierarchy_sync_envelope_rejects_direct_raw_metadata() -> None:
    envelope = _unchecked_hierarchy_sync_envelope(
        {"nested": {"raw_child_observations": [0.1, 0.2]}}
    )

    with pytest.raises(ValueError, match="raw_child_observations"):
        load_hierarchy_sync_envelope(envelope)


def test_load_hierarchy_sync_envelope_canonicalises_direct_envelope_metadata() -> None:
    metadata = {"site": "north", "nested": {"role": "edge"}}
    envelope = _unchecked_hierarchy_sync_envelope(metadata)

    accepted = load_hierarchy_sync_envelope(envelope)
    metadata["raw_phases"] = [0.1, 0.2]
    metadata["nested"]["time_series"] = [0.1, 0.2]  # type: ignore[index]

    assert accepted is not envelope
    assert accepted.summary.to_audit_record()["metadata"] == {
        "site": "north",
        "nested": {"role": "edge"},
    }


@pytest.mark.parametrize(
    "attribute",
    ["raw_phases", "raw_phase", "raw_phase_history", "raw_timeseries"],
)
def test_load_hierarchy_sync_envelope_rejects_raw_envelope_attribute(
    attribute: str,
) -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0),
        source_node="node-a",
        sequence=1,
    )
    object.__setattr__(envelope, attribute, [0.1, 0.2])

    with pytest.raises(ValueError, match=attribute):
        load_hierarchy_sync_envelope(envelope)


@pytest.mark.parametrize(
    ("envelope_kwargs", "summary_kwargs", "message"),
    [
        (
            {"sequence": True},
            {},
            "sequence must be an integer",
        ),
        (
            {"sequence": -1},
            {},
            "sequence must be >= 0",
        ),
        (
            {"sequence": 10**10000},
            {},
            "sequence",
        ),
        (
            {"protocol_version": ""},
            {},
            "protocol_version must be a non-empty string",
        ),
        (
            {"source_node": ""},
            {},
            "source_node must be a non-empty string",
        ),
        (
            {},
            {"R": math.nan},
            "R must be finite",
        ),
        (
            {},
            {"confidence": 2.0},
            "confidence must be finite and in \\[0, 1\\]",
        ),
    ],
)
def test_direct_hierarchy_sync_envelope_revalidates_all_fields(
    envelope_kwargs: dict[str, object],
    summary_kwargs: dict[str, object],
    message: str,
) -> None:
    summary = _unchecked_child_summary({}, **summary_kwargs)
    envelope = _unchecked_hierarchy_sync_envelope(
        summary=summary,
        **envelope_kwargs,
    )

    with pytest.raises(ValueError, match=message):
        load_hierarchy_sync_envelope(envelope)

    with pytest.raises(ValueError, match=message):
        ingest_hierarchy_sync_envelopes((envelope,))


def test_hierarchy_sync_envelope_construction_rejects_raw_metadata() -> None:
    summary = _unchecked_child_summary({"raw_phases": [0.1, 0.2]})

    with pytest.raises(ValueError, match="raw_phases"):
        build_hierarchy_sync_envelope(
            summary,
            source_node="node-a",
            sequence=1,
        )

    with pytest.raises(ValueError, match="raw_phases"):
        HierarchySyncEnvelope(
            protocol_version="spo-hierarchy-sync/v1",
            source_node="node-a",
            sequence=1,
            summary=summary,
        )


def test_hierarchy_sync_envelope_construction_rejects_non_json_safe_metadata() -> None:
    summary = _unchecked_child_summary({"site": object()})

    with pytest.raises(ValueError, match="JSON-safe"):
        build_hierarchy_sync_envelope(
            summary,
            source_node="node-a",
            sequence=1,
        )

    with pytest.raises(ValueError, match="JSON-safe"):
        HierarchySyncEnvelope(
            protocol_version="spo-hierarchy-sync/v1",
            source_node="node-a",
            sequence=1,
            summary=summary,
        )


def test_hierarchy_sync_ingestion_rejects_direct_raw_metadata() -> None:
    envelope = _unchecked_hierarchy_sync_envelope({"raw_phases": [0.1, 0.2]})

    with pytest.raises(ValueError, match="raw_phases"):
        ingest_hierarchy_sync_envelopes((envelope,))


def test_hierarchy_sync_ingestion_canonicalises_direct_envelope_metadata() -> None:
    metadata = {"site": "north", "nested": {"role": "edge"}}
    envelope = _unchecked_hierarchy_sync_envelope(metadata)

    ledger = ingest_hierarchy_sync_envelopes((envelope,))
    metadata["raw_phases"] = [0.1, 0.2]
    metadata["nested"]["time_series"] = [0.1, 0.2]  # type: ignore[index]

    assert ledger.accepted[0] is not envelope
    assert ledger.to_audit_record()["accepted"][0]["summary"]["metadata"] == {
        "site": "north",
        "nested": {"role": "edge"},
    }


@pytest.mark.parametrize(
    "attribute",
    ["raw_phases", "raw_phase", "raw_phase_history", "raw_timeseries"],
)
def test_hierarchy_sync_ingestion_rejects_raw_envelope_attribute(
    attribute: str,
) -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0),
        source_node="node-a",
        sequence=1,
    )
    object.__setattr__(envelope, attribute, [0.1, 0.2])

    with pytest.raises(ValueError, match=attribute):
        ingest_hierarchy_sync_envelopes((envelope,))


@pytest.mark.parametrize(
    "previous_sequence",
    [
        True,
        "1",
        pytest.param(10**10000, id="oversized"),
        pytest.param(-(2**60), id="negative-oversized"),
    ],
)
def test_hierarchy_sync_ingestion_rejects_invalid_previous_sequences(
    previous_sequence: object,
) -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0),
        source_node="node-a",
        sequence=2,
    )

    with pytest.raises(ValueError, match="sequence"):
        ingest_hierarchy_sync_envelopes(
            (envelope,),
            previous_sequences={"node-a": previous_sequence},  # type: ignore[dict-item]
        )


def test_hierarchy_sync_ingestion_rejects_negative_previous_sequence() -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0),
        source_node="node-a",
        sequence=2,
    )

    with pytest.raises(ValueError, match="sequence must be >= 0"):
        ingest_hierarchy_sync_envelopes(
            (envelope,),
            previous_sequences={"node-a": -1},
        )


def test_hierarchy_sync_ingestion_rejects_non_mapping_previous_sequences() -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "power", R=0.8, psi=0.0),
        source_node="node-a",
        sequence=2,
    )

    with pytest.raises(ValueError, match="previous_sequences must be a mapping"):
        ingest_hierarchy_sync_envelopes(
            (envelope,),
            previous_sequences=object(),  # type: ignore[arg-type]
        )


def test_hierarchy_transport_runtime_rejects_non_mapping_previous_sequences() -> None:
    with pytest.raises(ValueError, match="previous_sequences must be a mapping"):
        HierarchyTransportRuntime(previous_sequences=object())  # type: ignore[arg-type]


def test_hierarchy_gossip_consensus_rejects_direct_raw_metadata() -> None:
    envelope = _unchecked_hierarchy_sync_envelope(
        {"nested": {"time_series": [0.1, 0.2]}}
    )

    with pytest.raises(ValueError, match="time_series"):
        simulate_hierarchy_gossip_consensus(
            (envelope,),
            neighbour_map={},
        )


def test_public_entrypoints_reject_forged_object_types() -> None:
    with pytest.raises(ValueError, match="summary must be a ChildSupervisorSummary"):
        build_hierarchical_orchestration_plan((object(),))

    with pytest.raises(ValueError, match="envelope must be a HierarchySyncEnvelope"):
        ingest_hierarchy_sync_envelopes((object(),))


def test_direct_envelope_revalidation_rejects_negative_monotonic_time() -> None:
    envelope = _unchecked_hierarchy_sync_envelope()
    object.__setattr__(envelope, "monotonic_time_s", -0.1)

    with pytest.raises(ValueError, match="monotonic_time_s must be finite"):
        load_hierarchy_sync_envelope(envelope)

    with pytest.raises(ValueError, match="monotonic_time_s must be finite"):
        ingest_hierarchy_sync_envelopes((envelope,))


def test_forged_child_summary_rejects_non_mapping_metadata() -> None:
    summary = _unchecked_child_summary({})
    object.__setattr__(summary, "metadata", ["not", "mapping"])

    with pytest.raises(ValueError, match="metadata must be a decoded mapping"):
        build_hierarchical_orchestration_plan((summary,))


def test_raw_attribute_scanner_accepts_slot_only_adapter_objects() -> None:
    class SlotOnlyAdapter:
        __slots__ = ()

    _reject_raw_instance_attributes(SlotOnlyAdapter(), "adapter")


def _unchecked_hierarchy_sync_envelope(
    metadata: dict[str, object] | None = None,
    *,
    protocol_version: object = "spo-hierarchy-sync/v1",
    source_node: object = "node-a",
    sequence: object = 1,
    summary: ChildSupervisorSummary | None = None,
) -> HierarchySyncEnvelope:
    envelope = object.__new__(HierarchySyncEnvelope)
    object.__setattr__(envelope, "protocol_version", protocol_version)
    object.__setattr__(envelope, "source_node", source_node)
    object.__setattr__(envelope, "sequence", sequence)
    object.__setattr__(
        envelope,
        "summary",
        summary or _unchecked_child_summary(metadata or {}),
    )
    object.__setattr__(envelope, "monotonic_time_s", None)
    return envelope


class _TwoPassRawEnvelopeMapping(Mapping[str, object]):
    def __init__(self) -> None:
        self._passes = 0
        self._payload: dict[str, object] = {
            "protocol_version": "spo-hierarchy-sync/v1",
            "source_node": "node-a",
            "sequence": 1,
            "summary": {
                "name": "edge-a",
                "channel": "power",
                "R": 0.8,
                "psi": 0.0,
            },
            "raw_phases": [0.1, 0.2],
        }

    def __getitem__(self, key: str) -> object:
        return self._payload[key]

    def __iter__(self) -> Iterator[str]:
        self._passes += 1
        if self._passes == 1:
            return iter(("protocol_version", "source_node", "sequence", "summary"))
        return iter(self._payload)

    def __len__(self) -> int:
        return len(self._payload)


def _unchecked_child_summary(
    metadata: dict[str, object],
    *,
    name: object = "edge-a",
    channel: object = "power",
    R: object = 0.8,
    psi: object = 0.0,
    regime: object = "nominal",
    confidence: object = 1.0,
    extra_attrs: dict[str, object] | None = None,
) -> ChildSupervisorSummary:
    summary = object.__new__(ChildSupervisorSummary)
    object.__setattr__(summary, "name", name)
    object.__setattr__(summary, "channel", channel)
    object.__setattr__(summary, "R", R)
    object.__setattr__(summary, "psi", psi)
    object.__setattr__(summary, "regime", regime)
    object.__setattr__(summary, "confidence", confidence)
    object.__setattr__(summary, "metadata", metadata)
    for key, value in (extra_attrs or {}).items():
        object.__setattr__(summary, key, value)
    return summary


def test_hierarchy_gossip_consensus_moves_neighbours_towards_shared_state() -> None:
    node_a = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "grid", R=0.9, psi=0.0, confidence=1.0),
        source_node="node-a",
        sequence=1,
    )
    node_b = build_hierarchy_sync_envelope(
        ChildSupervisorSummary(
            "edge-b",
            "grid",
            R=0.3,
            psi=math.pi,
            confidence=1.0,
        ),
        source_node="node-b",
        sequence=1,
    )

    rounds = simulate_hierarchy_gossip_consensus(
        (node_a, node_b),
        neighbour_map={"node-a": ("node-b",), "node-b": ("node-a",)},
        rounds=2,
        self_weight=0.5,
    )

    assert [round_record.round_index for round_record in rounds] == [1, 2]
    first_round = rounds[0]
    assert [state.source_node for state in first_round.states] == ["node-a", "node-b"]
    assert [state.summary.R for state in first_round.states] == pytest.approx(
        [0.6, 0.6]
    )
    assert first_round.plan.parent_state.regime_id == "hierarchical_degraded"
    assert first_round.to_audit_record()["states"][0]["summary"]["metadata"] == {
        "consensus": "offline_gossip",
        "source_node": "node-a",
        "neighbour_count": 1,
    }
    assert rounds[1].plan.parent_R == pytest.approx(first_round.plan.parent_R)


def test_hierarchy_gossip_consensus_preserves_isolated_nodes_without_neighbours() -> (
    None
):
    isolated = build_hierarchy_sync_envelope(
        ChildSupervisorSummary(
            "node-a",
            "grid",
            R=0.84,
            psi=0.4,
            confidence=0.9,
        ),
        source_node="node-a",
        sequence=1,
    )

    rounds = simulate_hierarchy_gossip_consensus(
        (isolated,),
        neighbour_map={"node-a": ("missing-node",)},
        rounds=2,
        self_weight=0.3,
    )

    assert [state.summary.R for state in rounds[0].states] == pytest.approx([0.84])
    assert [state.summary.psi for state in rounds[0].states] == pytest.approx([0.4])
    assert rounds[1].states == rounds[0].states
    assert pytest.approx(0.84 * 0.9) == rounds[1].plan.parent_state.layers[0].R


def test_hierarchy_gossip_consensus_zero_confidence_falls_back_to_zero_order() -> None:
    node_a = build_hierarchy_sync_envelope(
        ChildSupervisorSummary(
            "edge-a",
            "grid",
            R=0.9,
            psi=0.0,
            confidence=0.0,
        ),
        source_node="node-a",
        sequence=1,
    )
    node_b = build_hierarchy_sync_envelope(
        ChildSupervisorSummary(
            "edge-b",
            "grid",
            R=0.3,
            psi=math.pi,
            confidence=0.0,
        ),
        source_node="node-b",
        sequence=1,
    )

    rounds = simulate_hierarchy_gossip_consensus(
        (node_a, node_b),
        neighbour_map={"node-a": ("node-b",), "node-b": ("node-a",)},
        rounds=1,
        self_weight=0.5,
    )

    assert [state.summary.R for state in rounds[0].states] == pytest.approx([0.0, 0.0])
    assert rounds[0].plan.parent_R == pytest.approx(0.0)
    assert rounds[0].plan.parent_psi == pytest.approx(0.0)


def test_hierarchy_gossip_consensus_carries_ingestion_rejections_once() -> None:
    stale = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("stale", "grid", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=1,
    )
    accepted = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("accepted", "grid", R=0.7, psi=0.0),
        source_node="node-b",
        sequence=2,
    )

    rounds = simulate_hierarchy_gossip_consensus(
        (stale, accepted),
        neighbour_map={"node-b": ("node-a",)},
        previous_sequences={"node-a": 1},
        rounds=2,
    )

    assert [record["reason"] for record in rounds[0].rejected] == [
        "stale_or_duplicate_sequence"
    ]
    assert rounds[1].rejected == ()
    assert [state.source_node for state in rounds[0].states] == ["node-b"]


@pytest.mark.parametrize(
    ("neighbour_map", "message"),
    [
        (None, "neighbour_map must be a mapping"),
        ({1: ("node-b",)}, "neighbour_map node keys must be strings"),
        ({"node-a": "node-b"}, "neighbour_map neighbours must be a sequence"),
        ({"node-a": 42}, "neighbour_map neighbours must be a sequence"),
        ({"node-a": (1,)}, "neighbour_map neighbour names must be strings"),
    ],
)
def test_hierarchy_gossip_consensus_rejects_invalid_neighbour_map(
    neighbour_map: object,
    message: str,
) -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "grid", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=1,
    )

    with pytest.raises(ValueError, match=message):
        simulate_hierarchy_gossip_consensus(
            (envelope,),
            neighbour_map=neighbour_map,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rounds": 0}, "rounds must be >= 1"),
        ({"rounds": "1"}, "rounds must be an integer"),
        ({"self_weight": 1.5}, "self_weight must be finite and in \\[0, 1\\]"),
        ({"self_weight": "0.5"}, "self_weight must be a finite number"),
    ],
)
def test_hierarchy_gossip_consensus_rejects_invalid_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    envelope = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "grid", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=1,
    )

    with pytest.raises(ValueError, match=message):
        simulate_hierarchy_gossip_consensus(
            (envelope,),
            neighbour_map={},
            **kwargs,  # type: ignore[arg-type]
        )


def test_hierarchy_transport_runtime_rejects_malformed_batch_atomically() -> None:
    runtime = HierarchyTransportRuntime()
    accepted_candidate = build_hierarchy_sync_envelope(
        ChildSupervisorSummary("edge-a", "grid", R=0.9, psi=0.0),
        source_node="node-a",
        sequence=7,
    )

    with pytest.raises(ValueError, match="source_node must be a non-empty string"):
        runtime.ingest(
            (
                accepted_candidate,
                {
                    "protocol_version": "spo-hierarchy-sync/v1",
                    "source_node": "",
                    "sequence": 8,
                    "summary": {
                        "name": "edge-b",
                        "channel": "grid",
                        "R": 0.8,
                        "psi": 0.0,
                    },
                },
            )
        )

    assert runtime.previous_sequences == {}


def test_hierarchy_escalation_records_remain_bounded_and_json_safe() -> None:
    plan = build_hierarchical_orchestration_plan(
        (
            ChildSupervisorSummary(
                "critical-edge",
                "grid",
                R=0.2,
                psi=math.pi,
                regime="critical-local-actuator-clamped",
                confidence=0.25,
                metadata={
                    "site": "north",
                    "operator_note": "summary-only",
                },
            ),
        ),
        critical_threshold=0.35,
        degraded_threshold=0.65,
        min_confidence=0.5,
    )

    records = [record.to_audit_record() for record in plan.escalations]

    assert records == [
        {
            "child": "critical-edge",
            "channel": "grid",
            "severity": "degraded",
            "reason": "child_summary_below_min_confidence",
            "R": 0.2,
            "confidence": 0.25,
            "child_regime": "critical-local-actuator-clamped",
        },
        {
            "child": "critical-edge",
            "channel": "grid",
            "severity": "critical",
            "reason": "child_coherence_below_critical",
            "R": 0.2,
            "confidence": 0.25,
            "child_regime": "critical-local-actuator-clamped",
        },
    ]
    assert all("metadata" not in record for record in records)
    assert all("raw_phases" not in record for record in records)
    json.dumps(records, allow_nan=False)
