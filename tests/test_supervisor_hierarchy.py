from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    build_hierarchical_orchestration_plan,
    build_hierarchy_sync_envelope,
    ingest_hierarchy_sync_envelopes,
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
