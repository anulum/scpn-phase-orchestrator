from __future__ import annotations

import math

import pytest

from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    build_hierarchical_orchestration_plan,
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
