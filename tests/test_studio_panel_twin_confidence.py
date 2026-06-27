# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Studio twin-confidence panel tests

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import cast

import pytest

from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceBaseline,
    TwinConfidenceScore,
    TwinDivergence,
    score_twin_confidence,
    summarise_twin_confidence,
)
from scpn_phase_orchestrator.studio.ui_helpers.panel_twin_confidence import (
    build_twin_confidence_studio_panel,
)


def _score(phase_js: float, order_w1: float) -> TwinConfidenceScore:
    baseline = TwinConfidenceBaseline(
        phase_js_mean=0.05,
        phase_js_std=0.01,
        order_w1_mean=0.02,
        order_w1_std=0.01,
        sample_count=100,
        band_z=3.0,
    )
    return score_twin_confidence(
        TwinDivergence(
            phase_js_divergence=phase_js,
            order_wasserstein=order_w1,
            n_bins=36,
            backend="python",
        ),
        baseline,
    )


def _score_records() -> tuple[dict[str, object], ...]:
    scores = (
        _score(0.05, 0.02),
        _score(0.07, 0.02),
        _score(0.12, 0.02),
    )
    return tuple(score.to_audit_record() for score in scores)


def _summary_record(
    records: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    scores = [
        TwinConfidenceScore(
            confidence=cast("float", record["confidence"]),
            status=cast("str", record["status"]),
            phase_js_divergence=cast("float", record["phase_js_divergence"]),
            order_wasserstein=cast("float", record["order_wasserstein"]),
            phase_js_z=cast("float", record["phase_js_z"]),
            order_w1_z=cast("float", record["order_w1_z"]),
            composite_z=cast("float", record["composite_z"]),
            phase_js_within_band=cast("bool", record["phase_js_within_band"]),
            order_w1_within_band=cast("bool", record["order_w1_within_band"]),
            backend=cast("str", record["backend"]),
            score_hash=cast("str", record["score_hash"]),
        )
        for record in records
    ]
    return summarise_twin_confidence(scores).to_audit_record()


def _rehash_summary(summary: Mapping[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in summary.items() if key != "summary_hash"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {**summary, "summary_hash": sha256(canonical.encode("utf-8")).hexdigest()}


def test_twin_confidence_panel_renders_real_monitor_audit_records() -> None:
    """The Studio panel accepts real twin-confidence score and summary records."""
    records = _score_records()
    summary = _summary_record(records)

    panel = build_twin_confidence_studio_panel(records, summary)

    assert panel["panel_kind"] == "studio_twin_confidence_panel"
    assert panel["monitor"] == "digital_twin_confidence"
    assert panel["claim_boundary"] == (
        "digital_twin_confidence_observability_not_actuation"
    )
    assert panel["non_actuating"] is True
    assert panel["execution_disabled"] is True
    assert panel["actuation_permitted"] is False
    assert panel["live_merge_permitted"] is False
    assert panel["hot_patch_permitted"] is False
    assert panel["score_count"] == 3
    assert panel["status_counts"] == {
        "healthy": 1,
        "warning": 1,
        "critical": 1,
    }
    assert panel["worst_status_level"] == 2
    assert panel["backends"] == ("python",)
    assert cast("Mapping[str, object]", panel["summary"])["worst_status"] == "critical"
    assert cast("Mapping[str, object]", panel["latest"])["status"] == "critical"
    assert cast("Mapping[str, object]", panel["worst"])["status"] == "critical"
    assert len(cast("tuple[Mapping[str, object], ...]", panel["series"])) == 3
    assert "not_actuation" in cast("str", panel["claim_boundary"])


@pytest.mark.parametrize(
    ("records", "match"),
    [
        ((), "non-empty sequence"),
        (cast("tuple[Mapping[str, object], ...]", {"confidence": 1.0}), "sequence"),
        ((cast("Mapping[str, object]", object()),), "entries must be mappings"),
    ],
)
def test_twin_confidence_panel_rejects_malformed_score_sequences(
    records: tuple[Mapping[str, object], ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_twin_confidence_studio_panel(records, _summary_record(_score_records()))


def test_twin_confidence_panel_rejects_bad_score_hash() -> None:
    records = list(_score_records())
    records[0] = {**records[0], "score_hash": "0" * 64}

    with pytest.raises(ValueError, match="score_hash does not match"):
        build_twin_confidence_studio_panel(
            tuple(records),
            _summary_record(_score_records()),
        )


def test_twin_confidence_panel_rejects_unknown_status() -> None:
    records = list(_score_records())
    records[0] = {**records[0], "status": "unknown"}

    with pytest.raises(ValueError, match="healthy, warning, or critical"):
        build_twin_confidence_studio_panel(
            tuple(records),
            _summary_record(_score_records()),
        )


def test_twin_confidence_panel_rejects_non_boolean_band_flags() -> None:
    records = list(_score_records())
    records[0] = {**records[0], "phase_js_within_band": "yes"}

    with pytest.raises(ValueError, match="phase_js_within_band"):
        build_twin_confidence_studio_panel(
            tuple(records),
            _summary_record(_score_records()),
        )


def test_twin_confidence_panel_rejects_non_mapping_summary() -> None:
    with pytest.raises(ValueError, match="summary_record must be a mapping"):
        build_twin_confidence_studio_panel(
            _score_records(),
            cast("Mapping[str, object]", object()),
        )


def test_twin_confidence_panel_rejects_summary_count_sum_drift() -> None:
    records = _score_records()
    summary = {**_summary_record(records), "healthy_count": 3}

    with pytest.raises(ValueError, match="sum to tick_count"):
        build_twin_confidence_studio_panel(records, summary)


def test_twin_confidence_panel_rejects_bad_summary_hash() -> None:
    records = _score_records()
    summary = {**_summary_record(records), "summary_hash": "0" * 64}

    with pytest.raises(ValueError, match="summary_hash does not match"):
        build_twin_confidence_studio_panel(records, summary)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("tick_count", 2, "tick_count"),
        ("healthy_count", 2, "status counts"),
        ("min_confidence", 1.0, "min_confidence"),
        ("mean_confidence", 1.0, "mean_confidence"),
        ("latest_confidence", 1.0, "latest_confidence"),
        ("latest_status", "healthy", "latest_status"),
        ("worst_status", "healthy", "worst_status"),
    ],
)
def test_twin_confidence_panel_rejects_summary_record_drift(
    field: str,
    value: object,
    match: str,
) -> None:
    records = _score_records()
    summary = {**_summary_record(records), field: value}

    if field == "tick_count":
        summary["critical_count"] = 0
    if field == "healthy_count":
        summary["critical_count"] = 0
    summary = _rehash_summary(summary)

    with pytest.raises(ValueError, match=match):
        build_twin_confidence_studio_panel(records, summary)


def test_twin_confidence_panel_reports_warning_as_worst_noncritical_status() -> None:
    scores = (_score(0.05, 0.02), _score(0.07, 0.02))
    records = tuple(score.to_audit_record() for score in scores)

    panel = build_twin_confidence_studio_panel(
        records,
        summarise_twin_confidence(scores).to_audit_record(),
    )

    assert panel["worst_status_level"] == 1
    assert cast("Mapping[str, object]", panel["summary"])["worst_status"] == "warning"


def test_twin_confidence_panel_reports_healthy_when_all_records_are_healthy() -> None:
    scores = (_score(0.05, 0.02),)
    records = tuple(score.to_audit_record() for score in scores)

    panel = build_twin_confidence_studio_panel(
        records,
        summarise_twin_confidence(scores).to_audit_record(),
    )

    assert panel["worst_status_level"] == 0
    assert cast("Mapping[str, object]", panel["summary"])["worst_status"] == "healthy"
