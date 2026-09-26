# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — twin-confidence exporter integration tests

"""Score summaries render through the dedicated exporter and public facade."""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceScore,
    summarise_twin_confidence,
)
from scpn_phase_orchestrator.monitor.twin_confidence import (
    twin_confidence_prometheus_text as facade_export,
)
from scpn_phase_orchestrator.monitor.twin_confidence_export import (
    twin_confidence_prometheus_text,
)


def _score(status: str, confidence: float) -> TwinConfidenceScore:
    return TwinConfidenceScore(
        confidence=confidence,
        status=status,
        phase_js_divergence=0.0,
        order_wasserstein=0.0,
        phase_js_z=0.0,
        order_w1_z=0.0,
        composite_z=0.0,
        phase_js_within_band=True,
        order_w1_within_band=True,
        backend="python",
        score_hash="x",
    )


# ---------------------------------------------------------------------
# Prometheus rendering
# ---------------------------------------------------------------------


def test_prometheus_text_contains_all_series() -> None:
    summary = summarise_twin_confidence(
        [_score("healthy", 1.0), _score("warning", 0.5), _score("critical", 0.1)]
    )
    text = twin_confidence_prometheus_text(summary)
    assert text.endswith("\n")
    assert "spo_twin_confidence_mean " in text
    assert "spo_twin_confidence_min " in text
    assert "spo_twin_confidence_latest " in text
    assert "spo_twin_confidence_tick_count 3" in text
    assert 'spo_twin_confidence_status_total{status="healthy"} 1' in text
    assert 'spo_twin_confidence_status_total{status="warning"} 1' in text
    assert 'spo_twin_confidence_status_total{status="critical"} 1' in text
    assert "spo_twin_confidence_worst_status_level 2" in text


@pytest.mark.parametrize(
    ("status", "level"),
    [("healthy", 0), ("warning", 1), ("critical", 2)],
)
def test_prometheus_worst_status_level(status: str, level: int) -> None:
    summary = summarise_twin_confidence([_score(status, 0.5)])
    text = twin_confidence_prometheus_text(summary)
    assert f"spo_twin_confidence_worst_status_level {level}" in text


def test_prometheus_custom_prefix() -> None:
    summary = summarise_twin_confidence([_score("healthy", 1.0)])
    text = twin_confidence_prometheus_text(summary, prefix="twin")
    assert "twin_twin_confidence_mean " in text


@pytest.mark.parametrize("prefix", ["", "   "])
def test_prometheus_rejects_empty_prefix(prefix: str) -> None:
    summary = summarise_twin_confidence([_score("healthy", 1.0)])
    with pytest.raises(ValueError, match="prefix"):
        twin_confidence_prometheus_text(summary, prefix=prefix)


def test_existing_export_facade_preserves_callable_identity() -> None:
    assert facade_export is twin_confidence_prometheus_text
