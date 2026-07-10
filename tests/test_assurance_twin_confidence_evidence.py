# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — twin-confidence assurance evidence tests

"""Tests for deriving twin-confidence assurance evidence from a score record.

``build_twin_confidence_evidence`` maps a serialised ``TwinConfidenceScore`` into
a ``twin_confidence`` evidence item, restating the score verbatim and rejecting a
record that is missing a required field or carries a confidence outside ``[0, 1]``.
"""

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.assurance import (
    TWIN_CONFIDENCE,
    build_twin_confidence_evidence,
)


def _score(**overrides: object) -> dict[str, object]:
    """Return a valid serialised twin-confidence score, with field overrides."""
    base: dict[str, object] = {
        "confidence": 0.87,
        "status": "healthy",
        "phase_js_divergence": 0.01,
        "order_wasserstein": 0.02,
        "phase_js_z": 0.5,
        "order_w1_z": 0.4,
        "composite_z": 0.64,
        "phase_js_within_band": True,
        "order_w1_within_band": True,
        "backend": "python",
        "score_hash": "f" * 64,
    }
    base.update(overrides)
    return base


def test_builds_evidence_from_valid_score() -> None:
    item = build_twin_confidence_evidence(_score())

    assert item.evidence_id == "twin-confidence-score"
    assert item.category == TWIN_CONFIDENCE
    assert item.record["confidence"] == 0.87
    assert item.record["status"] == "healthy"
    assert item.record["backend"] == "python"
    assert "healthy" in item.summary
    assert "0.870" in item.summary


def test_record_is_restated_verbatim() -> None:
    item = build_twin_confidence_evidence(_score(extra_field="kept"))

    assert item.record["extra_field"] == "kept"
    assert item.record["score_hash"] == "f" * 64


def test_boundary_confidence_values_accepted() -> None:
    for value in (0.0, 1.0):
        item = build_twin_confidence_evidence(_score(confidence=value))
        assert item.category == TWIN_CONFIDENCE


@pytest.mark.parametrize("confidence", [1.5, -0.1, 2.0])
def test_confidence_out_of_unit_range_rejected(confidence: float) -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        build_twin_confidence_evidence(_score(confidence=confidence))


def test_non_finite_confidence_rejected() -> None:
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        build_twin_confidence_evidence(_score(confidence=float("nan")))


def test_boolean_confidence_rejected() -> None:
    # A bool is an int subclass; it must not be read as a confidence value.
    with pytest.raises(ValueError, match="must be a number"):
        build_twin_confidence_evidence(_score(confidence=True))


def test_non_numeric_confidence_rejected() -> None:
    with pytest.raises(ValueError, match="must be a number"):
        build_twin_confidence_evidence(_score(confidence="high"))


def test_missing_status_rejected() -> None:
    score = _score()
    del score["status"]
    with pytest.raises(ValueError, match="'status'"):
        build_twin_confidence_evidence(score)


def test_blank_status_rejected() -> None:
    with pytest.raises(ValueError, match="'status'"):
        build_twin_confidence_evidence(_score(status="  "))


def test_missing_score_hash_rejected() -> None:
    score = _score()
    del score["score_hash"]
    with pytest.raises(ValueError, match="'score_hash'"):
        build_twin_confidence_evidence(score)


def test_non_mapping_rejected() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        build_twin_confidence_evidence(["not", "a", "mapping"])  # type: ignore[arg-type]
