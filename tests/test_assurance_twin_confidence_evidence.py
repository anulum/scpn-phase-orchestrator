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

import json

import pytest

import scpn_phase_orchestrator.assurance.twin_confidence_evidence as _twin_evidence
from scpn_phase_orchestrator.assurance import (
    TWIN_CONFIDENCE,
    build_twin_confidence_evidence,
)
from scpn_phase_orchestrator.monitor.twin_confidence import (
    TwinConfidenceBaseline,
    TwinDivergence,
    score_twin_confidence,
)

assert _twin_evidence is not None


_BASELINE = TwinConfidenceBaseline(
    phase_js_mean=0.01,
    phase_js_std=0.005,
    order_w1_mean=0.02,
    order_w1_std=0.01,
    sample_count=50,
    band_z=3.0,
)


def _real_record(
    phase_js: float = 0.012,
    order_w1: float = 0.022,
    baseline: TwinConfidenceBaseline = _BASELINE,
) -> dict[str, object]:
    """Return the audit record the real scorer produces for one divergence."""
    divergence = TwinDivergence(
        phase_js_divergence=phase_js,
        order_wasserstein=order_w1,
        n_bins=36,
        backend="python",
    )
    return score_twin_confidence(divergence, baseline).to_audit_record()


def _score(**overrides: object) -> dict[str, object]:
    """Return a real score record with field overrides (the hash then no longer fits).

    Used only for records that must be rejected before the hash is checked.
    """
    record = _real_record()
    record.update(overrides)
    return record


def test_builds_evidence_from_valid_score() -> None:
    record = _real_record()
    item = build_twin_confidence_evidence(record)

    assert item.evidence_id == "twin-confidence-score"
    assert item.category == TWIN_CONFIDENCE
    assert item.record == record
    assert item.record["status"] == "healthy"
    assert "healthy" in item.summary
    assert f"{record['confidence']:.3f}" in item.summary


def test_record_survives_a_json_round_trip() -> None:
    """A score persisted to disk and read back still matches its hash."""
    record = json.loads(json.dumps(_real_record()))
    assert build_twin_confidence_evidence(record).record == record


def test_boundary_confidence_values_accepted() -> None:
    at_mean = _real_record(phase_js=0.005, order_w1=0.01)
    degenerate = TwinConfidenceBaseline(
        phase_js_mean=0.0,
        phase_js_std=0.0,
        order_w1_mean=0.0,
        order_w1_std=0.0,
        sample_count=1,
        band_z=3.0,
    )
    far = _real_record(phase_js=0.5, order_w1=0.9, baseline=degenerate)
    assert (at_mean["confidence"], far["confidence"]) == (1.0, 0.0)
    for record in (at_mean, far):
        assert build_twin_confidence_evidence(record).category == TWIN_CONFIDENCE


def test_record_edited_after_scoring_is_rejected() -> None:
    """A critical tick relabelled healthy no longer matches its hash."""
    record = _real_record(phase_js=0.3, order_w1=0.8)
    assert record["status"] == "critical"
    record["status"] = "healthy"
    with pytest.raises(ValueError, match="score_hash does not match"):
        build_twin_confidence_evidence(record)


def test_extra_field_breaks_the_hash() -> None:
    with pytest.raises(ValueError, match="score_hash does not match"):
        build_twin_confidence_evidence(_score(extra_field="kept"))


def test_unknown_status_is_rejected() -> None:
    with pytest.raises(ValueError, match="'status' must be one of"):
        build_twin_confidence_evidence(_score(status="fine"))


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


def test_record_that_is_not_json_is_rejected() -> None:
    """A value JSON cannot encode cannot be hashed the way the scorer hashes."""
    with pytest.raises(ValueError, match="must be JSON-serialisable"):
        build_twin_confidence_evidence(_score(extra_field={1, 2}))
