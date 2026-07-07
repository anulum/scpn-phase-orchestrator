# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — early-warning detector auditor tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.evaluation.auditor import (
    DetectorAudit,
    audit_detector,
    audit_scoring_detector,
)


def _skilful_corpus(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    events = rng.normal(2.0, 1.0, 60)
    nulls = rng.normal(0.0, 1.0, 300)
    return events, nulls


class TestAuditDetector:
    def test_skilful_detector_beats_chance(self):
        events, nulls = _skilful_corpus()
        audit = audit_detector(
            event_scores=events,
            null_scores=nulls,
            detector_name="modal-growth",
            target_false_alarm=0.10,
            n_permutations=4000,
        )
        assert audit.detector_name == "modal-growth"
        assert audit.achieved_false_alarm == pytest.approx(0.10, abs=0.01)
        assert audit.detection_rate > audit.achieved_false_alarm
        assert audit.p_value < 0.01
        assert audit.beats_chance is True
        assert audit.n_events == 60
        assert audit.n_nulls == 300
        assert 0 <= audit.n_events_alarmed <= 60

    def test_no_skill_detector_does_not_beat_chance(self):
        rng = np.random.default_rng(1)
        events = rng.normal(0.0, 1.0, 60)
        nulls = rng.normal(0.0, 1.0, 300)
        audit = audit_detector(
            event_scores=events, null_scores=nulls, n_permutations=4000
        )
        assert audit.p_value > 0.05
        assert audit.beats_chance is False
        assert audit.detector_name == "detector"

    def test_p_value_property_matches_significance(self):
        events, nulls = _skilful_corpus()
        audit = audit_detector(
            event_scores=events, null_scores=nulls, n_permutations=500
        )
        assert audit.p_value == audit.significance.p_value

    def test_open_gate_threshold_is_minus_infinity(self):
        events, nulls = _skilful_corpus()
        audit = audit_detector(
            event_scores=events,
            null_scores=nulls,
            target_false_alarm=1.0,
            n_permutations=200,
        )
        assert audit.matched_threshold == float("-inf")
        assert audit.detection_rate == 1.0

    def test_empty_event_scores_rejected(self):
        with pytest.raises(ValueError, match="event_scores must not be empty"):
            audit_detector(event_scores=[], null_scores=[0.0])

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_alpha_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError, match="alpha must be in"):
            audit_detector(event_scores=[1.0], null_scores=[0.0], alpha=bad)

    def test_alpha_controls_beats_chance_flag(self):
        rng = np.random.default_rng(2)
        events = rng.normal(0.0, 1.0, 40)
        nulls = rng.normal(0.0, 1.0, 200)
        # alpha = 1.0 forces beats_chance True for any p < 1.
        audit = audit_detector(
            event_scores=events, null_scores=nulls, alpha=1.0, n_permutations=500
        )
        assert audit.alpha == 1.0
        assert audit.beats_chance == (audit.p_value < 1.0)


class TestDetectorAuditRecord:
    def test_to_record_serialises_finite_threshold(self):
        audit = audit_detector(
            event_scores=[2.0, 3.0], null_scores=[0.0, 1.0], n_permutations=200
        )
        record = audit.to_record()
        assert isinstance(record["matched_threshold"], float)
        assert isinstance(record["significance"], dict)
        assert record["n_events"] == 2

    def test_to_record_encodes_open_gate_as_string(self):
        audit = audit_detector(
            event_scores=[0.0],
            null_scores=[0.0, 1.0],
            target_false_alarm=1.0,
            n_permutations=100,
        )
        record = audit.to_record()
        assert record["matched_threshold"] == "-inf"

    def test_is_frozen(self):
        audit = audit_detector(event_scores=[1.0], null_scores=[0.0], n_permutations=50)
        assert isinstance(audit, DetectorAudit)
        with pytest.raises(AttributeError):
            # Deliberate frozen-attribute write to assert immutability at runtime.
            audit.beats_chance = True  # type: ignore[misc]


class TestAuditScoringDetector:
    def test_scores_each_series_then_audits(self):
        rng = np.random.default_rng(4)

        def slope(series):
            values = np.asarray(series, dtype=float)
            x = np.arange(values.shape[0])
            return float(np.polyfit(x, values, 1)[0])

        events = [
            np.linspace(0.0, rise, 20) + rng.normal(0.0, 0.05, 20)
            for rise in rng.uniform(0.8, 1.5, 30)
        ]
        nulls = [rng.normal(0.0, 0.05, 20) for _ in range(120)]
        audit = audit_scoring_detector(
            score=slope,
            event_series=events,
            null_series=nulls,
            detector_name="ols-slope",
            n_permutations=2000,
        )
        assert audit.detector_name == "ols-slope"
        assert audit.p_value < 0.01
        assert audit.detection_rate > audit.achieved_false_alarm

    def test_empty_event_series_rejected(self):
        with pytest.raises(ValueError, match="event_series must not be empty"):
            audit_scoring_detector(
                score=lambda s: 0.0, event_series=[], null_series=[[0.0]]
            )

    def test_empty_null_series_rejected(self):
        with pytest.raises(ValueError, match="null_series must not be empty"):
            audit_scoring_detector(
                score=lambda s: 0.0, event_series=[[0.0]], null_series=[]
            )
