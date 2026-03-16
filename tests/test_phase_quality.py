# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase quality tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


def _make_states(qualities, amplitudes=None):
    n = len(qualities)
    if amplitudes is None:
        amplitudes = [1.0] * n
    return [
        PhaseState(
            theta=0.0,
            omega=1.0,
            amplitude=amplitudes[i],
            quality=qualities[i],
            channel="P",
            node_id=f"n{i}",
        )
        for i in range(n)
    ]


def test_high_quality_score():
    scorer = PhaseQualityScorer()
    states = _make_states([0.95, 0.90, 0.92])
    score = scorer.score(states)
    assert score > 0.85


def test_collapse_detected_when_all_low():
    scorer = PhaseQualityScorer()
    states = _make_states([0.05, 0.02, 0.01, 0.08])
    assert scorer.detect_collapse(states, threshold=0.1) is True


def test_no_collapse_for_healthy_states():
    scorer = PhaseQualityScorer()
    states = _make_states([0.8, 0.7, 0.9, 0.6])
    assert scorer.detect_collapse(states, threshold=0.1) is False


def test_downweight_mask_zeros_low_quality():
    scorer = PhaseQualityScorer()
    states = _make_states([0.9, 0.1, 0.8, 0.05])
    mask = scorer.downweight_mask(states, min_quality=0.3)
    assert mask[0] == pytest.approx(0.9)
    assert mask[1] == pytest.approx(0.0)
    assert mask[2] == pytest.approx(0.8)
    assert mask[3] == pytest.approx(0.0)


def test_empty_states_score_zero():
    scorer = PhaseQualityScorer()
    assert scorer.score([]) == 0.0


def test_empty_states_collapse_true():
    scorer = PhaseQualityScorer()
    assert scorer.detect_collapse([]) is True


def test_downweight_mask_empty():
    scorer = PhaseQualityScorer()
    mask = scorer.downweight_mask([])
    assert len(mask) == 0
