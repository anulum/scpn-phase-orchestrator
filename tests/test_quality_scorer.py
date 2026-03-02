# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.oscillators.base import PhaseState
from scpn_phase_orchestrator.oscillators.quality import PhaseQualityScorer


def _ps(quality: float, amplitude: float = 1.0) -> PhaseState:
    return PhaseState(theta=0.0, omega=1.0, amplitude=amplitude,
                      quality=quality, channel="P", node_id="test")


def test_score_empty_returns_zero():
    assert PhaseQualityScorer().score([]) == 0.0


def test_score_uniform_quality():
    states = [_ps(0.8), _ps(0.8), _ps(0.8)]
    np.testing.assert_allclose(PhaseQualityScorer().score(states), 0.8, atol=1e-12)


def test_score_weighted_by_amplitude():
    states = [_ps(1.0, amplitude=10.0), _ps(0.0, amplitude=1e-15)]
    assert PhaseQualityScorer().score(states) > 0.9


def test_collapse_empty_is_true():
    assert PhaseQualityScorer().detect_collapse([]) is True


def test_collapse_all_high_quality():
    states = [_ps(0.9), _ps(0.8), _ps(0.7)]
    assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is False


def test_collapse_majority_low():
    states = [_ps(0.01), _ps(0.02), _ps(0.9)]
    assert PhaseQualityScorer().detect_collapse(states, threshold=0.1) is True


def test_downweight_mask_empty():
    mask = PhaseQualityScorer().downweight_mask([])
    assert len(mask) == 0


def test_downweight_mask_zeros_below_threshold():
    states = [_ps(0.5), _ps(0.1), _ps(0.8)]
    mask = PhaseQualityScorer().downweight_mask(states, min_quality=0.3)
    assert mask[0] > 0.0
    assert mask[1] == 0.0
    assert mask[2] > 0.0
