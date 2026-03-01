# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.oscillators.informational import InformationalExtractor

TWO_PI = 2.0 * np.pi


def test_regular_timestamps_high_quality():
    """Evenly-spaced events at 10 Hz should yield quality near 1.0."""
    timestamps = np.arange(0.0, 1.0, 0.1)
    extractor = InformationalExtractor(node_id="i0")
    states = extractor.extract(timestamps, sample_rate=0.0)
    assert len(states) == 1
    assert states[0].quality > 0.8


def test_irregular_timestamps_lower_quality():
    rng = np.random.default_rng(123)
    timestamps = np.sort(rng.uniform(0, 10, size=50))
    extractor = InformationalExtractor()
    states = extractor.extract(timestamps, sample_rate=0.0)
    assert states[0].quality < 1.0


def test_omega_positive_for_regular_events():
    timestamps = np.arange(0.0, 2.0, 0.05)
    extractor = InformationalExtractor()
    states = extractor.extract(timestamps, sample_rate=0.0)
    assert states[0].omega > 0.0


def test_theta_in_valid_range():
    timestamps = np.arange(0.0, 5.0, 0.2)
    extractor = InformationalExtractor()
    states = extractor.extract(timestamps, sample_rate=0.0)
    assert 0.0 <= states[0].theta < TWO_PI


def test_single_timestamp_zero_quality():
    extractor = InformationalExtractor()
    states = extractor.extract(np.array([1.0]), sample_rate=0.0)
    assert states[0].quality == 0.0
    assert states[0].omega == 0.0


def test_channel_is_informational():
    timestamps = np.arange(0.0, 1.0, 0.1)
    extractor = InformationalExtractor(node_id="info_x")
    states = extractor.extract(timestamps, sample_rate=0.0)
    assert states[0].channel == "I"
    assert states[0].node_id == "info_x"


def test_all_zero_intervals_returns_zero_quality():
    timestamps = np.array([1.0, 1.0, 1.0])
    extractor = InformationalExtractor()
    states = extractor.extract(timestamps, sample_rate=0.0)
    assert states[0].quality == 0.0
    assert states[0].omega == 0.0


def test_quality_score_empty():
    extractor = InformationalExtractor()
    assert extractor.quality_score([]) == 0.0


def test_quality_score_nonempty():
    timestamps = np.arange(0.0, 1.0, 0.1)
    extractor = InformationalExtractor()
    states = extractor.extract(timestamps, sample_rate=0.0)
    score = extractor.quality_score(states)
    assert 0.0 < score <= 1.0
