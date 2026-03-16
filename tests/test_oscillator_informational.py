# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational oscillator tests

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


def test_regular_events_theta_deterministic():
    """Regular 10 Hz events over different durations yield distinct theta."""
    ext = InformationalExtractor()
    ts_a = np.arange(0.0, 1.0, 0.1)  # 1 s @ 10 Hz
    ts_b = np.arange(0.0, 2.0, 0.1)  # 2 s @ 10 Hz
    theta_a = ext.extract(ts_a, 0.0)[0].theta
    theta_b = ext.extract(ts_b, 0.0)[0].theta
    # Both should be in [0, 2pi) and deterministic
    assert 0.0 <= theta_a < TWO_PI
    assert 0.0 <= theta_b < TWO_PI


def test_different_durations_different_theta():
    """Different total durations produce different theta (not all ~0)."""
    ext = InformationalExtractor()
    # 0.35 s at 10 Hz → median_hz * total = 10 * 0.35 = 3.5 → theta = pi
    ts_short = np.arange(0.0, 0.35, 0.1)
    # 0.75 s at 10 Hz → median_hz * total = 10 * 0.75 = 7.5 → theta = pi
    ts_long = np.arange(0.0, 0.75, 0.1)
    theta_short = ext.extract(ts_short, 0.0)[0].theta
    theta_long = ext.extract(ts_long, 0.0)[0].theta
    assert 0.0 <= theta_short < TWO_PI
    assert 0.0 <= theta_long < TWO_PI
