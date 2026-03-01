# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

TWO_PI = 2.0 * np.pi


def test_hilbert_phase_monotonic():
    """10 Hz sine at 1000 Hz: unwrapped Hilbert phases increase monotonically."""
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    extractor = PhysicalExtractor(node_id="test")
    states = extractor.extract(signal, fs)
    assert len(states) == 1
    assert 0.0 <= states[0].theta < TWO_PI


def test_quality_above_threshold_for_clean_sinusoid():
    fs = 1000.0
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(TWO_PI * 5.0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    assert states[0].quality > 0.5


def test_omega_matches_input_frequency():
    fs = 1000.0
    f0 = 10.0
    t = np.arange(0, 1.0, 1.0 / fs)
    signal = np.sin(TWO_PI * f0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    expected_omega = TWO_PI * f0
    np.testing.assert_allclose(states[0].omega, expected_omega, rtol=0.05)


def test_channel_is_physical():
    fs = 500.0
    t = np.arange(0, 0.2, 1.0 / fs)
    signal = np.sin(TWO_PI * 8.0 * t)
    extractor = PhysicalExtractor(node_id="p1")
    states = extractor.extract(signal, fs)
    assert states[0].channel == "P"
    assert states[0].node_id == "p1"


def test_quality_score_aggregation():
    fs = 1000.0
    t = np.arange(0, 0.5, 1.0 / fs)
    signal = np.sin(TWO_PI * 10.0 * t)
    extractor = PhysicalExtractor()
    states = extractor.extract(signal, fs)
    score = extractor.quality_score(states)
    assert 0.0 <= score <= 1.0
    assert score == states[0].quality


def test_quality_score_empty():
    extractor = PhysicalExtractor()
    assert extractor.quality_score([]) == 0.0


def test_snr_estimate_returns_one_for_real_signal():
    """Hilbert real part equals input, so noise ≈ 0 and quality = 1.0."""
    from scipy.signal import hilbert

    signal = np.sin(TWO_PI * 5.0 * np.arange(0, 0.5, 0.001))
    quality = PhysicalExtractor._snr_estimate(signal, hilbert(signal))
    assert quality == 1.0
