# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Wavelet / zero-crossing extractor tests

"""Behavioural tests for the wavelet-ridge and zero-crossing phase extractors
and the extractor factory: frequency recovery, phase validity, quality
behaviour, robustness to offset and noise, degenerate inputs, and input
validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators import (
    WaveletExtractor,
    ZeroCrossingExtractor,
)

TWO_PI = 2.0 * np.pi


def _sine(
    freq_hz: float, fs: float, seconds: float, *, offset: float = 0.0
) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    return np.sin(TWO_PI * freq_hz * t) + offset


# --------------------------------------------------------------------------- #
# ZeroCrossingExtractor
# --------------------------------------------------------------------------- #
class TestZeroCrossingExtractor:
    def test_recovers_frequency_of_pure_sine(self) -> None:
        fs, f0 = 256.0, 8.0
        states = ZeroCrossingExtractor().extract(_sine(f0, fs, 2.0), fs)
        assert len(states) == 1
        omega = states[0].omega
        assert omega == pytest.approx(TWO_PI * f0, rel=0.03)

    def test_phase_in_principal_range_and_quality_high(self) -> None:
        fs, f0 = 256.0, 5.0
        state = ZeroCrossingExtractor().extract(_sine(f0, fs, 2.0), fs)[0]
        assert 0.0 <= state.theta < TWO_PI
        assert state.quality > 0.9  # a clean sine has highly regular half-periods

    def test_dc_offset_does_not_change_frequency(self) -> None:
        fs, f0 = 256.0, 7.0
        clean = ZeroCrossingExtractor().extract(_sine(f0, fs, 2.0), fs)[0].omega
        shifted = (
            ZeroCrossingExtractor().extract(_sine(f0, fs, 2.0, offset=5.0), fs)[0].omega
        )
        assert shifted == pytest.approx(clean, rel=1e-6)

    def test_noise_lowers_quality_but_keeps_frequency(self) -> None:
        fs, f0 = 256.0, 8.0
        rng = np.random.default_rng(0)
        sig = _sine(f0, fs, 2.0) + rng.normal(0.0, 0.1, int(fs * 2.0))
        state = ZeroCrossingExtractor().extract(sig, fs)[0]
        assert state.omega == pytest.approx(TWO_PI * f0, rel=0.06)
        assert 0.0 < state.quality < 1.0

    def test_constant_signal_yields_zero_frequency_and_quality(self) -> None:
        state = ZeroCrossingExtractor().extract(np.ones(128), 256.0)[0]
        assert state.omega == 0.0
        assert state.quality == 0.0
        assert 0.0 <= state.theta < TWO_PI

    def test_amplitude_tracks_sine_amplitude(self) -> None:
        fs = 256.0
        state = ZeroCrossingExtractor().extract(3.0 * _sine(8.0, fs, 2.0), fs)[0]
        assert state.amplitude == pytest.approx(3.0, rel=0.05)

    @pytest.mark.parametrize(
        "bad_signal",
        [np.ones((4, 4)), np.array([1.0]), np.array([np.nan, 1.0, 2.0])],
    )
    def test_invalid_signal_rejected(self, bad_signal: np.ndarray) -> None:
        with pytest.raises(ValueError):
            ZeroCrossingExtractor().extract(bad_signal, 256.0)

    @pytest.mark.parametrize("bad_rate", [0.0, -1.0, float("inf"), float("nan")])
    def test_invalid_sample_rate_rejected(self, bad_rate: float) -> None:
        with pytest.raises(ValueError):
            ZeroCrossingExtractor().extract(_sine(8.0, 256.0, 1.0), bad_rate)

    def test_blank_node_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            ZeroCrossingExtractor(node_id="  ")

    def test_quality_score_aggregates(self) -> None:
        extractor = ZeroCrossingExtractor()
        states = extractor.extract(_sine(8.0, 256.0, 2.0), 256.0)
        assert extractor.quality_score(states) == pytest.approx(states[0].quality)
        assert extractor.quality_score([]) == 0.0


# --------------------------------------------------------------------------- #
# WaveletExtractor
# --------------------------------------------------------------------------- #
class TestWaveletExtractor:
    @pytest.mark.parametrize("f0", [8.0, 20.0, 40.0])
    def test_ridge_recovers_dominant_frequency(self, f0: float) -> None:
        fs = 256.0
        state = WaveletExtractor().extract(_sine(f0, fs, 2.0), fs)[0]
        # The log-spaced 48-scale grid resolves frequency to ~10 %.
        assert state.omega == pytest.approx(TWO_PI * f0, rel=0.10)

    def test_phase_valid_and_quality_high_for_clean_sine(self) -> None:
        fs = 256.0
        state = WaveletExtractor().extract(_sine(10.0, fs, 2.0), fs)[0]
        assert 0.0 <= state.theta < TWO_PI
        assert state.quality > 0.7

    def test_robust_to_additive_noise(self) -> None:
        fs, f0 = 256.0, 12.0
        rng = np.random.default_rng(1)
        sig = _sine(f0, fs, 2.0) + rng.normal(0.0, 0.3, int(fs * 2.0))
        state = WaveletExtractor().extract(sig, fs)[0]
        assert state.omega == pytest.approx(TWO_PI * f0, rel=0.12)

    def test_dominant_band_selected_over_weak_high_frequency(self) -> None:
        fs = 256.0
        t = np.arange(int(fs * 2.0)) / fs
        sig = np.sin(TWO_PI * 6.0 * t) + 0.1 * np.sin(TWO_PI * 50.0 * t)
        state = WaveletExtractor().extract(sig, fs)[0]
        assert state.omega == pytest.approx(TWO_PI * 6.0, rel=0.12)

    def test_constant_signal_yields_zero_frequency(self) -> None:
        state = WaveletExtractor().extract(np.full(256, 2.0), 256.0)[0]
        assert state.omega == 0.0
        assert state.quality == 0.0

    def test_amplitude_tracks_sine_amplitude(self) -> None:
        fs = 256.0
        state = WaveletExtractor().extract(2.5 * _sine(10.0, fs, 2.0), fs)[0]
        assert state.amplitude == pytest.approx(2.5, rel=0.05)

    @pytest.mark.parametrize(
        "bad_signal",
        [np.ones((4, 4)), np.array([1.0]), np.array([1.0, np.inf])],
    )
    def test_invalid_signal_rejected(self, bad_signal: np.ndarray) -> None:
        with pytest.raises(ValueError):
            WaveletExtractor().extract(bad_signal, 256.0)

    @pytest.mark.parametrize("bad_rate", [0.0, -2.0, float("nan")])
    def test_invalid_sample_rate_rejected(self, bad_rate: float) -> None:
        with pytest.raises(ValueError):
            WaveletExtractor().extract(_sine(8.0, 256.0, 1.0), bad_rate)


# --------------------------------------------------------------------------- #
# Edge / error-path coverage
# --------------------------------------------------------------------------- #
class TestEdgeCoverage:
    def test_zero_crossing_rejects_complex_signal(self) -> None:
        with pytest.raises(ValueError):
            ZeroCrossingExtractor().extract(np.array([1 + 1j, 2 + 2j, 3 - 1j]), 256.0)

    def test_zero_crossing_rejects_non_numeric_sample_rate(self) -> None:
        with pytest.raises(ValueError):
            ZeroCrossingExtractor().extract(_sine(8.0, 256.0, 1.0), "fast")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_hysteresis", [-0.1, 1.0, 1.5, float("nan")])
    def test_zero_crossing_rejects_invalid_hysteresis(
        self, bad_hysteresis: float
    ) -> None:
        with pytest.raises(ValueError, match="hysteresis"):
            ZeroCrossingExtractor(hysteresis=bad_hysteresis)

    def test_zero_crossing_subband_signal_has_no_confirmed_crossings(self) -> None:
        # A faint oscillation rides on a strong DC step that dominates the RMS,
        # so the deadband swallows it: no confirmed crossings, zero frequency.
        fs = 256.0
        t = np.arange(int(fs)) / fs
        sig = 100.0 * (t > 0.5).astype(float) + 0.01 * np.sin(TWO_PI * 8.0 * t)
        state = ZeroCrossingExtractor().extract(sig, fs)[0]
        assert state.omega == 0.0

    def test_interval_quality_guard_on_degenerate_intervals(self) -> None:
        assert ZeroCrossingExtractor._interval_quality(np.zeros(2)) == 0.0

    def test_wavelet_rejects_blank_node_id(self) -> None:
        with pytest.raises(ValueError):
            WaveletExtractor(node_id="")

    def test_wavelet_rejects_complex_signal(self) -> None:
        with pytest.raises(ValueError):
            WaveletExtractor().extract(np.array([1 + 1j, 2 - 1j, 0 + 3j]), 256.0)

    def test_wavelet_rejects_non_numeric_sample_rate(self) -> None:
        with pytest.raises(ValueError):
            WaveletExtractor().extract(_sine(8.0, 256.0, 1.0), None)  # type: ignore[arg-type]

    def test_wavelet_short_signal_has_empty_frequency_grid(self) -> None:
        # Too few samples for even three cycles below Nyquist: empty grid → the
        # extractor returns a zero-frequency fallback rather than raising.
        state = WaveletExtractor().extract(np.array([1.0, -1.0, 1.0, -1.0]), 256.0)[0]
        assert state.omega == 0.0
        assert state.quality == 0.0

    def test_wavelet_quality_score_aggregates(self) -> None:
        extractor = WaveletExtractor()
        states = extractor.extract(_sine(10.0, 256.0, 2.0), 256.0)
        assert extractor.quality_score(states) == pytest.approx(states[0].quality)
        assert extractor.quality_score([]) == 0.0

    def test_ridge_quality_guard_on_zero_envelope(self) -> None:
        assert (
            WaveletExtractor._ridge_quality(np.zeros(8, dtype=np.complex128), 2.0)
            == 0.0
        )
