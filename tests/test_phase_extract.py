# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase extraction tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.phase_extract import PhaseResult, extract_phases


def _sine(freq: float, fs: float, duration: float) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    return np.sin(2 * np.pi * freq * t)


class TestExtractPhases:
    def test_returns_phase_result(self):
        sig = _sine(10.0, 1000.0, 0.5)
        result = extract_phases(sig, fs=1000.0)
        assert isinstance(result, PhaseResult)
        assert len(result.phases) == len(sig)
        assert len(result.amplitudes) == len(sig)

    def test_phases_in_range(self):
        sig = _sine(5.0, 500.0, 1.0)
        result = extract_phases(sig, fs=500.0)
        assert np.all(result.phases >= 0)
        assert np.all(result.phases <= 2 * np.pi)

    def test_dominant_freq_correct(self):
        sig = _sine(25.0, 1000.0, 1.0)
        result = extract_phases(sig, fs=1000.0)
        assert abs(result.dominant_freq - 25.0) < 2.0

    def test_bandpass(self):
        sig = _sine(10.0, 1000.0, 1.0) + _sine(100.0, 1000.0, 1.0)
        result = extract_phases(sig, fs=1000.0, bandpass=(5.0, 20.0))
        assert abs(result.dominant_freq - 10.0) < 3.0

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="too short"):
            extract_phases(np.array([1.0, 2.0]), fs=100.0)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            extract_phases(np.ones((3, 4)), fs=100.0)

    def test_amplitudes_positive(self):
        sig = _sine(10.0, 500.0, 0.5)
        result = extract_phases(sig, fs=500.0)
        assert np.all(result.amplitudes >= 0)

    def test_instantaneous_freq_shape(self):
        sig = _sine(10.0, 500.0, 0.5)
        result = extract_phases(sig, fs=500.0)
        assert result.instantaneous_freq.shape == sig.shape
