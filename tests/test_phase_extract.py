# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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


class TestPhaseExtractPipelineWiring:
    """Pipeline: signal → extract_phases → omegas → engine."""

    def test_extracted_phases_drive_engine(self):
        """extract_phases → PhaseResult.phases → engine initial conditions.
        Proves autotune extraction feeds simulation."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        signals = np.column_stack(
            [
                _sine(5.0, 200.0, 1.0),
                _sine(10.0, 200.0, 1.0),
                _sine(15.0, 200.0, 1.0),
            ]
        )
        n = signals.shape[1]
        results = [extract_phases(signals[:, i], fs=200.0) for i in range(n)]
        phases = np.array([r.phases[-1] for r in results])
        omegas = np.array([r.instantaneous_freq[-1] for r in results])

        eng = UPDEEngine(n, dt=0.01)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                knm,
                0.0,
                0.0,
                alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
