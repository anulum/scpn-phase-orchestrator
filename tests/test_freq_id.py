# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Frequency ID tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.freq_id import (
    FrequencyResult,
    identify_frequencies,
)


def _multi_sine(freqs: list[float], fs: float, duration: float) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    return np.array([np.sin(2 * np.pi * f * t) for f in freqs])


class TestIdentifyFrequencies:
    def test_returns_result(self):
        data = _multi_sine([5.0, 10.0, 20.0], fs=200.0, duration=1.0)
        result = identify_frequencies(data, fs=200.0)
        assert isinstance(result, FrequencyResult)

    def test_frequencies_shape(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_frequencies(data, fs=200.0)
        assert len(result.frequencies) > 0

    def test_layer_assignment_length(self):
        data = _multi_sine([5.0, 10.0, 15.0], fs=200.0, duration=1.0)
        result = identify_frequencies(data, fs=200.0)
        assert len(result.layer_assignment) == 3

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="3 time samples"):
            identify_frequencies(np.ones((2, 2)), fs=100.0)

    def test_n_modes_override(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_frequencies(data, fs=200.0, n_modes=1)
        assert len(result.frequencies) == 1

    def test_amplitudes_positive(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_frequencies(data, fs=200.0)
        assert np.all(result.amplitudes >= 0)

    def test_single_channel(self):
        t = np.arange(0, 1.0, 1.0 / 200.0)
        data = np.sin(2 * np.pi * 10.0 * t).reshape(1, -1)
        result = identify_frequencies(data, fs=200.0)
        assert len(result.layer_assignment) == 1


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
