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


class TestFreqIdPipelineEndToEnd:
    """Full pipeline: signal → identify_frequencies → omegas → Engine → R → Regime."""

    def test_identified_freqs_drive_engine_regime(self):
        """identify_frequencies → omegas → UPDEEngine → R → RegimeManager."""
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        signal = _multi_sine([5.0, 10.0, 20.0], fs=200.0, duration=2.0)
        result = identify_frequencies(signal, fs=200.0)
        n = len(result.frequencies)
        assert n >= 1
        omegas = np.array(result.frequencies) * 2 * np.pi
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        assert regime.name in {"NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"}

    def test_performance_identify_2s_200hz_under_50ms(self):
        """identify_frequencies(2s @ 200Hz) < 50ms."""
        import time

        signal = _multi_sine([5.0, 10.0], fs=200.0, duration=2.0)
        identify_frequencies(signal, fs=200.0)  # warm-up
        t0 = time.perf_counter()
        for _ in range(20):
            identify_frequencies(signal, fs=200.0)
        elapsed = (time.perf_counter() - t0) / 20
        assert elapsed < 0.05, f"identify_frequencies took {elapsed * 1e3:.1f}ms"


# Pipeline wiring: identify_frequencies → omegas → UPDEEngine(RK4)
# → compute_order_parameter → RegimeManager. Performance: freq_id(2s@200Hz)<50ms.
