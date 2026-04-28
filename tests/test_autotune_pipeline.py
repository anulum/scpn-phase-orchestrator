# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Auto-tune pipeline tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.autotune.pipeline import (
    AutoTuneResult,
    identify_binding_spec,
)


def _multi_sine(freqs: list[float], fs: float, duration: float) -> np.ndarray:
    t = np.arange(0, duration, 1.0 / fs)
    return np.array([np.sin(2 * np.pi * f * t) for f in freqs])


class TestAutoTunePipeline:
    def test_returns_result(self):
        data = _multi_sine([5.0, 10.0, 20.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        assert isinstance(result, AutoTuneResult)

    def test_omegas_length(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        assert len(result.omegas) == 2

    def test_knm_shape(self):
        data = _multi_sine([5.0, 10.0, 15.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        assert result.knm.shape == (3, 3)

    def test_knm_non_negative(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        assert np.all(result.knm >= 0)

    def test_knm_zero_diagonal(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        np.testing.assert_array_equal(np.diag(result.knm), 0.0)

    def test_dominant_freqs(self):
        data = _multi_sine([5.0, 20.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        assert len(result.dominant_freqs) == 2
        assert abs(result.dominant_freqs[0] - 5.0) < 2.0

    def test_kc_estimate(self):
        data = _multi_sine([5.0, 10.0, 15.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        assert result.K_c_estimate >= 0

    def test_n_layers_override(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0, n_layers=4)
        assert result.n_layers == 4

    def test_alpha_zero(self):
        data = _multi_sine([5.0, 10.0], fs=200.0, duration=1.0)
        result = identify_binding_spec(data, fs=200.0)
        np.testing.assert_array_equal(result.alpha, np.zeros((2, 2)))


class TestAutoTunePipelineWiring:
    """Pipeline: raw signal → identify_binding_spec → engine simulation."""

    def test_identified_spec_drives_engine(self):
        """identify_binding_spec → K_nm + omegas → engine → R∈[0,1]."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import (
            compute_order_parameter,
        )

        data = _multi_sine([5.0, 10.0, 20.0], fs=200.0, duration=2.0)
        result = identify_binding_spec(data, fs=200.0)

        n = len(result.omegas)
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array(result.omegas)
        for _ in range(100):
            phases = eng.step(
                phases,
                omegas,
                result.knm,
                0.0,
                0.0,
                result.alpha,
            )
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
