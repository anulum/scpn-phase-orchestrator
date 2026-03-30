# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Envelope solver tests

"""Tests for modulation envelope extraction."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.envelope import (
    EnvelopeState,
    envelope_modulation_depth,
    extract_envelope,
)


class TestExtractEnvelope:
    def test_constant_signal_flat_envelope(self) -> None:
        amp = np.ones(100)
        env = extract_envelope(amp, window=10)
        assert env.shape == (100,)
        np.testing.assert_allclose(env, 1.0, atol=1e-10)

    def test_shape_preserved_1d(self) -> None:
        amp = np.random.default_rng(0).uniform(0, 1, 50)
        env = extract_envelope(amp, window=5)
        assert env.shape == (50,)

    def test_shape_preserved_2d(self) -> None:
        amp = np.random.default_rng(0).uniform(0, 1, (50, 4))
        env = extract_envelope(amp, window=5)
        assert env.shape == (50, 4)

    def test_empty_input(self) -> None:
        env = extract_envelope(np.array([]))
        assert env.size == 0

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window must be >= 1"):
            extract_envelope(np.ones(10), window=0)

    def test_rms_values_nonnegative(self) -> None:
        rng = np.random.default_rng(0)
        amp = rng.standard_normal(200)
        env = extract_envelope(amp, window=10)
        assert np.all(env >= 0.0)

    def test_window_1_is_absolute_value(self) -> None:
        amp = np.array([1.0, -2.0, 3.0, -4.0])
        env = extract_envelope(amp, window=1)
        np.testing.assert_allclose(env, np.abs(amp))

    def test_am_signal_envelope_tracks_modulation(self) -> None:
        """Amplitude-modulated sinusoid: envelope must track the modulation
        curve, not the carrier. This is the core use case."""
        t = np.linspace(0, 2 * np.pi, 1000)
        carrier = np.sin(50 * t)
        modulation = 1.0 + 0.5 * np.sin(t)  # slow AM envelope
        signal = carrier * modulation
        env = extract_envelope(signal, window=20)
        # RMS envelope should correlate with the modulation curve
        # Use correlation coefficient as a discriminatory check
        # (skip front padding region)
        env_core = env[50:-50]
        mod_core = modulation[50:-50]
        corr = float(np.corrcoef(env_core, mod_core)[0, 1])
        assert corr > 0.9, (
            f"Envelope should track AM modulation (corr={corr:.3f}, expected >0.9)"
        )

    def test_2d_per_column_independent(self) -> None:
        """2D envelope must process each column independently:
        constant column stays flat, varying column has non-trivial envelope."""
        n_t, n_osc = 100, 3
        amp = np.ones((n_t, n_osc))
        amp[:, 1] = np.linspace(0.1, 2.0, n_t)  # ramp
        amp[:, 2] = np.sin(np.linspace(0, 4 * np.pi, n_t))
        env = extract_envelope(amp, window=5)
        assert env.shape == (n_t, n_osc)
        # Column 0 (constant) must be flat
        np.testing.assert_allclose(env[:, 0], 1.0, atol=1e-10)
        # Column 1 (ramp) must increase
        assert env[-1, 1] > env[10, 1], "Ramp envelope should increase"

    def test_window_larger_than_signal(self) -> None:
        """Window > signal length: must still return valid output without crashing.
        Output length = max(window-1, 0) + max(len-window+1, 0) padding."""
        amp = np.array([1.0, 2.0, 3.0])
        env = extract_envelope(amp, window=10)
        assert np.all(np.isfinite(env))
        assert env.size > 0


class TestEnvelopeModulationDepth:
    def test_constant_returns_zero(self) -> None:
        assert envelope_modulation_depth(np.ones(100)) == 0.0

    def test_full_modulation(self) -> None:
        env = np.array([0.0, 1.0, 0.0, 1.0])
        assert envelope_modulation_depth(env) == pytest.approx(1.0)

    def test_partial_modulation(self) -> None:
        env = np.array([1.0, 3.0])
        # (3-1)/(3+1) = 0.5
        assert envelope_modulation_depth(env) == pytest.approx(0.5)

    def test_empty_returns_zero(self) -> None:
        assert envelope_modulation_depth(np.array([])) == 0.0

    def test_all_zeros_returns_zero(self) -> None:
        assert envelope_modulation_depth(np.zeros(10)) == 0.0

    def test_range(self) -> None:
        rng = np.random.default_rng(0)
        env = np.abs(rng.standard_normal(100)) + 0.01
        depth = envelope_modulation_depth(env)
        assert 0.0 <= depth <= 1.0


class TestEnvelopeState:
    def test_frozen(self) -> None:
        es = EnvelopeState(
            mean_amplitude=1.0,
            amplitude_spread=0.1,
            modulation_depth=0.5,
            subcritical_count=2,
        )
        with pytest.raises(AttributeError):
            es.mean_amplitude = 2.0  # type: ignore[misc]

    def test_fields(self) -> None:
        es = EnvelopeState(
            mean_amplitude=1.5,
            amplitude_spread=0.2,
            modulation_depth=0.3,
            subcritical_count=0,
        )
        assert es.mean_amplitude == 1.5
        assert es.subcritical_count == 0
