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
