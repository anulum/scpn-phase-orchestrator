# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling tests

"""Tests for phase-amplitude coupling (PAC) measurement."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.pac import modulation_index, pac_gate, pac_matrix

TWO_PI = 2.0 * np.pi


class TestModulationIndex:
    def test_uncoupled_signals_low_mi(self) -> None:
        rng = np.random.default_rng(0)
        theta = rng.uniform(0, TWO_PI, 1000)
        amp = rng.uniform(0.5, 1.5, 1000)
        mi = modulation_index(theta, amp)
        assert mi < 0.1

    def test_entrained_signals_detectable_mi(self) -> None:
        theta = np.linspace(0, TWO_PI * 10, 5000) % TWO_PI
        # Amplitude concentrated in narrow phase window → detectable PAC
        amp = np.exp(4.0 * np.cos(theta))
        mi = modulation_index(theta, amp)
        assert mi > 0.05

    def test_mi_in_unit_range(self) -> None:
        rng = np.random.default_rng(1)
        theta = rng.uniform(0, TWO_PI, 500)
        amp = np.abs(rng.standard_normal(500))
        mi = modulation_index(theta, amp)
        assert 0.0 <= mi <= 1.0

    def test_empty_input_returns_zero(self) -> None:
        assert modulation_index(np.array([]), np.array([])) == 0.0

    def test_zero_amplitude_returns_zero(self) -> None:
        theta = np.linspace(0, TWO_PI, 100)
        amp = np.zeros(100)
        assert modulation_index(theta, amp) == 0.0

    def test_mismatched_lengths_uses_shorter(self) -> None:
        theta = np.linspace(0, TWO_PI, 100)
        amp = np.ones(50)
        mi = modulation_index(theta, amp)
        assert 0.0 <= mi <= 1.0

    def test_n_bins_parameter(self) -> None:
        theta = np.linspace(0, TWO_PI * 10, 5000) % TWO_PI
        amp = np.exp(4.0 * np.cos(theta))
        mi_18 = modulation_index(theta, amp, n_bins=18)
        mi_36 = modulation_index(theta, amp, n_bins=36)
        assert mi_18 > 0.05
        assert mi_36 > 0.05


class TestPACMatrix:
    def test_shape(self) -> None:
        phases = np.random.default_rng(0).uniform(0, TWO_PI, (100, 4))
        amps = np.random.default_rng(1).uniform(0, 1, (100, 4))
        mat = pac_matrix(phases, amps)
        assert mat.shape == (4, 4)

    def test_values_in_range(self) -> None:
        phases = np.random.default_rng(0).uniform(0, TWO_PI, (200, 3))
        amps = np.abs(np.random.default_rng(1).standard_normal((200, 3)))
        mat = pac_matrix(phases, amps)
        assert np.all(mat >= 0.0)
        assert np.all(mat <= 1.0)

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            pac_matrix(np.zeros(10), np.zeros(10))

    def test_mismatched_n_raises(self) -> None:
        with pytest.raises(ValueError, match="same number"):
            pac_matrix(np.zeros((10, 3)), np.zeros((10, 4)))


class TestPACGate:
    def test_gate_open(self) -> None:
        assert pac_gate(0.5, threshold=0.3) is True

    def test_gate_closed(self) -> None:
        assert pac_gate(0.1, threshold=0.3) is False

    def test_gate_at_threshold(self) -> None:
        assert pac_gate(0.3, threshold=0.3) is True
