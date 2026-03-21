# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for winding number tracker

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.winding import winding_numbers, winding_vector


class TestWindingNumbers:
    def test_one_full_rotation(self):
        """Oscillator advancing past 2π should have winding number 1."""
        T = 100
        history = np.zeros((T, 2))
        # Overshoot slightly to avoid floor boundary ambiguity
        history[:, 0] = np.linspace(0, 2.1 * np.pi, T)
        history[:, 1] = np.zeros(T)  # stationary
        w = winding_numbers(history)
        assert w[0] == 1
        assert w[1] == 0

    def test_two_full_rotations(self):
        T = 200
        history = np.zeros((T, 1))
        history[:, 0] = np.linspace(0, 4.1 * np.pi, T)
        w = winding_numbers(history)
        assert w[0] == 2

    def test_negative_rotation(self):
        """Clockwise (decreasing phase) → negative winding number."""
        T = 100
        history = np.zeros((T, 1))
        history[:, 0] = np.linspace(0, -2 * np.pi, T)
        w = winding_numbers(history)
        assert w[0] == -1

    def test_no_rotation(self):
        """Small oscillation around zero → winding number 0."""
        T = 50
        history = np.zeros((T, 3))
        history[:, 0] = 0.1 * np.sin(np.linspace(0, 4 * np.pi, T))
        w = winding_numbers(history)
        assert w[0] == 0
        assert w[1] == 0
        assert w[2] == 0

    def test_wrapped_phases(self):
        """Even if phases are wrapped mod 2π, unwrapping recovers the winding."""
        T = 100
        raw = np.linspace(0, 6.1 * np.pi, T)
        wrapped = raw % (2 * np.pi)
        history = wrapped.reshape(-1, 1)
        w = winding_numbers(history)
        assert w[0] == 3

    def test_single_timestep(self):
        history = np.array([[0.0, 1.0]])
        w = winding_numbers(history)
        np.testing.assert_array_equal(w, [0, 0])

    def test_winding_vector_same_as_winding_numbers(self):
        T = 100
        history = np.zeros((T, 2))
        history[:, 0] = np.linspace(0, 4 * np.pi, T)
        history[:, 1] = np.linspace(0, -2 * np.pi, T)
        w = winding_numbers(history)
        v = winding_vector(history)
        np.testing.assert_array_equal(w, v)

    def test_mixed_windings(self):
        """Multiple oscillators with different winding numbers."""
        T = 200
        history = np.zeros((T, 3))
        history[:, 0] = np.linspace(0, 2.1 * np.pi, T)   # floor(1.05) = +1
        history[:, 1] = np.linspace(0, 6.1 * np.pi, T)   # floor(3.05) = +3
        history[:, 2] = np.linspace(0, -3.9 * np.pi, T)  # floor(-1.95) = -2
        w = winding_numbers(history)
        assert w[0] == 1
        assert w[1] == 3
        assert w[2] == -2
