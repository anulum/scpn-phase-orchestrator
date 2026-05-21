# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for winding number tracker

from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import winding as winding_module
from scpn_phase_orchestrator.monitor.winding import winding_numbers, winding_vector
from tests.typing_contracts import assert_precise_ndarray_hint


class TestWindingNumbers:
    def test_public_array_contracts_are_parameterised(self):
        hints = (
            get_type_hints(winding_numbers)["phases_history"],
            get_type_hints(winding_numbers)["return"],
            get_type_hints(winding_vector)["phases_history"],
            get_type_hints(winding_vector)["return"],
        )

        for hint in hints:
            assert_precise_ndarray_hint(hint)

        assert "float64" in str(hints[0])
        assert "int64" in str(hints[1])

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
        history[:, 0] = np.linspace(0, 2.1 * np.pi, T)  # floor(1.05) = +1
        history[:, 1] = np.linspace(0, 6.1 * np.pi, T)  # floor(3.05) = +3
        history[:, 2] = np.linspace(0, -3.9 * np.pi, T)  # floor(-1.95) = -2
        w = winding_numbers(history)
        assert w[0] == 1
        assert w[1] == 3
        assert w[2] == -2

    @pytest.mark.parametrize(
        "history",
        [
            np.array([[0.0, np.nan], [1.0, 2.0]], dtype=np.float64),
            np.array([[0.0, np.inf], [1.0, 2.0]], dtype=np.float64),
            np.array([0.0, np.nan], dtype=np.float64),
        ],
    )
    def test_rejects_non_finite_history(self, history: np.ndarray) -> None:
        with pytest.raises(ValueError, match="phases_history"):
            winding_numbers(history)

    def test_rejects_non_numeric_history(self) -> None:
        with pytest.raises(ValueError, match="phases_history"):
            winding_numbers([["not-a-phase"]])

    def test_accepts_array_like_history(self) -> None:
        history = np.column_stack(
            [
                np.linspace(0.0, 2.1 * np.pi, 10),
                np.linspace(0.0, -2.1 * np.pi, 10),
            ]
        ).tolist()
        w = winding_numbers(history)

        np.testing.assert_array_equal(w, [1, -2])


class TestWindingPipelineWiring:
    """Pipeline: engine trajectory → winding numbers."""

    def test_engine_trajectory_to_winding(self):
        """UPDEEngine → phase history → winding_numbers: counts
        how many full rotations each oscillator made."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([2.0, 4.0, 1.0, 3.0])
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        history = [phases.copy()]
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            history.append(phases.copy())
        traj = np.array(history)  # (T, n)

        w = winding_numbers(traj)
        assert w.shape == (n,)
        # Faster omegas should have more windings
        assert w[1] > w[2], "ω=4 should wind more than ω=1"


class TestWindingRustDispatch:
    def test_winding_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[np.ndarray, int, int]] = []

        def _fake_backend(flat: np.ndarray, t: int, n: int) -> np.ndarray:
            calls.append((flat, t, n))
            return np.array([1, -1], dtype=np.int64)

        monkeypatch.setattr(winding_module, "_dispatch", lambda: _fake_backend)
        history = np.array([[0.0, 0.0], [2.1 * np.pi, -2.1 * np.pi]], dtype=np.float64)
        w = winding_numbers(history)
        np.testing.assert_array_equal(w, [1, -1])
        assert len(calls) == 1

    def test_winding_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_backend(_flat: np.ndarray, _t: int, _n: int) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(winding_module, "_dispatch", lambda: _raising_backend)
        history = np.array([[0.0, 0.0], [2.1 * np.pi, -2.1 * np.pi]], dtype=np.float64)
        w = winding_numbers(history)
        np.testing.assert_array_equal(w, [0, -1])
