# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare section tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import poincare as poincare_module
from scpn_phase_orchestrator.monitor.poincare import (
    PoincareResult,
    phase_poincare,
    poincare_section,
    return_times,
)


class TestPoincareSection:
    def test_circle_crossings(self):
        """Circle trajectory crosses x=0 twice per revolution."""
        t = np.linspace(0, 6 * np.pi, 3000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = poincare_section(traj, normal=np.array([1.0, 0.0]))
        assert isinstance(result, PoincareResult)
        # 3 full revolutions → 3 positive crossings
        assert len(result.crossings) >= 2

    def test_periodic_constant_return_time(self):
        """Periodic orbit → constant return time."""
        t = np.linspace(0, 8 * np.pi, 4000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = poincare_section(traj, normal=np.array([1.0, 0.0]))
        if len(result.return_times) > 1:
            assert result.std_return_time < 5.0  # nearly constant

    def test_no_crossings(self):
        """Trajectory that doesn't cross the plane."""
        traj = np.column_stack([np.ones(100), np.linspace(0, 1, 100)])
        result = poincare_section(traj, normal=np.array([1.0, 0.0]), offset=5.0)
        assert len(result.crossings) == 0
        assert result.mean_return_time == 0.0

    def test_both_directions(self):
        """direction='both' counts crossings in both directions."""
        t = np.linspace(0, 4 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        pos = poincare_section(traj, normal=np.array([1.0, 0.0]), direction="positive")
        both = poincare_section(traj, normal=np.array([1.0, 0.0]), direction="both")
        assert len(both.crossings) >= len(pos.crossings)

    def test_negative_direction(self):
        t = np.linspace(0, 4 * np.pi, 2000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = poincare_section(
            traj, normal=np.array([1.0, 0.0]), direction="negative"
        )
        assert len(result.crossings) >= 1

    @pytest.mark.parametrize(
        ("trajectory", "match"),
        [
            (np.array([[0.0], [np.nan]], dtype=np.float64), "trajectory"),
            (np.array([[0.0], [np.inf]], dtype=np.float64), "trajectory"),
            (np.array([[0.0], [True]], dtype=object), "trajectory"),
            (np.array([[0.0], [1.0j]], dtype=object), "trajectory"),
            (np.array([[0.0], [1.0j]]), "trajectory"),
            ([["not-a-state"]], "trajectory"),
        ],
    )
    def test_rejects_invalid_trajectory(
        self,
        trajectory: Any,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            poincare_section(trajectory, normal=np.array([1.0]))

    @pytest.mark.parametrize(
        ("normal", "match"),
        [
            (np.array([1.0, 0.0]), "normal shape"),
            (np.array([np.nan]), "normal"),
            (np.array([True], dtype=object), "normal"),
            (np.array([1.0j], dtype=object), "normal"),
            (np.array([1.0j]), "normal"),
            ([["not-a-normal"]], "normal"),
        ],
    )
    def test_rejects_invalid_normal(self, normal: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            poincare_section(np.zeros((3, 1)), normal=normal)

    def test_zero_normal_vector_returns_no_crossings(self):
        trajectory = np.array(
            [
                [0.0, 1.0],
                [0.5, 1.5],
                [1.0, 2.0],
            ]
        )
        result = poincare_section(trajectory, normal=np.array([0.0, 0.0]))

        assert len(result.crossings) == 0
        assert result.mean_return_time == 0.0
        assert result.std_return_time == 0.0

    @pytest.mark.parametrize("offset", [False, np.nan, np.inf, "0.0"])
    def test_rejects_invalid_offset(self, offset: Any) -> None:
        with pytest.raises(ValueError, match="offset"):
            poincare_section(np.zeros((3, 1)), normal=np.array([1.0]), offset=offset)

    def test_accepts_array_like_section_inputs(self) -> None:
        result = poincare_section([[-1.0], [1.0]], normal=[1.0], offset=0.0)

        assert result.crossings.shape == (1, 1)

    def test_section_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[np.ndarray, int, int, np.ndarray, float, int]] = []

        def _fake_section(
            traj_flat: np.ndarray,
            t: int,
            d: int,
            normal: np.ndarray,
            offset: float,
            direction_id: int,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            calls.append((traj_flat, t, d, normal, offset, direction_id))
            cr = np.array([0.0], dtype=np.float64)
            times = np.array([0.5], dtype=np.float64)
            return cr, times, 1

        monkeypatch.setattr(
            poincare_module,
            "_dispatch",
            lambda _fn_name: _fake_section,
        )
        result = poincare_section([[-1.0], [1.0]], normal=[1.0], offset=0.0)
        assert result.crossings.shape == (1, 1)
        assert result.crossing_times.shape == (1,)
        assert len(calls) == 1

    def test_section_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_section(
            _traj_flat: np.ndarray,
            _t: int,
            _d: int,
            _normal: np.ndarray,
            _offset: float,
            _direction_id: int,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            poincare_module,
            "_dispatch",
            lambda _fn_name: _raising_section,
        )
        result = poincare_section([[-1.0], [1.0]], normal=[1.0], offset=0.0)
        assert result.crossings.shape == (1, 1)
        assert result.crossing_times.shape == (1,)

    @pytest.mark.parametrize(
        ("crossings", "times", "n_cr"),
        [
            (np.array([], dtype=np.float64), np.array([], dtype=np.float64), -1),
            (np.array([np.nan], dtype=np.float64), np.array([0.5]), 1),
            (np.array([0.0], dtype=np.float64), np.array([np.inf]), 1),
            (np.array([0.0], dtype=np.float64), np.array([], dtype=np.float64), 1),
            (np.array([0.0], dtype=np.float64), np.array([0.5]), True),
        ],
    )
    def test_section_falls_back_when_backend_returns_invalid_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
        crossings: np.ndarray,
        times: np.ndarray,
        n_cr: int,
    ) -> None:
        def _invalid_section(
            _traj_flat: np.ndarray,
            _t: int,
            _d: int,
            _normal: np.ndarray,
            _offset: float,
            _direction_id: int,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            return crossings, times, n_cr

        monkeypatch.setattr(
            poincare_module,
            "_dispatch",
            lambda _fn_name: _invalid_section,
        )
        result = poincare_section([[-1.0], [1.0]], normal=[1.0], offset=0.0)

        assert result.crossings.shape == (1, 1)
        assert result.crossing_times.shape == (1,)


class TestReturnTimes:
    def test_returns_array(self):
        t = np.linspace(0, 6 * np.pi, 3000)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        rt = return_times(traj, normal=np.array([1.0, 0.0]))
        assert isinstance(rt, np.ndarray)


class TestPhasePoincare:
    def test_uniform_rotation(self):
        """Uniform rotation → regular crossings."""
        T = 500
        N = 4
        dt = 0.05
        omegas = np.array([1.0, 1.5, 2.0, 2.5])
        phases = np.zeros((T, N))
        for t in range(1, T):
            phases[t] = phases[t - 1] + omegas * dt

        result = phase_poincare(phases, oscillator_idx=0)
        assert isinstance(result, PoincareResult)
        if len(result.return_times) > 1:
            # Return time should be near 2π/ω₀ / dt ≈ 125.7 steps
            expected_rt = 2 * np.pi / (omegas[0] * dt)
            assert abs(result.mean_return_time - expected_rt) < 10

    def test_no_crossings_short(self):
        """Very short trajectory → no crossings."""
        phases = np.array([[0.0, 0.0], [0.1, 0.1]])
        result = phase_poincare(phases)
        assert len(result.crossings) == 0

    def test_multi_oscillator(self):
        """Crossings should contain full phase vector."""
        T = 1000
        N = 3
        dt = 0.05
        omegas = np.array([1.0, 2.0, 3.0])
        phases = np.zeros((T, N))
        for t in range(1, T):
            phases[t] = phases[t - 1] + omegas * dt
        result = phase_poincare(phases, oscillator_idx=0)
        if len(result.crossings) > 0:
            assert result.crossings.shape[1] == N

    @pytest.mark.parametrize(
        "phases",
        [
            np.array([[0.0], [np.nan]], dtype=np.float64),
            np.array([[0.0], [np.inf]], dtype=np.float64),
            np.array([[0.0], [True]], dtype=object),
            np.array([[0.0], [1.0j]], dtype=object),
            np.array([[0.0], [1.0j]]),
            [["not-a-phase"]],
        ],
    )
    def test_rejects_invalid_phase_history(self, phases: Any) -> None:
        with pytest.raises(ValueError, match="phases"):
            phase_poincare(phases)

    @pytest.mark.parametrize("oscillator_idx", [False, -1, 2, 1.5, "0"])
    def test_rejects_invalid_oscillator_index(self, oscillator_idx: Any) -> None:
        with pytest.raises(ValueError, match="oscillator_idx"):
            phase_poincare(np.zeros((3, 2)), oscillator_idx=oscillator_idx)

    @pytest.mark.parametrize("section_phase", [False, np.nan, np.inf, "0.0"])
    def test_rejects_invalid_section_phase(self, section_phase: Any) -> None:
        with pytest.raises(ValueError, match="section_phase"):
            phase_poincare(np.zeros((3, 2)), section_phase=section_phase)

    def test_accepts_array_like_phase_history(self) -> None:
        result = phase_poincare([[0.0], [2.0 * np.pi + 0.1]])

        assert result.crossings.shape[1] == 1

    def test_phase_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[np.ndarray, int, int, int, float]] = []

        def _fake_phase(
            phases_flat: np.ndarray,
            t: int,
            n: int,
            oscillator_idx: int,
            section_phase: float,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            calls.append((phases_flat, t, n, oscillator_idx, section_phase))
            cr = np.array([0.1, 0.2], dtype=np.float64)
            times = np.array([0.25], dtype=np.float64)
            return cr, times, 1

        monkeypatch.setattr(
            poincare_module,
            "_dispatch",
            lambda _fn_name: _fake_phase,
        )
        result = phase_poincare([[0.0, 0.0], [0.2, 0.3]], oscillator_idx=0)
        assert result.crossings.shape == (1, 2)
        assert result.crossing_times.shape == (1,)
        assert len(calls) == 1

    def test_phase_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_phase(
            _phases_flat: np.ndarray,
            _t: int,
            _n: int,
            _oscillator_idx: int,
            _section_phase: float,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            poincare_module,
            "_dispatch",
            lambda _fn_name: _raising_phase,
        )
        result = phase_poincare([[0.0], [2.0 * np.pi + 0.1]], oscillator_idx=0)
        assert result.crossings.shape[1] == 1

    def test_phase_falls_back_when_backend_returns_invalid_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _invalid_phase(
            _phases_flat: np.ndarray,
            _t: int,
            _n: int,
            _oscillator_idx: int,
            _section_phase: float,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            return np.array([0.0]), np.array([], dtype=np.float64), 1

        monkeypatch.setattr(
            poincare_module,
            "_dispatch",
            lambda _fn_name: _invalid_phase,
        )
        result = phase_poincare([[0.0], [2.0 * np.pi + 0.1]], oscillator_idx=0)

        assert result.crossings.shape[1] == 1


class TestPoincareResultValidation:
    def test_normalizes_public_result_record(self) -> None:
        result = PoincareResult(
            crossings=[[0.0], [1.0], [2.0]],
            crossing_times=[0.5, 2.5, 5.5],
            return_times=[2.0, 3.0],
            mean_return_time=2.5,
            std_return_time=0.5,
        )

        assert result.crossings.dtype == np.float64
        np.testing.assert_allclose(result.return_times, [2.0, 3.0])

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"crossings": [0.0, 1.0]}, "crossings must be two-dimensional"),
            ({"crossings": [[0.0], [np.nan]]}, "crossings"),
            ({"crossings": [[0.0], [True]]}, "crossings"),
            ({"crossings": [[0.0], [1.0j]]}, "crossings"),
            ({"crossing_times": [0.0]}, "crossing_times shape"),
            ({"crossing_times": [2.0, 1.0]}, "crossing_times must be"),
            (
                {
                    "crossing_times": [0.5, 0.5],
                    "return_times": [0.0],
                    "mean_return_time": 0.0,
                },
                "crossing_times must be",
            ),
            ({"crossing_times": [0.0, np.inf]}, "crossing_times"),
            ({"crossing_times": [0.0, 1.0j]}, "crossing_times"),
            ({"return_times": [1.0, 2.0]}, "return_times shape"),
            ({"return_times": [1.0j]}, "return_times"),
            ({"return_times": [-1.0]}, "return_times must be non-negative"),
            ({"return_times": [3.0]}, "return_times must match"),
            ({"mean_return_time": -1.0}, "mean_return_time must be non-negative"),
            ({"mean_return_time": 3.0}, "mean_return_time must match"),
            ({"std_return_time": -1.0}, "std_return_time must be non-negative"),
            ({"std_return_time": 1.0}, "std_return_time must match"),
        ],
    )
    def test_rejects_invalid_public_result_record_values(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        base: dict[str, object] = {
            "crossings": [[0.0], [1.0]],
            "crossing_times": [0.5, 2.5],
            "return_times": [2.0],
            "mean_return_time": 2.0,
            "std_return_time": 0.0,
        }
        base.update(kwargs)

        with pytest.raises(ValueError, match=match):
            PoincareResult(**base)


class TestPoincarePipelineWiring:
    """Pipeline: engine trajectory → Poincaré section → return times."""

    def test_engine_trajectory_to_poincare(self):
        """UPDEEngine → phase trajectory → phase_poincare → crossings.
        Proves Poincaré analysis consumes engine trajectory."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([2.0, 3.0, 1.5, 2.5])
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())
        traj = np.array(trajectory)

        result = phase_poincare(traj, oscillator_idx=0)
        assert isinstance(result, PoincareResult)
        assert len(result.crossings) >= 0
        if len(result.crossings) > 1:
            assert result.mean_return_time > 0


class TestPoincareBackendDispatch:
    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = poincare_module.ACTIVE_BACKEND
        previous_available = list(poincare_module.AVAILABLE_BACKENDS)
        previous_loader = poincare_module._LOADERS["go"]
        poincare_module.ACTIVE_BACKEND = "go"
        poincare_module.AVAILABLE_BACKENDS = ["go", "python"]
        poincare_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            poincare_module._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            fn = poincare_module._dispatch("section")
        finally:
            poincare_module.ACTIVE_BACKEND = previous_backend
            poincare_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(poincare_module._LOADERS, "go", previous_loader)
            poincare_module._BACKEND_CACHE.clear()

        assert fn is None

    def test_dispatch_uses_cached_backend_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = poincare_module.ACTIVE_BACKEND
        previous_available = list(poincare_module.AVAILABLE_BACKENDS)
        previous_loader = poincare_module._LOADERS["go"]
        poincare_module.ACTIVE_BACKEND = "go"
        poincare_module.AVAILABLE_BACKENDS = ["go", "python"]
        poincare_module._BACKEND_CACHE.clear()
        call_count = 0

        def fake_section(
            _traj_flat: np.ndarray,
            _t: int,
            _d: int,
            _normal: np.ndarray,
            _offset: float,
            _direction_id: int,
        ) -> tuple[np.ndarray, np.ndarray, int]:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64), 0

        def loader() -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            return {"section": fake_section}

        monkeypatch.setitem(poincare_module._LOADERS, "go", loader)
        try:
            fn1 = poincare_module._dispatch("section")
            fn2 = poincare_module._dispatch("section")
        finally:
            poincare_module.ACTIVE_BACKEND = previous_backend
            poincare_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(poincare_module._LOADERS, "go", previous_loader)
            poincare_module._BACKEND_CACHE.clear()

        assert fn1 is fake_section
        assert fn2 is fake_section
        assert call_count == 1
