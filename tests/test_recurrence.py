# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence analysis tests

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import recurrence as recurrence_module
from scpn_phase_orchestrator.monitor.recurrence import (
    RQAResult,
    cross_recurrence_matrix,
    cross_rqa,
    recurrence_matrix,
    rqa,
)


class _ArrayConversionFailure:
    def __array__(self, dtype: object | None = None) -> np.ndarray[Any, Any]:
        raise ValueError("synthetic conversion failure")


class TestRecurrenceMatrix:
    def test_identical_trajectory_is_all_recurrent(self):
        """Constant trajectory → every point recurs with every other."""
        traj = np.ones((20, 2))
        R = recurrence_matrix(traj, epsilon=0.1)
        assert R.shape == (20, 20)
        assert R.all()

    def test_diverging_trajectory_sparse(self):
        """Linearly increasing → few recurrences for small ε."""
        traj = np.arange(50).astype(float)[:, np.newaxis]
        R = recurrence_matrix(traj, epsilon=0.5)
        # Only adjacent points (|i-j|<=0.5) should recur
        np.fill_diagonal(R, False)
        assert R.sum() == 0  # no off-diagonal recurrences

    def test_periodic_trajectory_structured(self):
        """Sine wave → recurrence at period intervals."""
        t = np.linspace(0, 4 * np.pi, 200)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        R = recurrence_matrix(traj, epsilon=0.3)
        rr = R.sum() / R.size
        assert 0.01 < rr < 0.5  # structured, not trivial

    def test_angular_metric(self):
        """Angular metric uses chord distance on circle."""
        traj = np.array([0.0, np.pi, 0.1])[:, np.newaxis]
        R = recurrence_matrix(traj, epsilon=0.3, metric="angular")
        assert R[0, 2]  # 0.0 and 0.1 are close
        assert not R[0, 1]  # 0 and π are far

    def test_1d_input(self):
        """1D array input should work."""
        traj = np.zeros(10)
        R = recurrence_matrix(traj, epsilon=0.1)
        assert R.shape == (10, 10)

    @pytest.mark.parametrize(
        ("trajectory", "match"),
        [
            (np.array([0.0, np.nan], dtype=np.float64), "trajectory"),
            (np.array([[0.0], [np.inf]], dtype=np.float64), "trajectory"),
            (np.array([0.0, True], dtype=object), "trajectory"),
            (np.array(["0.0", "1.0"], dtype=str), "trajectory"),
            (np.array([0.0, "1.0"], dtype=object), "trajectory"),
            (np.array([0.0 + 1.0j, 1.0 + 0.0j]), "trajectory"),
            (np.zeros((1, 1, 1), dtype=np.float64), "trajectory"),
            ([["not-a-state"]], "trajectory"),
        ],
    )
    def test_rejects_invalid_trajectory_buffers(
        self,
        trajectory: Any,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            recurrence_matrix(trajectory, epsilon=0.1)

    @pytest.mark.parametrize("epsilon", [False, np.nan, np.inf, "0.1", -0.1])
    def test_rejects_invalid_epsilon(self, epsilon: Any) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            recurrence_matrix(np.zeros(4), epsilon=epsilon)

    @pytest.mark.parametrize("metric", ["manhattan", "", None])
    def test_rejects_unknown_metric(self, metric: Any) -> None:
        with pytest.raises(ValueError, match="metric"):
            recurrence_matrix(np.zeros(4), epsilon=0.1, metric=metric)

    def test_accepts_array_like_trajectory(self) -> None:
        R = recurrence_matrix([0.0, 0.05, 0.2], epsilon=0.1)

        assert R.shape == (3, 3)
        assert R.dtype == bool

    def test_non_rust_dispatch_uses_flat_buffers(self, monkeypatch) -> None:
        calls: list[tuple[int, int, float, bool]] = []

        def fake_rm(traj_flat, t, d, epsilon, angular):
            calls.append((traj_flat.shape[0], int(t), float(epsilon), bool(angular)))
            return (
                np.array(
                    [1, 0, 0, 1],
                    dtype=np.uint8,
                )
                .reshape(2, 2)
                .ravel()
            )

        import scpn_phase_orchestrator.monitor.recurrence as rec_mod

        monkeypatch.setattr(
            rec_mod,
            "_dispatch",
            lambda fn_name: fake_rm if fn_name == "rm" else None,
        )
        R = recurrence_matrix(
            np.array([[0.0], [1.0]]),
            epsilon=0.5,
            metric="angular",
        )

        np.testing.assert_array_equal(R, np.array([[True, False], [False, True]]))
        assert calls == [(2, 2, 0.5, True)]

    @pytest.mark.parametrize(
        "backend_output",
        [
            np.array([1, 0, 1], dtype=np.uint8),
            np.array([1, 0, 0, 2], dtype=np.uint8),
            np.array([1, 0, 0, np.nan], dtype=np.float64),
            np.array(["1", "0", "0", "1"], dtype=str),
            np.array([1, "0", 0, 1], dtype=object),
            np.array(["one", "zero", "zero", "one"], dtype=object),
            _ArrayConversionFailure(),
            np.array([0, 0, 0, 1], dtype=np.uint8),
            np.array([1, 1, 0, 1], dtype=np.uint8),
        ],
    )
    def test_backend_invalid_matrix_payload_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch, backend_output: Any
    ) -> None:
        def fake_rm(_traj_flat, _t, _d, _epsilon, _angular):
            return backend_output

        import scpn_phase_orchestrator.monitor.recurrence as rec_mod

        monkeypatch.setattr(
            rec_mod,
            "_dispatch",
            lambda fn_name: fake_rm if fn_name == "rm" else None,
        )
        with pytest.raises(ValueError):
            recurrence_matrix(np.array([[0.0], [1.0]]), epsilon=0.5)

    def test_backend_expected_shape_mismatch_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_rm(_traj_flat, _t, _d, _epsilon, _angular):
            return np.array([1, 0, 0, 1], dtype=np.uint8)

        import scpn_phase_orchestrator.monitor.recurrence as rec_mod

        monkeypatch.setattr(
            rec_mod,
            "_dispatch",
            lambda fn_name: fake_rm if fn_name == "rm" else None,
        )
        monkeypatch.setattr(
            rec_mod,
            "_expected_recurrence_matrix",
            lambda *_args, **_kwargs: np.ones((1, 1), dtype=bool),
        )

        with pytest.raises(ValueError, match="expected output shape"):
            recurrence_matrix(np.array([[0.0], [1.0]]), epsilon=0.5)

    def test_private_alias_helpers_fail_closed_on_conversion_errors(self) -> None:
        assert not recurrence_module._contains_boolean_alias(_ArrayConversionFailure())
        assert not recurrence_module._contains_numeric_string_alias(
            _ArrayConversionFailure()
        )
        assert not recurrence_module._is_numeric_string_alias(1.0)

    def test_dispatch_skips_unknown_and_empty_backends(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_active = recurrence_module.ACTIVE_BACKEND
        previous_available = recurrence_module.AVAILABLE_BACKENDS
        monkeypatch.setattr(recurrence_module, "ACTIVE_BACKEND", "unknown")
        monkeypatch.setattr(recurrence_module, "AVAILABLE_BACKENDS", ["empty"])
        monkeypatch.setitem(recurrence_module._LOADERS, "empty", lambda: {})
        recurrence_module._BACKEND_CACHE.clear()
        try:
            assert recurrence_module._dispatch("rm") is None
        finally:
            recurrence_module._BACKEND_CACHE.clear()
            recurrence_module.ACTIVE_BACKEND = previous_active
            recurrence_module.AVAILABLE_BACKENDS = previous_available

    def test_julia_loader_returns_backend_functions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_name = (
            "scpn_phase_orchestrator.experimental.accelerators.monitor."
            "_recurrence_julia"
        )
        fake_module = ModuleType(module_name)

        def fake_recurrence_matrix() -> None:
            return None

        def fake_cross_recurrence_matrix() -> None:
            return None

        fake_module.recurrence_matrix_julia = fake_recurrence_matrix
        fake_module.cross_recurrence_matrix_julia = fake_cross_recurrence_matrix
        monkeypatch.setitem(sys.modules, module_name, fake_module)
        monkeypatch.setattr(recurrence_module, "require_juliacall_main", lambda: None)

        loaded = recurrence_module._load_julia_fns()

        assert loaded == {
            "rm": fake_recurrence_matrix,
            "cross_rm": fake_cross_recurrence_matrix,
        }


class TestRQA:
    def test_periodic_high_determinism(self):
        """Periodic signal → high DET (diagonal lines dominate)."""
        t = np.linspace(0, 6 * np.pi, 300)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = rqa(traj, epsilon=0.2)
        assert isinstance(result, RQAResult)
        assert result.determinism > 0.4
        assert result.max_diagonal > 5

    def test_random_low_determinism(self):
        """Random noise → low DET."""
        rng = np.random.default_rng(42)
        traj = rng.normal(0, 1, (100, 2))
        result = rqa(traj, epsilon=0.3)
        assert result.determinism < 0.5

    def test_laminar_trapping(self):
        """Trajectory with stuck segments → laminarity > 0."""
        # Stay at origin for 20 steps, then move, then return
        traj = np.zeros((60, 1))
        traj[20:40, 0] = np.linspace(0, 5, 20)
        result = rqa(traj, epsilon=0.1, v_min=3)
        assert result.laminarity > 0
        assert result.trapping_time > 0

    def test_empty_trajectory(self):
        """Very short trajectory still returns valid result."""
        traj = np.array([[0.0], [1.0]])
        result = rqa(traj, epsilon=0.01)
        assert result.recurrence_rate == 0.0

    def test_empty_trajectory_avoids_zero_division(self):
        result = rqa(np.array([], dtype=np.float64), epsilon=0.1)
        assert result.recurrence_rate == 0.0
        assert result.determinism == 0.0
        assert result.avg_diagonal == 0.0
        assert result.max_diagonal == 0
        assert result.max_vertical == 0

    def test_angular_rqa(self):
        """RQA with angular metric for phase data."""
        t = np.linspace(0, 8 * np.pi, 200)
        traj = t[:, np.newaxis] % (2 * np.pi)
        result = rqa(traj, epsilon=0.2, metric="angular")
        assert result.recurrence_rate > 0
        assert result.determinism > 0

    def test_rqa_fields(self):
        """All RQA fields present and finite."""
        t = np.linspace(0, 4 * np.pi, 100)
        traj = np.column_stack([np.sin(t), np.cos(t)])
        result = rqa(traj, epsilon=0.3)
        assert np.isfinite(result.recurrence_rate)
        assert np.isfinite(result.determinism)
        assert np.isfinite(result.avg_diagonal)
        assert np.isfinite(result.entropy_diagonal)
        assert np.isfinite(result.laminarity)
        assert np.isfinite(result.trapping_time)
        assert isinstance(result.max_diagonal, int)
        assert isinstance(result.max_vertical, int)

    @pytest.mark.parametrize("line_min", [False, 0, -1, 1.5, "2"])
    def test_rejects_invalid_line_minimum(self, line_min: Any) -> None:
        with pytest.raises(ValueError, match="l_min"):
            rqa(np.zeros(4), epsilon=0.1, l_min=line_min)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"recurrence_rate": -0.1}, "recurrence_rate"),
            ({"recurrence_rate": 1.1}, "recurrence_rate"),
            ({"determinism": False}, "determinism"),
            ({"determinism": np.nan}, "determinism"),
            ({"avg_diagonal": "0.1"}, "avg_diagonal"),
            ({"avg_diagonal": -0.1}, "avg_diagonal"),
            ({"max_diagonal": -1}, "max_diagonal"),
            ({"max_diagonal": 1.5}, "max_diagonal"),
            ({"entropy_diagonal": np.inf}, "entropy_diagonal"),
            ({"laminarity": 1.1}, "laminarity"),
            ({"trapping_time": -0.1}, "trapping_time"),
            ({"max_vertical": -1}, "max_vertical"),
            ({"avg_diagonal": 3.0, "max_diagonal": 2}, "avg_diagonal"),
            (
                {"avg_diagonal": 0.0, "entropy_diagonal": 0.1, "max_diagonal": 0},
                "entropy_diagonal",
            ),
            ({"trapping_time": 3.0, "max_vertical": 2}, "trapping_time"),
        ],
    )
    def test_rejects_invalid_public_rqa_result_values(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        base: dict[str, object] = {
            "recurrence_rate": 0.5,
            "determinism": 0.4,
            "avg_diagonal": 2.0,
            "max_diagonal": 3,
            "entropy_diagonal": 0.1,
            "laminarity": 0.2,
            "trapping_time": 1.5,
            "max_vertical": 2,
        }
        base.update(kwargs)

        with pytest.raises(ValueError, match=match):
            RQAResult(**base)


class TestCrossRecurrence:
    def test_identical_trajectories(self):
        """Same trajectory → cross-recurrence = auto-recurrence."""
        t = np.linspace(0, 2 * np.pi, 50)
        traj = np.sin(t)[:, np.newaxis]
        CR = cross_recurrence_matrix(traj, traj, epsilon=0.1)
        R = recurrence_matrix(traj, epsilon=0.1)
        np.testing.assert_array_equal(CR, R)

    def test_phase_shifted_recurrence(self):
        """Phase-shifted signals should show off-diagonal structure."""
        t = np.linspace(0, 4 * np.pi, 100)
        a = np.sin(t)[:, np.newaxis]
        b = np.sin(t + np.pi / 4)[:, np.newaxis]
        CR = cross_recurrence_matrix(a, b, epsilon=0.3)
        assert CR.sum() > 0

    def test_cross_rqa_returns_result(self):
        """cross_rqa returns valid RQAResult."""
        t = np.linspace(0, 4 * np.pi, 100)
        a = np.column_stack([np.sin(t), np.cos(t)])
        b = np.column_stack([np.sin(t + 0.2), np.cos(t + 0.2)])
        result = cross_rqa(a, b, epsilon=0.3)
        assert isinstance(result, RQAResult)
        assert result.recurrence_rate > 0

    def test_uncorrelated_low_crqa(self):
        """Uncorrelated trajectories → low cross-determinism."""
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, (80, 2))
        b = rng.normal(0, 1, (80, 2))
        result = cross_rqa(a, b, epsilon=0.3)
        assert result.determinism < 0.5

    def test_cross_recurrence_angular_metric(self):
        """Angular metric in cross_recurrence_matrix."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 2 * np.pi, (20, 2))
        b = rng.uniform(0, 2 * np.pi, (20, 2))
        CR = cross_recurrence_matrix(a, b, epsilon=1.0, metric="angular")
        assert CR.shape == (20, 20)
        assert CR.dtype == bool

    def test_cross_rqa_angular_metric(self):
        """Angular metric in cross_rqa."""
        t = np.linspace(0, 4 * np.pi, 50)
        a = np.column_stack([np.sin(t), np.cos(t)])
        b = np.column_stack([np.sin(t + 0.1), np.cos(t + 0.1)])
        result = cross_rqa(a, b, epsilon=1.0, metric="angular")
        assert isinstance(result, RQAResult)
        assert 0.0 <= result.recurrence_rate <= 1.0

    @pytest.mark.parametrize(
        ("traj_a", "traj_b", "match"),
        [
            (np.array([0.0, np.nan]), np.zeros(2), "traj_a"),
            (np.zeros(2), np.array([0.0, np.inf]), "traj_b"),
            (np.array([0.0, True], dtype=object), np.zeros(2), "traj_a"),
            (np.zeros(2), np.array([0.0, False], dtype=object), "traj_b"),
            (np.array(["0.0", "1.0"], dtype=str), np.zeros(2), "traj_a"),
            (np.zeros(2), np.array([0.0, "1.0"], dtype=object), "traj_b"),
            (np.array([0.0 + 1.0j, 1.0]), np.zeros(2), "traj_a"),
            (np.zeros(2), np.array([0.0, 1.0j]), "traj_b"),
            ([["not-a-state"]], np.zeros((1, 1)), "traj_a"),
        ],
    )
    def test_cross_rejects_invalid_trajectory_buffers(
        self,
        traj_a: Any,
        traj_b: Any,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            cross_recurrence_matrix(traj_a, traj_b, epsilon=0.1)

    @pytest.mark.parametrize("epsilon", [False, np.nan, np.inf, "0.1", -0.1])
    def test_cross_rejects_invalid_epsilon(self, epsilon: Any) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            cross_recurrence_matrix(np.zeros(4), np.zeros(4), epsilon=epsilon)

    def test_cross_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="trajectories must match"):
            cross_recurrence_matrix(np.zeros((2, 1)), np.zeros((3, 1)), epsilon=0.1)

    def test_cross_empty_trajectory_returns_empty_matrix(self) -> None:
        result = cross_recurrence_matrix(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
            epsilon=0.1,
        )

        assert result.shape == (0, 0)
        assert result.dtype == bool

    @pytest.mark.parametrize("metric", ["manhattan", "", None])
    def test_cross_rejects_unknown_metric(self, metric: Any) -> None:
        with pytest.raises(ValueError, match="metric"):
            cross_recurrence_matrix(
                np.zeros(4),
                np.zeros(4),
                epsilon=0.1,
                metric=metric,
            )

    def test_cross_backend_invalid_matrix_payload_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_cross_rm(_a_flat, _b_flat, _t, _d, _epsilon, _angular):
            return np.array([1, 0, 0, 2], dtype=np.uint8)

        monkeypatch.setattr(
            recurrence_module,
            "_dispatch",
            lambda fn_name: fake_cross_rm if fn_name == "cross_rm" else None,
        )
        with pytest.raises(ValueError):
            cross_recurrence_matrix(np.array([0.0, 1.0]), np.array([0.0, 2.0]), 0.5)


class TestRecurrencePipelineWiring:
    """Pipeline: engine trajectory → delay embed → RQA → determinism."""

    def test_engine_trajectory_to_rqa(self):
        """UPDEEngine → trajectory → delay_embed → rqa: determinism
        quantifies recurrence in coupled oscillator dynamics."""
        from scpn_phase_orchestrator.monitor.embedding import delay_embed
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.array([1.0, 1.5, 2.0, 0.5])
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        trajectory = []
        for _ in range(300):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(float(phases[0]))
        signal = np.array(trajectory)

        emb = delay_embed(signal, delay=5, dimension=3)
        result = rqa(emb, epsilon=0.3)
        assert isinstance(result, RQAResult)
        assert 0.0 <= result.recurrence_rate <= 1.0
        assert 0.0 <= result.determinism <= 1.0


class TestRecurrenceBackendFallbacks:
    def test_recurrence_matrix_backend_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_rm(
            _traj_flat: np.ndarray, _t: int, _d: int, _epsilon: float, _angular: bool
        ) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            recurrence_module,
            "_dispatch",
            lambda fn_name: _raising_rm if fn_name == "rm" else None,
        )
        traj = np.array([[0.0], [1.0]], dtype=np.float64)
        R = recurrence_matrix(traj, epsilon=0.5, metric="euclidean")
        np.testing.assert_array_equal(R, np.array([[True, False], [False, True]]))

    def test_cross_recurrence_backend_failure_falls_back_to_python(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_cross_rm(
            _a_flat: np.ndarray,
            _b_flat: np.ndarray,
            _t: int,
            _d: int,
            _epsilon: float,
            _angular: bool,
        ) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            recurrence_module,
            "_dispatch",
            lambda fn_name: _raising_cross_rm if fn_name == "cross_rm" else None,
        )
        traj_a = np.array([[0.0], [1.0]], dtype=np.float64)
        traj_b = np.array([[0.0], [2.0]], dtype=np.float64)
        CR = cross_recurrence_matrix(traj_a, traj_b, epsilon=0.5, metric="euclidean")
        np.testing.assert_array_equal(CR, np.array([[True, False], [False, False]]))

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = recurrence_module.ACTIVE_BACKEND
        previous_available = list(recurrence_module.AVAILABLE_BACKENDS)
        previous_loader = recurrence_module._LOADERS["go"]
        recurrence_module.ACTIVE_BACKEND = "go"
        recurrence_module.AVAILABLE_BACKENDS = ["go", "python"]
        recurrence_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            recurrence_module._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            fn = recurrence_module._dispatch("rm")
        finally:
            recurrence_module.ACTIVE_BACKEND = previous_backend
            recurrence_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(recurrence_module._LOADERS, "go", previous_loader)
            recurrence_module._BACKEND_CACHE.clear()

        assert fn is None

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = recurrence_module.ACTIVE_BACKEND
        previous_available = list(recurrence_module.AVAILABLE_BACKENDS)
        previous_loader = recurrence_module._LOADERS["go"]
        recurrence_module.ACTIVE_BACKEND = "go"
        recurrence_module.AVAILABLE_BACKENDS = ["go", "python"]
        recurrence_module._BACKEND_CACHE.clear()
        call_count = 0

        def fake_rm(
            _traj_flat: np.ndarray,
            _t: int,
            _d: int,
            _epsilon: float,
            _angular: bool,
        ) -> np.ndarray:
            return np.array([1], dtype=np.uint8)

        def loader() -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            return {"rm": fake_rm}

        monkeypatch.setitem(recurrence_module._LOADERS, "go", loader)
        try:
            fn1 = recurrence_module._dispatch("rm")
            fn2 = recurrence_module._dispatch("rm")
        finally:
            recurrence_module.ACTIVE_BACKEND = previous_backend
            recurrence_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(recurrence_module._LOADERS, "go", previous_loader)
            recurrence_module._BACKEND_CACHE.clear()

        assert fn1 is fake_rm
        assert fn2 is fake_rm
        assert call_count == 1
