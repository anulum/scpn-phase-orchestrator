# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov guard tests

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import lyapunov as lyapunov_mod
from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard, LyapunovState


def _all_to_all(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestLyapunovFunction:
    def test_spectrum_min_steps_uses_python_fallback_path(self, monkeypatch):
        n = 4
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        omegas = np.array([1.0, 1.1, 1.2, 1.3])
        knm = _all_to_all(n, k=0.4)
        alpha = np.zeros((n, n))

        monkeypatch.setattr(lyapunov_mod, "_dispatch", lambda: None)
        spectrum = lyapunov_mod.lyapunov_spectrum(
            phases, omegas, knm, alpha, n_steps=1, qr_interval=10
        )

        assert spectrum.shape == (n,)
        assert np.all(spectrum == 0.0)

    def test_synchronized_minimum(self):
        phases = np.zeros(4)
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        # All phases equal → cos(0)=1 → V is at minimum
        assert state.V < 0
        assert state.in_basin

    def test_anti_phase_maximum(self):
        phases = np.array([0.0, np.pi, 0.0, np.pi])
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        # Anti-phase → higher V (less negative)
        assert state.V > -1.0

    def test_V_decreases_toward_sync(self):
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        # Start spread, move toward sync
        phases1 = np.array([0.0, 0.3, 0.6, 0.9])
        phases2 = np.array([0.0, 0.1, 0.2, 0.3])
        s1 = guard.evaluate(phases1, knm)
        s2 = guard.evaluate(phases2, knm)
        assert s2.V < s1.V
        assert s2.dV_dt < 0

    def test_dV_dt_zero_on_first_call(self):
        guard = LyapunovGuard()
        state = guard.evaluate(np.zeros(3), _all_to_all(3))
        assert state.dV_dt == 0.0

    def test_basin_inside(self):
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff < np.pi / 2
        assert state.in_basin

    def test_basin_outside(self):
        phases = np.array([0.0, 0.0, 0.0, np.pi])
        knm = _all_to_all(4)
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff > np.pi / 2
        assert not state.in_basin

    def test_empty_phases(self):
        guard = LyapunovGuard()
        state = guard.evaluate(np.array([]), np.zeros((0, 0)))
        assert state.V == 0.0
        assert state.in_basin

    def test_no_connections(self):
        phases = np.array([0.0, 1.0, 2.0])
        knm = np.zeros((3, 3))
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.V == 0.0
        assert state.max_phase_diff == 0.0

    def test_reset_clears_prev(self):
        guard = LyapunovGuard()
        guard.evaluate(np.zeros(3), _all_to_all(3))
        guard.reset()
        state = guard.evaluate(np.ones(3), _all_to_all(3))
        assert state.dV_dt == 0.0

    def test_custom_basin_threshold(self):
        phases = np.array([0.0, 1.0])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        guard = LyapunovGuard(basin_threshold=0.5)
        state = guard.evaluate(phases, knm)
        assert not state.in_basin

    @pytest.mark.parametrize(
        "basin_threshold",
        [False, 0.0, -0.1, np.nan, np.inf, "1.0"],
    )
    def test_rejects_invalid_basin_threshold(self, basin_threshold: Any) -> None:
        with pytest.raises(ValueError, match="basin_threshold"):
            LyapunovGuard(basin_threshold=basin_threshold)

    @pytest.mark.parametrize(
        ("phases", "knm", "match"),
        [
            (np.array([0.0, np.nan]), np.zeros((2, 2)), "phases"),
            (np.array([0.0, np.inf]), np.zeros((2, 2)), "phases"),
            ([["not-a-phase"]], np.zeros((1, 1)), "phases"),
            (np.array([True, False]), np.zeros((2, 2)), "phases"),
            (np.array([0.0, True], dtype=object), np.zeros((2, 2)), "phases"),
            (np.zeros(2), np.zeros((2, 1)), "knm shape"),
            (np.zeros(2), np.array([[0.0, np.nan], [0.0, 0.0]]), "knm"),
            (np.zeros(2), np.array([[False, True], [True, False]]), "knm"),
            (
                np.zeros(2),
                np.array([[0.0, True], [1.0, 0.0]], dtype=object),
                "knm",
            ),
        ],
    )
    def test_evaluate_rejects_invalid_inputs(
        self,
        phases: Any,
        knm: Any,
        match: str,
    ) -> None:
        guard = LyapunovGuard()
        with pytest.raises(ValueError, match=match):
            guard.evaluate(phases, knm)

    def test_evaluate_accepts_array_like_inputs(self) -> None:
        guard = LyapunovGuard()
        state = guard.evaluate([0.0, 0.1], [[0.0, 1.0], [1.0, 0.0]])

        assert isinstance(state.V, float)

    def test_wrapping_phase_diff(self):
        # Phases near 0 and 2π should have small diff, not large
        phases = np.array([0.1, 2 * np.pi - 0.1])
        knm = np.array([[0.0, 1.0], [1.0, 0.0]])
        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert state.max_phase_diff < 0.3
        assert state.in_basin


class TestLyapunovPipelineWiring:
    """Verify LyapunovGuard wires into the SPO engine pipeline:
    UPDEEngine → phases → LyapunovGuard → basin detection."""

    def test_engine_to_lyapunov_guard(self):
        """Run UPDEEngine 200 steps → feed phases to LyapunovGuard.
        Proves the Lyapunov monitor accepts engine output."""

        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _all_to_all(n, k=0.5)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        guard = LyapunovGuard()
        state = guard.evaluate(phases, knm)
        assert isinstance(state.V, float)
        assert isinstance(state.in_basin, bool)
        assert state.max_phase_diff >= 0.0

    def test_lyapunov_evaluate_performance(self):
        """LyapunovGuard.evaluate(N=64) must stay within CI runner budgets."""
        import os
        import sys
        import time

        n = 64
        knm = _all_to_all(n, k=0.3)
        phases = np.random.default_rng(0).uniform(0, 2 * np.pi, n)
        guard = LyapunovGuard()

        # Warm up
        guard.evaluate(phases, knm)

        t0 = time.perf_counter()
        for _ in range(100):
            guard.evaluate(phases, knm)
        elapsed = (time.perf_counter() - t0) / 100
        limit = (
            0.005 if os.getenv("CI") else 0.002 if sys.platform == "darwin" else 0.001
        )
        assert elapsed < limit, (
            f"evaluate(64) took {elapsed * 1000:.2f}ms, limit {limit * 1000:.1f}ms"
        )


class TestLyapunovStateValidation:
    def test_normalizes_public_monitor_record(self) -> None:
        state = LyapunovState(
            V=np.float64(-1.0),
            dV_dt=np.float64(-0.1),
            in_basin=True,
            max_phase_diff=np.float64(0.25),
        )

        assert state.V == -1.0
        assert state.dV_dt == -0.1
        assert state.in_basin is True
        assert state.max_phase_diff == 0.25

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"V": np.nan}, "V"),
            ({"V": False}, "V"),
            ({"dV_dt": np.inf}, "dV_dt"),
            ({"dV_dt": False}, "dV_dt"),
            ({"in_basin": 1}, "in_basin must be a boolean flag"),
            ({"max_phase_diff": -0.1}, "max_phase_diff must be non-negative"),
            ({"max_phase_diff": np.pi + 0.1}, "max_phase_diff must be <= pi"),
            ({"max_phase_diff": np.nan}, "max_phase_diff"),
        ],
    )
    def test_rejects_invalid_public_monitor_record_values(
        self, kwargs: dict[str, object], match: str
    ) -> None:
        base: dict[str, object] = {
            "V": -1.0,
            "dV_dt": 0.0,
            "in_basin": True,
            "max_phase_diff": 0.25,
        }
        base.update(kwargs)

        with pytest.raises(ValueError, match=match):
            LyapunovState(**base)


class TestLyapunovRustDispatch:
    def test_spectrum_uses_rust_backend_signature_when_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                float,
                int,
                int,
                float,
                float,
            ]
        ] = []

        def _fake_backend(
            p: np.ndarray,
            o: np.ndarray,
            k_flat: np.ndarray,
            a_flat: np.ndarray,
            dt: float,
            n_steps: int,
            qr_interval: int,
            zeta: float,
            psi: float,
        ) -> np.ndarray:
            calls.append((p, o, k_flat, a_flat, dt, n_steps, qr_interval, zeta, psi))
            return np.array([3.0, 2.0, 1.0], dtype=np.float64)

        monkeypatch.setattr(lyapunov_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(lyapunov_mod, "_dispatch", lambda: _fake_backend)
        phases = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        knm = _all_to_all(3, k=0.4)
        alpha = np.zeros((3, 3), dtype=np.float64)

        spec = lyapunov_mod.lyapunov_spectrum(
            phases, omegas, knm, alpha, dt=0.01, n_steps=10, qr_interval=2
        )
        np.testing.assert_allclose(spec, [3.0, 2.0, 1.0], atol=1e-12)
        assert len(calls) == 1
        assert calls[0][2].ndim == 1
        assert calls[0][3].ndim == 1

    def test_spectrum_falls_back_to_python_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_backend(
            *_args: object,
            **_kwargs: object,
        ) -> np.ndarray:
            raise RuntimeError("boom")

        monkeypatch.setattr(lyapunov_mod, "ACTIVE_BACKEND", "rust")
        monkeypatch.setattr(lyapunov_mod, "_dispatch", lambda: _raising_backend)
        phases = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        omegas = np.array([1.0, 1.1, 1.2], dtype=np.float64)
        knm = _all_to_all(3, k=0.4)
        alpha = np.zeros((3, 3), dtype=np.float64)

        spec = lyapunov_mod.lyapunov_spectrum(
            phases, omegas, knm, alpha, dt=0.01, n_steps=10, qr_interval=2
        )
        assert spec.shape == (3,)
        assert np.all(np.isfinite(spec))

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = lyapunov_mod.ACTIVE_BACKEND
        previous_available = list(lyapunov_mod.AVAILABLE_BACKENDS)
        previous_loader = lyapunov_mod._LOADERS["go"]
        lyapunov_mod.ACTIVE_BACKEND = "go"
        lyapunov_mod.AVAILABLE_BACKENDS = ["go", "python"]
        lyapunov_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            lyapunov_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = lyapunov_mod._dispatch()
        finally:
            lyapunov_mod.ACTIVE_BACKEND = previous_backend
            lyapunov_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(lyapunov_mod._LOADERS, "go", previous_loader)
            lyapunov_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = lyapunov_mod.ACTIVE_BACKEND
        previous_available = list(lyapunov_mod.AVAILABLE_BACKENDS)
        previous_loader = lyapunov_mod._LOADERS["go"]
        lyapunov_mod.ACTIVE_BACKEND = "go"
        lyapunov_mod.AVAILABLE_BACKENDS = ["go", "python"]
        lyapunov_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            _phases: np.ndarray,
            _omegas: np.ndarray,
            _knm: np.ndarray,
            _alpha: np.ndarray,
            _dt: float,
            _n_steps: int,
            _qr_interval: int,
            _zeta: float,
            _psi: float,
        ) -> np.ndarray:
            return np.zeros(3, dtype=np.float64)

        def loader():
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(lyapunov_mod._LOADERS, "go", loader)
        try:
            b1 = lyapunov_mod._dispatch()
            b2 = lyapunov_mod._dispatch()
        finally:
            lyapunov_mod.ACTIVE_BACKEND = previous_backend
            lyapunov_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(lyapunov_mod._LOADERS, "go", previous_loader)
            lyapunov_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1


# Salvaged module-specific behavioural contracts from deleted broad tests.
class TestLyapunovGuardValidation:
    def test_rejects_zero_basin_threshold(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be positive"):
            LyapunovGuard(basin_threshold=0.0)

    def test_rejects_negative_basin_threshold(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be positive"):
            LyapunovGuard(basin_threshold=-0.5)

    def test_default_basin_threshold_is_half_pi(self) -> None:
        m = LyapunovGuard()
        assert abs(m._basin_threshold - np.pi / 2.0) < 1e-12

    def test_rejects_basin_threshold_above_geodesic_phase_limit(self) -> None:
        with pytest.raises(ValueError, match="basin_threshold must be <= pi"):
            LyapunovGuard(basin_threshold=np.pi + 1e-6)
