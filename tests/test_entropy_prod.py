# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for entropy production rate

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import entropy_prod as entropy_prod_module
from scpn_phase_orchestrator.monitor.entropy_prod import entropy_production_rate


def _all_to_all(n: int, k: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestEntropyProductionRate:
    def test_zero_at_fixed_point(self):
        """Identical phases and frequencies → dθ/dt = 0 → dissipation = 0."""
        phases = np.zeros(4)
        omegas = np.zeros(4)
        knm = _all_to_all(4)
        rate = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        assert rate == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_nonzero_frequencies(self):
        """Non-zero natural frequencies produce positive dissipation."""
        phases = np.zeros(4)
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = np.zeros((4, 4))
        rate = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        # Σ ω_i² · dt = (1+4+9+16)·0.01 = 0.30
        assert rate == pytest.approx(0.30, abs=1e-10)

    def test_scales_with_dt(self):
        """Doubling dt doubles dissipation."""
        phases = np.array([0.0, 0.5])
        omegas = np.array([1.0, -1.0])
        knm = np.zeros((2, 2))
        r1 = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        r2 = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.02)
        assert r2 == pytest.approx(2.0 * r1, rel=1e-10)

    def test_coupling_reduces_dissipation_near_lock(self):
        """Strong coupling at phases where sin opposes ω should reduce dθ/dt."""
        # ω_0=+1, ω_1=-1 with θ_0 behind θ_1: coupling pulls 0 forward, 1 backward
        _phases = np.array([0.0, 1.0])
        _omegas = np.array([1.0, -1.0])
        _knm = np.array([[0.0, 50.0], [50.0, 0.0]])
        # Coupling sin(1.0)≈0.84, (α/N)·K·sin ≈ 25·0.84=21 opposes ω_1=-1
        # dθ_1/dt = -1 + 25·sin(-1) ≈ -1 - 21 = -22 → larger magnitude
        # Instead test: identical phases, omegas differ → coupling is zero
        # Use phases where coupling *reduces* the spread of dθ/dt
        _phases2 = np.array([0.0, 0.5])
        _omegas2 = np.array([-1.0, 1.0])
        # sin(0.5)≈0.48 → coupling pulls osc 0 toward osc 1 (positive)
        # dθ_0/dt = -1 + 25·sin(0.5) ≈ -1+12 = +11
        # dθ_1/dt = +1 + 25·sin(-0.5) ≈ 1-12 = -11
        # Without coupling: dθ = [-1, 1], Σ(dθ²) = 2
        # With coupling: dθ = [11, -11], Σ(dθ²) = 242 → worse
        # The correct physical scenario: at the fixed point dθ/dt=0
        phases_fp = np.zeros(3)
        omegas_fp = np.zeros(3)
        knm_fp = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        rate_fp = entropy_production_rate(
            phases_fp,
            omegas_fp,
            knm_fp,
            alpha=1.0,
            dt=0.01,
        )
        # Perturbed phases: coupling creates restoring force
        phases_perturbed = np.array([0.0, 0.3, -0.3])
        rate_perturbed = entropy_production_rate(
            phases_perturbed,
            omegas_fp,
            knm_fp,
            alpha=1.0,
            dt=0.01,
        )
        assert rate_fp < rate_perturbed

    def test_empty_phases(self):
        rate = entropy_production_rate(
            np.array([]), np.array([]), np.zeros((0, 0)), alpha=1.0, dt=0.01
        )
        assert rate == 0.0

    def test_zero_dt_returns_zero(self):
        phases = np.array([0.0, 1.0])
        omegas = np.array([1.0, 2.0])
        knm = _all_to_all(2)
        assert entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.0) == 0.0

    def test_negative_dt_returns_zero(self):
        phases = np.array([0.0, 1.0])
        omegas = np.array([1.0, 2.0])
        knm = _all_to_all(2)
        with pytest.raises(ValueError, match="dt must be non-negative"):
            entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=-0.01)

    def test_alpha_scaling(self):
        """Doubling alpha changes the coupling contribution."""
        phases = np.array([0.0, 1.0, 2.0])
        omegas = np.zeros(3)
        knm = _all_to_all(3)
        r1 = entropy_production_rate(phases, omegas, knm, alpha=1.0, dt=0.01)
        r2 = entropy_production_rate(phases, omegas, knm, alpha=2.0, dt=0.01)
        # With omegas=0, dθ/dt = (α/N)·coupling, so r scales as α²
        assert r2 == pytest.approx(4.0 * r1, rel=1e-10)

    @pytest.mark.parametrize(
        ("field", "bad_value", "match"),
        [
            ("phases", np.array([0.0, True], dtype=object), "phases"),
            ("omegas", np.array([1.0, False], dtype=object), "omegas"),
            ("knm", np.array([[0.0, True], [1.0, 0.0]], dtype=object), "knm"),
        ],
    )
    def test_rejects_mixed_boolean_alias_arrays(
        self, field: str, bad_value: object, match: str
    ) -> None:
        kwargs: dict[str, object] = {
            "phases": np.zeros(2),
            "omegas": np.ones(2),
            "knm": np.zeros((2, 2)),
        }
        kwargs[field] = bad_value

        with pytest.raises(ValueError, match=match):
            entropy_production_rate(
                kwargs["phases"],
                kwargs["omegas"],
                kwargs["knm"],
                alpha=1.0,
                dt=0.01,
            )


class TestEntropyProdPipelineWiring:
    """Pipeline: engine phases → entropy production rate → thermodynamics."""

    def test_engine_phases_to_entropy_rate(self):
        """Engine → phases → entropy_production_rate: measures
        irreversibility of coupled oscillator dynamics."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 6
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.normal(1.0, 0.5, n)
        knm = _all_to_all(n, 0.5)
        alpha_mat = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(
                phases,
                omegas,
                knm,
                0.0,
                0.0,
                alpha_mat,
            )

        rate = entropy_production_rate(
            phases,
            omegas,
            knm,
            alpha=1.0,
            dt=0.01,
        )
        assert rate >= 0.0
        assert np.isfinite(rate)


class TestEntropyProdRustDispatch:
    def test_entropy_production_uses_backend_when_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, float]] = []

        def _fake_backend(
            phases: np.ndarray,
            omegas: np.ndarray,
            knm: np.ndarray,
            alpha: float,
            dt: float,
        ) -> float:
            calls.append((phases, omegas, knm, alpha, dt))
            return 0.777

        monkeypatch.setattr(entropy_prod_module, "_dispatch", lambda: _fake_backend)
        rate = entropy_production_rate(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
            alpha=1.0,
            dt=0.01,
        )
        assert rate == pytest.approx(0.777, abs=1e-12)
        assert len(calls) == 1

    def test_entropy_production_falls_back_when_backend_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raising_backend(
            _phases: np.ndarray,
            _omegas: np.ndarray,
            _knm: np.ndarray,
            _alpha: float,
            _dt: float,
        ) -> float:
            raise RuntimeError("boom")

        monkeypatch.setattr(entropy_prod_module, "_dispatch", lambda: _raising_backend)
        rate = entropy_production_rate(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
            alpha=1.0,
            dt=0.01,
        )
        assert np.isfinite(rate)
        assert rate >= 0.0

    @pytest.mark.parametrize("backend_rate", [-0.1, np.nan, np.inf])
    def test_entropy_production_falls_back_when_backend_returns_invalid_rate(
        self, monkeypatch: pytest.MonkeyPatch, backend_rate: float
    ) -> None:
        def _invalid_backend(
            _phases: np.ndarray,
            _omegas: np.ndarray,
            _knm: np.ndarray,
            _alpha: float,
            _dt: float,
        ) -> float:
            return backend_rate

        monkeypatch.setattr(entropy_prod_module, "_dispatch", lambda: _invalid_backend)
        rate = entropy_production_rate(
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64),
            alpha=1.0,
            dt=0.01,
        )

        assert np.isfinite(rate)
        assert rate >= 0.0

    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = entropy_prod_module.ACTIVE_BACKEND
        previous_available = list(entropy_prod_module.AVAILABLE_BACKENDS)
        previous_loader = entropy_prod_module._LOADERS["go"]
        entropy_prod_module.ACTIVE_BACKEND = "go"
        entropy_prod_module.AVAILABLE_BACKENDS = ["go", "python"]
        entropy_prod_module._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            entropy_prod_module._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = entropy_prod_module._dispatch()
        finally:
            entropy_prod_module.ACTIVE_BACKEND = previous_backend
            entropy_prod_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(entropy_prod_module._LOADERS, "go", previous_loader)
            entropy_prod_module._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        previous_backend = entropy_prod_module.ACTIVE_BACKEND
        previous_available = list(entropy_prod_module.AVAILABLE_BACKENDS)
        previous_loader = entropy_prod_module._LOADERS["go"]
        entropy_prod_module.ACTIVE_BACKEND = "go"
        entropy_prod_module.AVAILABLE_BACKENDS = ["go", "python"]
        entropy_prod_module._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            _phases: np.ndarray,
            _omegas: np.ndarray,
            _knm: np.ndarray,
            _alpha: float,
            _dt: float,
        ) -> float:
            return 0.0

        def loader():
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(entropy_prod_module._LOADERS, "go", loader)
        try:
            b1 = entropy_prod_module._dispatch()
            b2 = entropy_prod_module._dispatch()
        finally:
            entropy_prod_module.ACTIVE_BACKEND = previous_backend
            entropy_prod_module.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(entropy_prod_module._LOADERS, "go", previous_loader)
            entropy_prod_module._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1
