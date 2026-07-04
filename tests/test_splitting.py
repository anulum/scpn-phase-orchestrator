# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Splitting integrator tests

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

SPLITTING_REFERENCE = Path("docs/reference/api/upde_splitting.md")


class _ArrayProtocolFailure:
    def __array__(self, *_args, **_kwargs):
        raise TypeError("array protocol failed")


def _coupled_knm(n: int, k: float = 0.5) -> np.ndarray:
    knm = np.full((n, n), k)
    np.fill_diagonal(knm, 0.0)
    return knm


class TestSplittingEngine:
    def test_output_in_range(self):
        n = 6
        eng = SplittingEngine(n, dt=0.01)
        phases = np.linspace(0, 2 * np.pi, n, endpoint=False)
        omegas = np.ones(n)
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        assert np.all(result >= 0)
        assert np.all(result < 2 * np.pi)

    def test_zero_coupling_pure_rotation(self):
        n = 4
        dt = 0.01
        eng = SplittingEngine(n, dt=dt)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (phases + dt * omegas) % (2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_synchronization(self):
        n = 8
        eng = SplittingEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _coupled_knm(n, k=1.0)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=500)
        R, _ = compute_order_parameter(phases)
        assert R > 0.9

    def test_agrees_with_monolithic_rk4(self):
        n = 4
        dt = 0.001
        rng = np.random.default_rng(42)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n) * 1.5
        knm = _coupled_knm(n, k=0.3)
        alpha = np.zeros((n, n))

        split = SplittingEngine(n, dt=dt)
        mono = UPDEEngine(n, dt=dt, method="rk4")

        ps = phases0.copy()
        pm = phases0.copy()
        for _ in range(100):
            ps = split.step(ps, omegas, knm, 0.0, 0.0, alpha)
            pm = mono.step(pm, omegas, knm, 0.0, 0.0, alpha)

        # Should agree to O(dt²) — splitting and monolithic RK4 are both 2nd/4th order
        diff = np.abs(ps - pm)
        diff = np.minimum(diff, 2 * np.pi - diff)
        assert np.max(diff) < 0.01

    def test_run_n_steps(self):
        n = 4
        eng = SplittingEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.ones(n)
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        assert result.shape == (n,)

    def test_external_drive(self):
        n = 4
        eng = SplittingEngine(n, dt=0.01)
        phases = np.zeros(n)
        omegas = np.zeros(n)
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        # psi=1.0 so sin(1.0 - 0) ≈ 0.841 — nonzero drive
        result = eng.step(phases, omegas, knm, 1.0, 1.0, alpha)
        assert not np.allclose(result, phases)

    def test_preserves_sync(self):
        n = 6
        eng = SplittingEngine(n, dt=0.01)
        phases = np.full(n, 1.0)
        omegas = np.ones(n) * 2.0
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        result = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=200)
        R, _ = compute_order_parameter(result)
        assert R > 0.99

    def test_negative_dt_falls_back_to_python_reference(self, monkeypatch):
        import scpn_phase_orchestrator.upde.splitting as split_mod

        monkeypatch.setattr(
            split_mod,
            "_dispatch",
            lambda: (_ for _ in ()).throw(
                AssertionError("dispatch was used for negative dt"),
            ),
        )

        n = 4
        dt = -0.02
        eng = SplittingEngine(n, dt=dt)
        phases = np.array([0.0, 0.5, 1.0, 1.5])
        omegas = np.array([1.0, 2.0, 3.0, 4.0])
        knm = np.zeros((n, n), dtype=np.float64)
        alpha = np.zeros((n, n), dtype=np.float64)
        result = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        expected = (phases + dt * omegas) % (2 * np.pi)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_invalid_n_steps_rejected(self):
        n = 4
        eng = SplittingEngine(n, dt=0.01)
        with pytest.raises(ValueError, match="n_steps"):
            eng.run(
                np.zeros(n),
                np.ones(n),
                _coupled_knm(n),
                0.0,
                0.0,
                np.zeros((n, n)),
                n_steps=0,
            )

    def test_accelerated_backend_phase_domain_is_rejected(self, monkeypatch):
        import scpn_phase_orchestrator.upde.splitting as split_mod

        def malformed_backend(
            _phases,
            _omegas,
            _knm_flat,
            _alpha_flat,
            _n,
            _zeta,
            _psi,
            _dt,
            _n_steps,
        ):
            return np.array([0.0, 2.0 * np.pi], dtype=np.float64)

        monkeypatch.setattr(split_mod, "_dispatch", lambda: malformed_backend)

        eng = SplittingEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="backend output phases"):
            eng.step(
                np.zeros(2),
                np.ones(2),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.zeros((2, 2)),
            )

    def test_accelerated_backend_numeric_string_output_is_rejected(
        self,
        monkeypatch,
    ):
        import scpn_phase_orchestrator.upde.splitting as split_mod

        def malformed_backend(
            _phases,
            _omegas,
            _knm_flat,
            _alpha_flat,
            _n,
            _zeta,
            _psi,
            _dt,
            _n_steps,
        ):
            return np.array(["0.1", "0.2"], dtype=object)

        monkeypatch.setattr(split_mod, "_dispatch", lambda: malformed_backend)

        eng = SplittingEngine(2, dt=0.01)
        with pytest.raises(ValueError, match="numeric-string"):
            eng.step(
                np.zeros(2),
                np.ones(2),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.zeros((2, 2)),
            )

    @pytest.mark.parametrize(
        ("phases", "match"),
        [
            (np.array(["", "bad"], dtype=object), "finite float array"),
            (_ArrayProtocolFailure(), "finite float array"),
        ],
    )
    def test_uncoercible_state_arrays_raise_public_validation_error(
        self,
        phases,
        match: str,
    ):
        eng = SplittingEngine(2, dt=0.01)

        with pytest.raises(ValueError, match=match):
            eng.run(
                phases,
                np.ones(2),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.zeros((2, 2)),
                n_steps=1,
            )

    @pytest.mark.parametrize(
        ("backend_output", "match"),
        [
            ("0.1", "numeric-string"),
            (np.array(["bad", "payload"], dtype=object), "finite phase vector"),
            (np.array([0.1], dtype=np.float64), "backend output shape"),
            (np.array([0.1, np.nan], dtype=np.float64), "finite phases"),
        ],
    )
    def test_accelerated_backend_malformed_outputs_are_rejected(
        self,
        monkeypatch,
        backend_output,
        match: str,
    ):
        import scpn_phase_orchestrator.upde.splitting as split_mod

        def malformed_backend(
            _phases,
            _omegas,
            _knm_flat,
            _alpha_flat,
            _n,
            _zeta,
            _psi,
            _dt,
            _n_steps,
        ):
            return backend_output

        monkeypatch.setattr(split_mod, "_dispatch", lambda: malformed_backend)

        eng = SplittingEngine(2, dt=0.01)
        with pytest.raises(ValueError, match=match):
            eng.step(
                np.zeros(2),
                np.ones(2),
                np.zeros((2, 2)),
                0.0,
                0.0,
                np.zeros((2, 2)),
            )


class TestSplittingDispatch:
    def test_dispatch_falls_back_to_python_when_loader_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.splitting as split_mod

        previous_backend = split_mod.ACTIVE_BACKEND
        previous_available = list(split_mod.AVAILABLE_BACKENDS)
        previous_loader = split_mod._LOADERS["go"]
        split_mod.ACTIVE_BACKEND = "go"
        split_mod.AVAILABLE_BACKENDS = ["go", "python"]
        split_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            split_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go backend unavailable")),
        )
        try:
            backend = split_mod._dispatch()
        finally:
            split_mod.ACTIVE_BACKEND = previous_backend
            split_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(split_mod._LOADERS, "go", previous_loader)
            split_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_returns_none_when_all_non_python_backends_fail(self, monkeypatch):
        import scpn_phase_orchestrator.upde.splitting as split_mod

        previous_backend = split_mod.ACTIVE_BACKEND
        previous_available = list(split_mod.AVAILABLE_BACKENDS)
        previous_loader = split_mod._LOADERS["go"]
        split_mod.ACTIVE_BACKEND = "go"
        split_mod.AVAILABLE_BACKENDS = []
        split_mod._BACKEND_CACHE.clear()
        monkeypatch.setitem(
            split_mod._LOADERS,
            "go",
            lambda: (_ for _ in ()).throw(ImportError("go unavailable")),
        )
        try:
            backend = split_mod._dispatch()
        finally:
            split_mod.ACTIVE_BACKEND = previous_backend
            split_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(split_mod._LOADERS, "go", previous_loader)
            split_mod._BACKEND_CACHE.clear()

        assert backend is None

    def test_dispatch_uses_cached_loader_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import scpn_phase_orchestrator.upde.splitting as split_mod

        previous_backend = split_mod.ACTIVE_BACKEND
        previous_available = list(split_mod.AVAILABLE_BACKENDS)
        previous_loader = split_mod._LOADERS["go"]
        split_mod.ACTIVE_BACKEND = "go"
        split_mod.AVAILABLE_BACKENDS = ["go", "python"]
        split_mod._BACKEND_CACHE.clear()
        call_count = 0

        def fake_backend(
            phases: np.ndarray,
            _omegas: np.ndarray,
            _knm_flat: np.ndarray,
            _alpha_flat: np.ndarray,
            _n: int,
            _zeta: float,
            _psi: float,
            _dt: float,
            _n_steps: int,
        ) -> np.ndarray:
            return np.asarray(phases, dtype=np.float64)

        def loader():
            nonlocal call_count
            call_count += 1
            return fake_backend

        monkeypatch.setitem(split_mod._LOADERS, "go", loader)
        try:
            b1 = split_mod._dispatch()
            b2 = split_mod._dispatch()
        finally:
            split_mod.ACTIVE_BACKEND = previous_backend
            split_mod.AVAILABLE_BACKENDS = previous_available
            monkeypatch.setitem(split_mod._LOADERS, "go", previous_loader)
            split_mod._BACKEND_CACHE.clear()

        assert b1 is fake_backend
        assert b2 is fake_backend
        assert call_count == 1


class TestSplittingSymplecticProperties:
    """Symplectic splitting: energy-like invariant tests."""

    def test_splitting_reversibility(self):
        """Forward + backward with dt → −dt should return near original."""
        n = 4
        dt = 0.001
        fwd = SplittingEngine(n, dt=dt)
        bwd = SplittingEngine(n, dt=-dt)
        rng = np.random.default_rng(99)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n) * 2.0
        knm = _coupled_knm(n, k=0.3)
        alpha = np.zeros((n, n))
        phases = phases0.copy()
        for _ in range(50):
            phases = fwd.step(phases, omegas, knm, 0.0, 0.0, alpha)
        for _ in range(50):
            phases = bwd.step(phases, omegas, knm, 0.0, 0.0, alpha)
        diff = np.abs(phases - phases0)
        diff = np.minimum(diff, 2 * np.pi - diff)
        assert np.max(diff) < 0.05, f"Reversibility error: {np.max(diff)}"

    def test_splitting_finite_for_many_steps(self):
        """SplittingEngine remains finite after 2000 steps."""
        n = 8
        eng = SplittingEngine(n, dt=0.01)
        rng = np.random.default_rng(77)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.uniform(-3, 3, n)
        knm = _coupled_knm(n, k=0.5)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=2000)
        assert np.all(np.isfinite(phases))
        assert np.all(phases >= 0.0)
        assert np.all(phases < 2 * np.pi)


class TestSplittingPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → SplittingEngine → R → RegimeManager."""

    def test_coupling_splitting_regime(self):
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        n = 16
        cb = CouplingBuilder()
        cs = cb.build(n_layers=n, base_strength=0.5, decay_alpha=0.2)
        eng = SplittingEngine(n, dt=0.01)
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        phases = eng.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, n_steps=300)
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        rm = RegimeManager(hysteresis=0.05)
        regime = rm.evaluate(state, BoundaryState())
        assert regime.name in {"NOMINAL", "DEGRADED", "CRITICAL", "RECOVERY"}

    def test_splitting_vs_monolithic_R_convergence(self):
        """Splitting and UPDEEngine(rk4) → same R within tolerance."""
        n = 8
        rng = np.random.default_rng(55)
        phases0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _coupled_knm(n, k=0.5)
        alpha = np.zeros((n, n))
        dt = 0.005
        split_eng = SplittingEngine(n, dt=dt)
        mono_eng = UPDEEngine(n, dt=dt, method="rk4")
        ps = split_eng.run(phases0.copy(), omegas, knm, 0.0, 0.0, alpha, n_steps=400)
        pm = mono_eng.run(phases0.copy(), omegas, knm, 0.0, 0.0, alpha, n_steps=400)
        r_split, _ = compute_order_parameter(ps)
        r_mono, _ = compute_order_parameter(pm)
        assert abs(r_split - r_mono) < 0.1, (
            f"Split R={r_split:.4f} vs Mono R={r_mono:.4f}"
        )

    def test_performance_splitting_step_64_under_3ms(self):
        """SplittingEngine.step(64 oscillators) < 3ms budget.

        Budget relaxed from 1ms to 3ms: Windows CI runners consistently
        exceed 1ms due to timer resolution and virtualisation overhead.
        """
        import time

        n = 64
        eng = SplittingEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = _coupled_knm(n)
        alpha = np.zeros((n, n))
        eng.step(phases, omegas, knm, 0.0, 0.0, alpha)  # warm-up
        t0 = time.perf_counter()
        for _ in range(500):
            eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        elapsed = (time.perf_counter() - t0) / 500
        assert elapsed < 3e-3, f"split.step(64) took {elapsed * 1e3:.2f}ms"


def test_splitting_reference_documents_numeric_string_contract() -> None:
    doc = SPLITTING_REFERENCE.read_text(encoding="utf-8")

    assert "numeric-string aliases before float coercion" in doc
    assert "direct Go/Julia/Mojo" in doc


# Pipeline wiring: SplittingEngine tests exercise full pipeline
# CouplingBuilder → SplittingEngine → compute_order_parameter → RegimeManager.
# Symplectic: reversibility, finite stability. Cross-check: splitting vs RK4
# R-convergence. Performance: step(64)<1ms.
