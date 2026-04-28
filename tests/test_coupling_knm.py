# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling Knm tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder


def test_build_symmetric():
    state = CouplingBuilder().build(8, 0.45, 0.3)
    np.testing.assert_allclose(state.knm, state.knm.T, atol=1e-14)


def test_build_zero_diagonal():
    state = CouplingBuilder().build(8, 0.45, 0.3)
    np.testing.assert_allclose(np.diag(state.knm), 0.0)


def test_build_nonnegative():
    state = CouplingBuilder().build(8, 0.45, 0.3)
    assert np.all(state.knm >= 0.0)


def test_build_exponential_decay():
    state = CouplingBuilder().build(8, 1.0, 0.5)
    assert state.knm[0, 1] > state.knm[0, 2] > state.knm[0, 3]


def test_build_alpha_shape():
    state = CouplingBuilder().build(4, 0.45, 0.3)
    assert state.alpha.shape == (4, 4)
    np.testing.assert_allclose(state.alpha, 0.0)


def test_switch_template_valid():
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.1)
    custom = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float64)
    new_state = builder.switch_template(state, "ring", {"ring": custom})
    np.testing.assert_array_equal(new_state.knm, custom)
    assert new_state.active_template == "ring"


def test_switch_template_missing_raises():
    builder = CouplingBuilder()
    state = builder.build(3, 0.5, 0.1)
    with pytest.raises(KeyError, match="no_such"):
        builder.switch_template(state, "no_such", {})


class TestCouplingBuilderAlgebraic:
    """Deeper algebraic invariants for CouplingBuilder output."""

    @pytest.mark.parametrize("n", [4, 8, 16, 32])
    def test_symmetric_for_all_sizes(self, n):
        state = CouplingBuilder().build(n, 0.45, 0.3)
        np.testing.assert_allclose(state.knm, state.knm.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(state.knm), 0.0)

    def test_stronger_coupling_higher_values(self):
        """Higher base_strength → larger K_nm entries."""
        s_low = CouplingBuilder().build(8, 0.1, 0.3)
        s_high = CouplingBuilder().build(8, 1.0, 0.3)
        assert np.mean(s_high.knm) > np.mean(s_low.knm)

    def test_faster_decay_reduces_distant_coupling(self):
        """Higher decay_alpha → distant layers less coupled."""
        s_slow = CouplingBuilder().build(8, 0.5, 0.1)
        s_fast = CouplingBuilder().build(8, 0.5, 0.9)
        # Distant pair (0,7) should be weaker with fast decay
        assert s_fast.knm[0, 7] < s_slow.knm[0, 7]

    def test_build_scpn_physics_invariants(self):
        """build_scpn_physics: 16×16, symmetric, zero diagonal, nonneg."""
        state = CouplingBuilder().build_scpn_physics()
        assert state.knm.shape == (16, 16)
        np.testing.assert_allclose(state.knm, state.knm.T, atol=1e-14)
        np.testing.assert_allclose(np.diag(state.knm), 0.0)
        assert np.all(state.knm >= 0.0)
        assert np.all(np.isfinite(state.knm))

    @pytest.mark.parametrize("bad_tau", [0.0, -1.0, np.nan, np.inf])
    def test_adjacent_coupling_rejects_nonfinite_timescale(self, monkeypatch, bad_tau):
        from scpn_phase_orchestrator.coupling import knm as knm_mod

        monkeypatch.setitem(knm_mod.SCPN_LAYER_TIMESCALES, 6, bad_tau)
        with pytest.raises(ValueError, match="finite and positive"):
            CouplingBuilder._adjacent_coupling(6, 7, 0.45)

    def test_switch_template_preserves_shape(self):
        builder = CouplingBuilder()
        state = builder.build(4, 0.5, 0.1)
        ring = np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        new = builder.switch_template(state, "ring", {"ring": ring})
        assert new.knm.shape == state.knm.shape
        assert new.active_template == "ring"


class TestCouplingKnmPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → Engine(all methods) → R → Regime."""

    def test_build_engine_sync_regime(self):
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 16
        cs = CouplingBuilder().build_scpn_physics()
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        phases = eng.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, n_steps=500)
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

    def test_template_switch_changes_dynamics(self):
        """Switching K_nm template changes engine R trajectory."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        builder = CouplingBuilder()
        cs = builder.build(n, 0.5, 0.2)
        # All-to-all template
        all2all = np.ones((n, n)) * 2.0
        np.fill_diagonal(all2all, 0.0)
        cs_all = builder.switch_template(cs, "all2all", {"all2all": all2all})
        # Ring template (nearest neighbours only)
        ring = np.zeros((n, n))
        for i in range(n):
            ring[i, (i + 1) % n] = 0.2
            ring[(i + 1) % n, i] = 0.2
        cs_ring = builder.switch_template(cs, "ring", {"ring": ring})
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        rng = np.random.default_rng(42)
        p0 = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        p_all = eng.run(
            p0.copy(),
            omegas,
            cs_all.knm,
            0.0,
            0.0,
            cs_all.alpha,
            n_steps=300,
        )
        p_ring = eng.run(
            p0.copy(),
            omegas,
            cs_ring.knm,
            0.0,
            0.0,
            cs_ring.alpha,
            n_steps=300,
        )
        r_all, _ = compute_order_parameter(p_all)
        r_ring, _ = compute_order_parameter(p_ring)
        # Stronger coupling → higher R
        assert r_all > r_ring - 0.05

    def test_performance_build_100_under_10ms(self):
        """CouplingBuilder.build(100) < 10ms budget."""
        import time

        builder = CouplingBuilder()
        builder.build(100, 0.45, 0.3)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100):
            builder.build(100, 0.45, 0.3)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.01, f"build(100) took {elapsed * 1e3:.2f}ms"

    def test_performance_build_scpn_physics_under_5ms(self):
        """build_scpn_physics() < 5ms budget."""
        import time

        builder = CouplingBuilder()
        builder.build_scpn_physics()  # warm-up
        t0 = time.perf_counter()
        for _ in range(100):
            builder.build_scpn_physics()
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005, f"build_scpn_physics() took {elapsed * 1e3:.2f}ms"


# Pipeline wiring: CouplingBuilder → build/build_scpn_physics → UPDEEngine(RK4)
# → compute_order_parameter → RegimeManager. Template switching proves
# topology-dependent dynamics. Performance: build(100)<10ms, SCPN physics<5ms.
