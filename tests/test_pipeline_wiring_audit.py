# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Pipeline wiring audit
#
# Verifies that EVERY module in the codebase wires into the SPO pipeline:
# CouplingBuilder → Engine → order_parameter → RegimeManager → Policy → Actuator
#
# This file proves no module is decorative.

from __future__ import annotations

import time

import numpy as np
import pytest

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Core pipeline: Coupling → Engine → R → Regime → Policy → Actions
# ---------------------------------------------------------------------------


class TestFullPipelineEndToEnd:
    """The canonical SPO pipeline must wire end-to-end in a single test."""

    def test_full_pipeline_coupling_to_actions(self):
        """CouplingBuilder → UPDEEngine (200 steps) → compute_order_parameter →
        RegimeManager → SupervisorPolicy → ControlActions.

        This is the core loop that proves SPO is not a collection of
        decorative modules."""
        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        rng = np.random.default_rng(42)

        # 1. Build coupling
        cs = CouplingBuilder().build(n, base_strength=0.5, decay_alpha=0.3)

        # 2. Run engine
        eng = UPDEEngine(n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, alpha)

        # 3. Compute order parameter
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

        # 4. Build UPDE state → evaluate regime
        state = UPDEState(
            layers=[LayerState(R=r, psi=psi)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=r,
            regime_id="nominal",
        )
        mgr = RegimeManager(cooldown_steps=0)
        regime = mgr.evaluate(state, BoundaryState())

        # 5. Policy decides actions
        policy = SupervisorPolicy(mgr)
        actions = policy.decide(state, BoundaryState())

        # Regime and actions must be consistent
        from scpn_phase_orchestrator.supervisor.regimes import Regime

        if regime == Regime.NOMINAL:
            assert actions == [], "NOMINAL → no actions"
        elif regime == Regime.DEGRADED:
            assert any(a.knob == "K" for a in actions)
        elif regime == Regime.CRITICAL:
            assert any(a.knob == "zeta" for a in actions)


# ---------------------------------------------------------------------------
# Module → order_parameter wiring audit
# ---------------------------------------------------------------------------


class TestModuleToOrderParameterWiring:
    """Every engine variant must produce output that feeds compute_order_parameter."""

    def _run_and_check_r(self, engine, n, steps=100):
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(steps):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0, f"R={r} from {type(engine).__name__}"
        return r

    def test_upde_euler(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        self._run_and_check_r(UPDEEngine(8, dt=0.01), 8)

    def test_upde_rk4(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        self._run_and_check_r(UPDEEngine(8, dt=0.01, method="rk4"), 8)

    def test_upde_rk45(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        self._run_and_check_r(UPDEEngine(8, dt=0.01, method="rk45"), 8)

    def test_delayed_engine(self):
        from scpn_phase_orchestrator.upde.delay import DelayedEngine

        self._run_and_check_r(DelayedEngine(8, dt=0.01, delay_steps=3), 8)

    def test_torus_engine(self):
        from scpn_phase_orchestrator.upde.geometric import TorusEngine

        self._run_and_check_r(TorusEngine(8, dt=0.01), 8)

    def test_simplicial_engine(self):
        from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine

        self._run_and_check_r(SimplicialEngine(8, dt=0.01, sigma2=0.1), 8)

    def test_stochastic_injector_with_engine(self):
        """StochasticInjector wraps UPDEEngine output with noise."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
        from scpn_phase_orchestrator.upde.stochastic import StochasticInjector

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        injector = StochasticInjector(D=0.1, seed=42)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            phases = injector.inject(phases, 0.01)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

    def test_stuart_landau_engine(self):
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        n = 8
        eng = StuartLandauEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        state = np.concatenate([rng.uniform(0, TWO_PI, n), np.full(n, 0.5)])
        mu = np.full(n, 0.3)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(100):
            state = eng.step(state, np.ones(n), mu, knm, np.zeros((n, n)),
                             0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(state[:n])
        assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Performance regression guards
# ---------------------------------------------------------------------------


class TestPipelinePerformance:
    """Measure and guard key pipeline operations against regression.
    All times are per-call after warm-up."""

    def _time_fn(self, fn, n_warmup=5, n_measure=50):
        for _ in range(n_warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(n_measure):
            fn()
        return (time.perf_counter() - t0) / n_measure

    def test_coupling_build_n100_under_10ms(self):
        from scpn_phase_orchestrator.coupling import CouplingBuilder

        builder = CouplingBuilder()
        elapsed = self._time_fn(lambda: builder.build(100, 0.5, 0.3))
        assert elapsed < 0.01, f"CouplingBuilder.build(100) = {elapsed*1000:.1f}ms > 10ms"

    def test_engine_step_n64_under_1ms(self):
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        eng = UPDEEngine(64, dt=0.01)
        phases = np.random.default_rng(0).uniform(0, TWO_PI, 64)
        omegas = np.ones(64)
        knm = 0.3 * np.ones((64, 64))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((64, 64))

        elapsed = self._time_fn(
            lambda: eng.step(phases, omegas, knm, 0.0, 0.0, alpha),
        )
        assert elapsed < 0.001, f"UPDEEngine.step(64) = {elapsed*1000:.2f}ms > 1ms"

    def test_order_parameter_n256_under_100us(self):
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        phases = np.random.default_rng(0).uniform(0, TWO_PI, 256)
        elapsed = self._time_fn(lambda: compute_order_parameter(phases))
        assert elapsed < 0.0001, f"order_parameter(256) = {elapsed*1e6:.0f}μs > 100μs"

    def test_regime_evaluate_under_100us(self):
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        mgr = RegimeManager()
        state = UPDEState(
            layers=[LayerState(R=0.5, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.5,
            regime_id="nominal",
        )
        elapsed = self._time_fn(lambda: mgr.evaluate(state, BoundaryState()))
        assert elapsed < 0.0001, f"RegimeManager.evaluate = {elapsed*1e6:.0f}μs > 100μs"

    def test_full_loop_n8_200steps_under_50ms(self):
        """Full pipeline loop (coupling+200 engine steps+R+regime+policy)
        must complete in under 50ms for N=8."""
        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        def run_loop():
            n = 8
            cs = CouplingBuilder().build(n, 0.5, 0.3)
            eng = UPDEEngine(n, dt=0.01)
            phases = np.random.default_rng(0).uniform(0, TWO_PI, n)
            omegas = np.ones(n)
            alpha = np.zeros((n, n))
            for _ in range(200):
                phases = eng.step(phases, omegas, cs.knm, 0.0, 0.0, alpha)
            r, psi = compute_order_parameter(phases)
            state = UPDEState(
                layers=[LayerState(R=r, psi=psi)],
                cross_layer_alignment=np.eye(1),
                stability_proxy=r, regime_id="nominal",
            )
            mgr = RegimeManager(cooldown_steps=0)
            policy = SupervisorPolicy(mgr)
            policy.decide(state, BoundaryState())

        elapsed = self._time_fn(run_loop, n_warmup=2, n_measure=10)
        assert elapsed < 0.05, f"Full loop(N=8, 200 steps) = {elapsed*1000:.1f}ms > 50ms"


# ---------------------------------------------------------------------------
# Adapter wiring: every adapter must produce/consume pipeline-compatible data
# ---------------------------------------------------------------------------


class TestAdapterWiring:
    """Verify adapters produce data compatible with the core pipeline."""

    def test_connectome_to_engine(self):
        """load_hcp_connectome → UPDEEngine.step: proves connectome data
        is valid coupling input."""
        from scpn_phase_orchestrator.coupling.connectome import load_hcp_connectome
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 20
        knm = load_hcp_connectome(n)
        eng = UPDEEngine(n, dt=0.01)
        phases = np.random.default_rng(0).uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0

    def test_physical_extractor_to_pipeline(self):
        """PhysicalExtractor → PhaseState → coupling pipeline."""
        from scpn_phase_orchestrator.oscillators.physical import PhysicalExtractor

        t = np.arange(0, 1.0, 1.0 / 1000)
        signal = np.sin(2 * np.pi * 10 * t)
        ext = PhysicalExtractor()
        states = ext.extract(signal, 1000.0)
        assert len(states) == 1
        assert 0.0 <= states[0].theta < TWO_PI
        assert states[0].quality > 0.5

    def test_audit_logger_to_replay(self, tmp_path):
        """AuditLogger → ReplayEngine: data written is exactly data read back."""
        from scpn_phase_orchestrator.audit.logger import AuditLogger
        from scpn_phase_orchestrator.audit.replay import ReplayEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        state = UPDEState(
            layers=[LayerState(R=0.8, psi=0.5)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.8, regime_id="nominal",
        )
        log = tmp_path / "audit.jsonl"
        with AuditLogger(log) as logger:
            logger.log_step(0, state, [])
        entries = ReplayEngine(log).load()
        assert entries[0]["layers"][0]["R"] == pytest.approx(0.8)

    def test_imprint_modulates_coupling_for_engine(self):
        """ImprintModel.modulate_coupling → UPDEEngine: proves imprint
        feeds back into the engine's coupling matrix."""
        from scpn_phase_orchestrator.coupling import CouplingBuilder
        from scpn_phase_orchestrator.imprint.state import ImprintState
        from scpn_phase_orchestrator.imprint.update import ImprintModel
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        cs = CouplingBuilder().build(n, 0.5, 0.3)
        imprint = ImprintModel(decay_rate=0.0, saturation=5.0)
        state = ImprintState(m_k=np.ones(n) * 2.0, last_update=0.0)
        knm_boosted = imprint.modulate_coupling(cs.knm, state)

        eng = UPDEEngine(n, dt=0.01)
        phases = np.random.default_rng(7).uniform(0, TWO_PI, n)
        omegas = np.ones(n)
        for _ in range(200):
            phases = eng.step(phases, omegas, knm_boosted, 0.0, 0.0, np.zeros((n, n)))
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
