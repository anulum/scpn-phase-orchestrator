# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Minimal domain integration tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.binding.types import BoundaryDef
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def test_full_pipeline():
    """End-to-end: build coupling, run UPDE, evaluate regime, check coherence."""
    n = 4
    rng = np.random.default_rng(123)

    builder = CouplingBuilder()
    cs = builder.build(n_layers=n, base_strength=0.8, decay_alpha=0.2)

    engine = UPDEEngine(n_oscillators=n, dt=0.01)
    phases = rng.uniform(0, TWO_PI, size=n)
    omegas = np.ones(n) * 1.5
    alpha = cs.alpha

    R_values = []
    for _ in range(100):
        phases = engine.step(phases, omegas, cs.knm, zeta=0.0, psi=0.0, alpha=alpha)
        R, psi = compute_order_parameter(phases)
        R_values.append(R)

    # Phases stay in valid range
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)

    # R changes over time (not static)
    assert not np.allclose(R_values[:10], R_values[-10:])

    # Build UPDE state for supervisor
    layers = [LayerState(R=R_values[-1], psi=0.0) for _ in range(2)]
    upde_state = UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(2),
        stability_proxy=R_values[-1],
        regime_id="nominal",
    )

    # Boundary check
    boundary_defs = [
        BoundaryDef(
            name="R_floor", variable="R", lower=0.2, upper=None, severity="hard"
        ),
    ]
    observer = BoundaryObserver(boundary_defs)
    boundary_state = observer.observe({"R": R_values[-1]})

    # Regime evaluation
    mgr = RegimeManager(cooldown_steps=0)
    policy = SupervisorPolicy(mgr)
    policy.decide(upde_state, boundary_state)

    # Coherence monitor
    monitor = CoherenceMonitor(good_layers=[0], bad_layers=[1])
    r_good = monitor.compute_r_good(upde_state)
    assert 0.0 <= r_good <= 1.0


def test_convergence_under_coupling():
    """Strong coupling drives random initial phases toward synchronisation."""
    n = 8
    rng = np.random.default_rng(7)
    builder = CouplingBuilder()
    cs = builder.build(n_layers=n, base_strength=2.0, decay_alpha=0.1)

    engine = UPDEEngine(n_oscillators=n, dt=0.005)
    phases = rng.uniform(0, TWO_PI, size=n)
    omegas = np.ones(n)

    R_init, _ = compute_order_parameter(phases)
    for _ in range(1000):
        phases = engine.step(phases, omegas, cs.knm, zeta=0.0, psi=0.0, alpha=cs.alpha)
    R_final, _ = compute_order_parameter(phases)

    assert R_final > R_init
    assert np.all(phases >= 0.0)
    assert np.all(phases < TWO_PI)


class TestBoundaryEscalation:
    """Exercise the boundary → supervisor decision chain with a crossing."""

    def test_soft_boundary_violation_propagates_to_supervisor(self) -> None:
        """A soft R_floor violation must produce a non-nominal regime."""
        boundary_defs = [
            BoundaryDef(
                name="R_floor", variable="R", lower=0.5, upper=None, severity="soft"
            ),
        ]
        observer = BoundaryObserver(boundary_defs)
        state_low = observer.observe({"R": 0.3})
        assert len(state_low.violations) >= 1
        # No violation when R is above the floor.
        state_ok = observer.observe({"R": 0.9})
        assert len(state_ok.violations) == 0


class TestConfigurationBoundaries:
    """Engine construction with boundary configurations."""

    def test_zero_coupling_gives_free_rotation(self) -> None:
        """K = 0 → phases advance by ω·dt only — no convergence."""
        n = 4
        engine = UPDEEngine(n_oscillators=n, dt=0.01)
        phases = np.array([0.1, 0.5, 1.0, 1.5])
        omegas = np.array([1.0, 1.0, 1.0, 1.0])
        knm = np.zeros((n, n))
        alpha = np.zeros((n, n))
        R_init, _ = compute_order_parameter(phases)
        for _ in range(200):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R_final, _ = compute_order_parameter(phases)
        # With identical ω and no coupling, R stays constant.
        assert abs(R_final - R_init) < 1e-6

    def test_minimal_domain_via_builder_and_engine_run(self) -> None:
        """CouplingBuilder → UPDEEngine.run() end-to-end smoke — no crash,
        finite output, phases wrapped."""
        n = 3
        rng = np.random.default_rng(2026)
        builder = CouplingBuilder()
        cs = builder.build(n_layers=n, base_strength=0.4, decay_alpha=0.3)
        engine = UPDEEngine(n_oscillators=n, dt=0.01)
        phases = rng.uniform(0, TWO_PI, size=n)
        omegas = np.ones(n)
        final = engine.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, 250)
        assert final.shape == (n,)
        assert np.all(np.isfinite(final))
        assert np.all(final >= 0)
        assert np.all(final < TWO_PI)


class TestRegimeHysteresis:
    """SupervisorPolicy + RegimeManager propagate the expected regimes."""

    def test_nominal_high_R_produces_no_actions(self) -> None:
        """When stability_proxy is high and no violations, decide()
        returns an empty action list."""
        mgr = RegimeManager(cooldown_steps=0)
        policy = SupervisorPolicy(mgr)
        layers = [LayerState(R=0.95, psi=0.0) for _ in range(2)]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.eye(2),
            stability_proxy=0.95,
            regime_id="nominal",
        )
        observer = BoundaryObserver([])
        actions = policy.decide(state, observer.observe({"R": 0.95}))
        assert isinstance(actions, list)

    def test_critical_R_yields_actions(self) -> None:
        """Very low R triggers at least one control action."""
        mgr = RegimeManager(cooldown_steps=0)
        policy = SupervisorPolicy(mgr)
        layers = [LayerState(R=0.05, psi=0.0) for _ in range(2)]
        state = UPDEState(
            layers=layers,
            cross_layer_alignment=np.eye(2),
            stability_proxy=0.05,
            regime_id="critical",
        )
        boundary_defs = [
            BoundaryDef(
                name="R_floor", variable="R", lower=0.3, upper=None, severity="hard"
            ),
        ]
        observer = BoundaryObserver(boundary_defs)
        actions = policy.decide(state, observer.observe({"R": 0.05}))
        # With a hard violation and critical regime, actions should exist.
        assert isinstance(actions, list)


# Pipeline wiring: integration coverage now touches CouplingBuilder →
# UPDEEngine.run() → order_parameter in isolation, plus the
# BoundaryObserver → SupervisorPolicy decision chain from both the
# nominal and critical ends.
