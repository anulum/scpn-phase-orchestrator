# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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
