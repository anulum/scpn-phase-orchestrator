# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Full domainpack cycle integration test

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.monitor.coherence import CoherenceMonitor
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi
DOMAINPACKS = Path(__file__).resolve().parent.parent / "domainpacks"


def test_minimal_domain_full_cycle() -> None:
    """Load minimal_domain → couple → UPDE → observe → regime."""
    spec = load_binding_spec(DOMAINPACKS / "minimal_domain" / "binding_spec.yaml")

    assert spec.name == "minimal_domain"
    assert len(spec.layers) == 2
    n_osc = sum(len(lay.oscillator_ids) for lay in spec.layers)
    assert n_osc == 4

    builder = CouplingBuilder()
    cs = builder.build(
        n_layers=n_osc,
        base_strength=spec.coupling.base_strength,
        decay_alpha=spec.coupling.decay_alpha,
    )

    engine = UPDEEngine(n_oscillators=n_osc, dt=spec.sample_period_s)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, size=n_osc)
    omegas = np.ones(n_osc) * 1.0

    zeta = spec.drivers.physical.get("zeta", 0.0)
    psi = spec.drivers.physical.get("psi", 0.0)

    R_history = []
    for _ in range(200):
        phases = engine.step(phases, omegas, cs.knm, zeta, psi, cs.alpha)
        R, _ = compute_order_parameter(phases)
        R_history.append(R)

    assert np.all(phases >= 0.0) and np.all(phases < TWO_PI)

    # Boundary observation
    observer = BoundaryObserver(spec.boundaries)
    boundary_state = observer.observe({"R": R_history[-1]})

    # Regime evaluation
    layer_states = [LayerState(R=R_history[-1], psi=0.0) for _ in spec.layers]
    upde_state = UPDEState(
        layers=layer_states,
        cross_layer_alignment=np.eye(len(spec.layers)),
        stability_proxy=R_history[-1],
        regime_id="nominal",
    )
    mgr = RegimeManager(cooldown_steps=0)
    regime = mgr.evaluate(upde_state, boundary_state)
    assert regime.value in ("nominal", "degraded", "critical", "recovery")

    # Coherence
    monitor = CoherenceMonitor(
        good_layers=spec.objectives.good_layers,
        bad_layers=spec.objectives.bad_layers,
    )
    r_good = monitor.compute_r_good(upde_state)
    assert 0.0 <= r_good <= 1.0


def test_geometry_walk_domainpack() -> None:
    """Load geometry_walk binding spec, verify coupling and UPDE run."""
    spec = load_binding_spec(DOMAINPACKS / "geometry_walk" / "binding_spec.yaml")
    assert spec.name == "geometry_walk"

    n_osc = sum(len(lay.oscillator_ids) for lay in spec.layers)
    builder = CouplingBuilder()
    cs = builder.build(
        n_layers=n_osc,
        base_strength=spec.coupling.base_strength,
        decay_alpha=spec.coupling.decay_alpha,
    )

    engine = UPDEEngine(n_oscillators=n_osc, dt=spec.sample_period_s)
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, TWO_PI, size=n_osc)
    omegas = np.ones(n_osc)
    phases = engine.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, n_steps=50)
    assert phases.shape == (n_osc,)
    assert np.all(np.isfinite(phases))


def test_queuewaves_domainpack() -> None:
    """Load queuewaves binding spec, verify correct parse and coupling build."""
    spec = load_binding_spec(DOMAINPACKS / "queuewaves" / "binding_spec.yaml")
    assert spec.name == "queuewaves"
    assert spec.safety_tier in ("research", "clinical", "industrial", "production")

    n_osc = sum(len(lay.oscillator_ids) for lay in spec.layers)
    assert n_osc >= 1

    builder = CouplingBuilder()
    cs = builder.build(
        n_layers=max(n_osc, 2),
        base_strength=spec.coupling.base_strength,
        decay_alpha=spec.coupling.decay_alpha,
    )
    assert cs.knm.shape[0] == cs.knm.shape[1]
    np.testing.assert_allclose(cs.knm, cs.knm.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(cs.knm), 0.0, atol=1e-15)


def test_bio_stub_domainpack() -> None:
    """Load bio_stub binding spec, verify parse."""
    spec = load_binding_spec(DOMAINPACKS / "bio_stub" / "binding_spec.yaml")
    assert spec.name == "bio_stub"
    assert len(spec.layers) >= 1
