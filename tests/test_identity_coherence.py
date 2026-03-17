# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Identity coherence domainpack tests

"""Verify the identity_coherence domainpack dynamics.

Tests cover: Kuramoto synchronization, disruption resilience,
imprint accumulation, Stuart-Landau conviction strength, and
layer-aware coupling structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.geometry_constraints import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
)
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel
from scpn_phase_orchestrator.monitor.boundaries import BoundaryObserver
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

SPEC_PATH = (
    Path(__file__).resolve().parent.parent
    / "domainpacks"
    / "identity_coherence"
    / "binding_spec.yaml"
)
TWO_PI = 2.0 * np.pi


@pytest.fixture
def spec():
    return load_binding_spec(SPEC_PATH)


@pytest.fixture
def layer_map(spec):
    osc_idx = 0
    ranges = {}
    for layer in spec.layers:
        n = len(layer.oscillator_ids)
        ranges[layer.index] = list(range(osc_idx, osc_idx + n))
        osc_idx += n
    return ranges


@pytest.fixture
def n_osc(spec):
    return sum(len(layer.oscillator_ids) for layer in spec.layers)


@pytest.fixture
def omegas(n_osc):
    """Tightly clustered natural frequencies for synchronization."""
    rng = np.random.default_rng(0)
    return 1.0 + rng.uniform(-0.02, 0.02, n_osc)


@pytest.fixture
def knm(n_osc, layer_map):
    """Layer-aware coupling matrix."""
    k = np.zeros((n_osc, n_osc))
    k_intra = 0.5
    k_cross = 0.15
    for ids in layer_map.values():
        for i in ids:
            for j in ids:
                if i != j:
                    k[i, j] = k_intra
    for li in layer_map:
        for lj in layer_map:
            if li < lj:
                for i in layer_map[li]:
                    for j in layer_map[lj]:
                        k[i, j] = k_cross
                        k[j, i] = k_cross
    return k


def test_spec_valid(spec):
    errors = validate_binding_spec(spec)
    assert errors == []


def test_spec_has_6_layers(spec):
    assert len(spec.layers) == 6


def test_spec_35_oscillators(spec):
    total = sum(len(layer.oscillator_ids) for layer in spec.layers)
    assert total == 35


def test_reconstruction_from_chaos(n_osc, omegas, knm):
    """Random phases synchronize under coupling (R < 0.5 -> R > 0.9)."""
    engine = UPDEEngine(n_osc, dt=0.01)
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n_osc)
    alpha = np.zeros((n_osc, n_osc))

    r_initial, _ = compute_order_parameter(phases)
    assert r_initial < 0.5

    for _ in range(500):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    r_final, _ = compute_order_parameter(phases)
    assert r_final > 0.9


def test_disruption_degrades_domain_knowledge(
    n_osc,
    omegas,
    knm,
    layer_map,
):
    """Noise injection into domain_knowledge lowers that layer's R."""
    engine = UPDEEngine(n_osc, dt=0.01)
    alpha = np.zeros((n_osc, n_osc))
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n_osc)

    # Synchronize first
    for _ in range(500):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    r_synced, _ = compute_order_parameter(phases[layer_map[4]])
    assert r_synced > 0.9

    # Disrupt domain_knowledge (layer 4) for 200 steps
    for _ in range(200):
        phases[layer_map[4]] += rng.normal(0, 1.0, len(layer_map[4]))
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    r_disrupted, _ = compute_order_parameter(phases[layer_map[4]])
    assert r_disrupted < r_synced


def test_core_identity_resilient_under_domain_disruption(
    n_osc,
    omegas,
    knm,
    layer_map,
):
    """Core layers (0-3) stay coherent when domain knowledge (4) is disrupted."""
    engine = UPDEEngine(n_osc, dt=0.01)
    alpha = np.zeros((n_osc, n_osc))
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n_osc)

    for _ in range(500):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    # Disrupt domain_knowledge only
    for _ in range(200):
        phases[layer_map[4]] += rng.normal(0, 1.0, len(layer_map[4]))
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    # Core layers should remain coherent
    core_ids = []
    for idx in [0, 1, 2, 3]:
        core_ids.extend(layer_map[idx])
    r_core, _ = compute_order_parameter(phases[core_ids])
    assert r_core > 0.7


def test_recovery_after_disruption(n_osc, omegas, knm, layer_map):
    """Domain knowledge re-synchronizes after noise removal."""
    engine = UPDEEngine(n_osc, dt=0.01)
    alpha = np.zeros((n_osc, n_osc))
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, TWO_PI, n_osc)

    # Synchronize
    for _ in range(500):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    # Disrupt
    for _ in range(200):
        phases[layer_map[4]] += rng.normal(0, 1.0, len(layer_map[4]))
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    # Recover (no noise)
    for _ in range(300):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    r_recovered, _ = compute_order_parameter(phases[layer_map[4]])
    assert r_recovered > 0.85


def test_imprint_accumulation(spec, n_osc):
    """ImprintModel accumulates exposure and saturates."""
    model = ImprintModel(
        spec.imprint_model.decay_rate,
        spec.imprint_model.saturation,
    )
    state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    exposure = np.ones(n_osc) * 0.8

    for _ in range(1000):
        state = model.update(state, exposure, 0.01)

    assert np.all(state.m_k > 0.0)
    assert np.all(state.m_k <= spec.imprint_model.saturation)


def test_imprint_modulates_coupling(spec, n_osc, knm):
    """Coupling strength increases with imprint."""
    model = ImprintModel(
        spec.imprint_model.decay_rate,
        spec.imprint_model.saturation,
    )
    zero_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    saturated_state = ImprintState(
        m_k=np.full(n_osc, spec.imprint_model.saturation),
        last_update=0.0,
    )

    k_zero = model.modulate_coupling(knm, zero_state)
    k_sat = model.modulate_coupling(knm, saturated_state)

    # Saturated imprint should produce stronger coupling
    assert np.all(k_sat >= k_zero)
    assert np.sum(k_sat) > np.sum(k_zero)


def test_knm_symmetric_non_negative(knm):
    """Coupling matrix satisfies geometry constraints."""
    constraints = [SymmetryConstraint(), NonNegativeConstraint()]
    projected = project_knm(knm, constraints)
    assert np.allclose(projected, projected.T)
    assert np.all(projected >= 0)


def test_knm_layer_structure(knm, layer_map):
    """Within-layer coupling is stronger than cross-layer."""
    intra_values = []
    cross_values = []
    for ids in layer_map.values():
        for i in ids:
            for j in ids:
                if i != j:
                    intra_values.append(knm[i, j])
    for li in layer_map:
        for lj in layer_map:
            if li < lj:
                for i in layer_map[li]:
                    for j in layer_map[lj]:
                        cross_values.append(knm[i, j])

    assert np.mean(intra_values) > np.mean(cross_values)


def test_stuart_landau_conviction_strength(n_osc, omegas, knm):
    """Supercritical oscillators (mu > 0) sustain nonzero amplitude."""
    sl_engine = StuartLandauEngine(n_osc, dt=0.01)
    rng = np.random.default_rng(42)

    mu = np.ones(n_osc)  # supercritical
    epsilon = 0.3
    knm_r = knm * 0.2
    alpha = np.zeros((n_osc, n_osc))

    phases = rng.uniform(0, TWO_PI, n_osc)
    amplitudes = rng.uniform(0.5, 1.5, n_osc)
    state = np.concatenate([phases, amplitudes])

    for _ in range(500):
        state = sl_engine.step(
            state,
            omegas,
            mu,
            knm,
            knm_r,
            0.0,
            0.0,
            alpha,
            epsilon,
        )

    mean_amp = sl_engine.compute_mean_amplitude(state)
    assert mean_amp > 0.5


def test_stuart_landau_subcritical_fade(n_osc, omegas, knm):
    """Subcritical oscillators (mu < 0) decay toward zero amplitude."""
    sl_engine = StuartLandauEngine(n_osc, dt=0.01)
    rng = np.random.default_rng(42)

    mu = np.full(n_osc, -1.0)  # subcritical
    epsilon = 0.3
    knm_r = knm * 0.2
    alpha = np.zeros((n_osc, n_osc))

    phases = rng.uniform(0, TWO_PI, n_osc)
    amplitudes = np.ones(n_osc)
    state = np.concatenate([phases, amplitudes])

    for _ in range(500):
        state = sl_engine.step(
            state,
            omegas,
            mu,
            knm,
            knm_r,
            0.0,
            0.0,
            alpha,
            epsilon,
        )

    mean_amp = sl_engine.compute_mean_amplitude(state)
    assert mean_amp < 0.3


def test_boundary_identity_collapse_fires(spec, n_osc):
    """R_good below 0.1 triggers hard boundary violation."""
    observer = BoundaryObserver(spec.boundaries)
    obs = {"R_good": 0.05, "layer_2_R": 0.5, "layer_3_R": 0.5}
    state = observer.observe(obs)
    assert state.hard_violations


def test_regime_manager_detects_degradation():
    """Low R triggers DEGRADED/CRITICAL regime."""
    rm = RegimeManager(hysteresis=0.05, cooldown_steps=1)
    low_r_state = UPDEState(
        layers=[LayerState(R=0.2, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=0.2,
        regime_id="nominal",
    )
    from scpn_phase_orchestrator.monitor.boundaries import BoundaryState

    bstate = BoundaryState(hard_violations=[], soft_violations=[])
    proposed = rm.evaluate(low_r_state, bstate)
    assert proposed == Regime.CRITICAL
