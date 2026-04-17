# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Identity coherence domainpack tests

"""Verify the identity_coherence domainpack dynamics.

Tests cover: Kuramoto synchronization, disruption resilience,
imprint accumulation, Stuart-Landau conviction strength,
layer-aware coupling structure, policy rules, soft boundaries,
modulate_lag, degenerate phases, and end-to-end smoke test.
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


# --- P0: PolicyEngine rules fire correctly ---


def test_policy_rules_fire_on_degraded_state(spec):
    """PolicyEngine rules fire when stability_proxy is low."""
    from scpn_phase_orchestrator.supervisor.policy_rules import (
        PolicyEngine,
        load_policy_rules,
    )

    policy_path = SPEC_PATH.parent / "policy.yaml"
    rules = load_policy_rules(policy_path)
    engine = PolicyEngine(rules)

    degraded_state = UPDEState(
        layers=[LayerState(R=0.3, psi=0.0) for _ in spec.layers],
        cross_layer_alignment=np.zeros((6, 6)),
        stability_proxy=0.3,
        regime_id="critical",
    )
    actions = engine.evaluate(
        Regime.CRITICAL,
        degraded_state,
        spec.objectives.good_layers,
        spec.objectives.bad_layers,
    )
    assert len(actions) > 0
    knobs = {a.knob for a in actions}
    assert "K" in knobs


def test_policy_nominal_rule_fires(spec):
    """identity_nominal rule fires in NOMINAL with high stability_proxy."""
    from scpn_phase_orchestrator.supervisor.policy_rules import (
        PolicyEngine,
        load_policy_rules,
    )

    policy_path = SPEC_PATH.parent / "policy.yaml"
    rules = load_policy_rules(policy_path)
    engine = PolicyEngine(rules)

    nominal_state = UPDEState(
        layers=[LayerState(R=0.9, psi=0.0) for _ in spec.layers],
        cross_layer_alignment=np.zeros((6, 6)),
        stability_proxy=0.9,
        regime_id="nominal",
    )
    actions = engine.evaluate(
        Regime.NOMINAL,
        nominal_state,
        spec.objectives.good_layers,
        spec.objectives.bad_layers,
    )
    names = {a.justification for a in actions}
    assert any("identity_nominal" in n for n in names)


# --- P0: End-to-end smoke test ---


def test_main_runs_end_to_end():
    """Smoke test: main() completes without error (reduced steps)."""
    import domainpacks.identity_coherence.run as id_run

    original = id_run.STEPS
    id_run.STEPS = 50
    try:
        id_run.main()
    finally:
        id_run.STEPS = original


def test_stuart_landau_runs_end_to_end():
    """Smoke test: run_stuart_landau() completes without error."""
    import domainpacks.identity_coherence.run as id_run

    original = id_run.STEPS
    id_run.STEPS = 50
    try:
        id_run.run_stuart_landau()
    finally:
        id_run.STEPS = original


# --- P1: Soft boundary violations ---


def test_soft_boundary_relationship_drift(spec):
    """layer_2_R below 0.3 triggers relationship_drift soft violation."""
    observer = BoundaryObserver(spec.boundaries)
    obs = {"R_good": 0.9, "layer_2_R": 0.25, "layer_3_R": 0.5}
    state = observer.observe(obs)
    assert state.soft_violations
    assert any("relationship_drift" in str(v) for v in state.soft_violations)


def test_soft_boundary_aesthetic_erosion(spec):
    """layer_3_R below 0.2 triggers aesthetic_erosion soft violation."""
    observer = BoundaryObserver(spec.boundaries)
    obs = {"R_good": 0.9, "layer_2_R": 0.5, "layer_3_R": 0.15}
    state = observer.observe(obs)
    assert state.soft_violations
    assert any("aesthetic_erosion" in str(v) for v in state.soft_violations)


# --- P1: Stuart-Landau subcritical→recovery transition ---


def test_stuart_landau_disruption_recovery_transition(
    n_osc,
    omegas,
    knm,
    layer_map,
):
    """Domain knowledge amplitude decays subcritically, then recovers."""
    sl_engine = StuartLandauEngine(n_osc, dt=0.01)
    rng = np.random.default_rng(42)
    epsilon = 0.3
    knm_r = knm * 0.2
    alpha = np.zeros((n_osc, n_osc))
    dk_ids = layer_map[4]

    mu = np.ones(n_osc)
    phases = rng.uniform(0, TWO_PI, n_osc)
    amplitudes = np.ones(n_osc)
    state = np.concatenate([phases, amplitudes])

    # Supercritical: amplitudes sustain
    for _ in range(300):
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
    amp_sustained = float(np.mean(state[n_osc:][dk_ids]))
    assert amp_sustained > 0.8

    # Subcritical disruption: domain knowledge decays
    mu[dk_ids] = -0.5
    for _ in range(300):
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
    amp_decayed = float(np.mean(state[n_osc:][dk_ids]))
    assert amp_decayed < amp_sustained

    # Recovery: mu restored
    mu[dk_ids] = 1.0
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
    amp_recovered = float(np.mean(state[n_osc:][dk_ids]))
    assert amp_recovered > 0.8


# --- P1: modulate_lag produces non-zero offsets ---


def test_imprint_modulates_lag(spec, n_osc):
    """Saturated imprint produces non-zero phase lag offsets."""
    model = ImprintModel(
        spec.imprint_model.decay_rate,
        spec.imprint_model.saturation,
    )
    alpha_zero = np.zeros((n_osc, n_osc))
    sat = spec.imprint_model.saturation

    # Non-uniform imprint: first half saturated, second half zero
    m_k = np.zeros(n_osc)
    m_k[: n_osc // 2] = sat
    imprint = ImprintState(m_k=m_k, last_update=0.0)

    alpha_mod = model.modulate_lag(alpha_zero, imprint)
    assert not np.allclose(alpha_mod, 0.0)
    # Antisymmetric offset: alpha_mod[i,j] = -alpha_mod[j,i] when m_k differs
    assert np.allclose(alpha_mod, -alpha_mod.T)


# --- P1: modulate_mu scales bifurcation parameter ---


def test_imprint_modulates_mu(spec, n_osc):
    """Saturated imprint increases mu beyond base value."""
    model = ImprintModel(
        spec.imprint_model.decay_rate,
        spec.imprint_model.saturation,
    )
    mu_base = np.ones(n_osc)
    sat_state = ImprintState(
        m_k=np.full(n_osc, spec.imprint_model.saturation),
        last_update=0.0,
    )
    zero_state = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    mu_sat = model.modulate_mu(mu_base, sat_state)
    mu_zero = model.modulate_mu(mu_base, zero_state)

    assert np.all(mu_sat >= mu_zero)
    assert np.all(mu_sat > mu_base)


# --- P2: Degenerate phases (all identical) stability ---


def test_identical_phases_stay_synchronized(n_osc, omegas, knm):
    """All-identical phases remain at R=1.0 under coupling."""
    engine = UPDEEngine(n_osc, dt=0.01)
    phases = np.zeros(n_osc)
    alpha = np.zeros((n_osc, n_osc))

    for _ in range(100):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

    assert np.all(np.isfinite(phases))
    r, _ = compute_order_parameter(phases)
    assert r > 0.95


# --- P2: _build_identity_knm direct test ---


def test_build_identity_knm_properties(spec):
    """Production coupling matrix is symmetric, non-negative, zero diagonal."""
    from domainpacks.identity_coherence.run import (
        _build_identity_knm,
        _build_layer_map,
    )

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)
    knm_actual = _build_identity_knm(n_osc, layer_map)

    assert np.allclose(knm_actual, knm_actual.T)
    assert np.all(knm_actual >= 0)
    assert np.allclose(np.diag(knm_actual), 0.0)


# --- P/I/S extraction tests ---


def test_extract_identity_phases_shape(spec):
    """P/I/S extraction returns correct phase array shape."""
    from domainpacks.identity_coherence.run import (
        OMEGAS,
        _build_layer_map,
        extract_identity_phases,
    )

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)
    imprint = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    phases, all_states = extract_identity_phases(spec, layer_map, OMEGAS, imprint)

    assert phases.shape == (n_osc,)
    assert np.all(np.isfinite(phases))
    assert len(all_states) > 0


def test_extract_identity_phases_all_channels(spec):
    """Extraction produces states from all three channels."""
    from domainpacks.identity_coherence.run import (
        OMEGAS,
        _build_layer_map,
        extract_identity_phases,
    )

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)
    imprint = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    _, all_states = extract_identity_phases(spec, layer_map, OMEGAS, imprint)

    channels = {s.channel for s in all_states}
    assert "P" in channels
    assert "I" in channels
    assert "S" in channels


def test_imprint_affects_extraction_quality(spec):
    """High imprint produces higher P-channel quality (less noise)."""
    from domainpacks.identity_coherence.run import (
        OMEGAS,
        _build_layer_map,
        extract_identity_phases,
    )

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)

    fresh = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)
    _, states_fresh = extract_identity_phases(spec, layer_map, OMEGAS, fresh, seed=42)

    saturated = ImprintState(m_k=np.full(n_osc, 0.8), last_update=100.0)
    _, states_sat = extract_identity_phases(spec, layer_map, OMEGAS, saturated, seed=42)

    p_fresh = [s.quality for s in states_fresh if s.channel == "P"]
    p_sat = [s.quality for s in states_sat if s.channel == "P"]

    # Saturated imprint = less noise = higher quality
    assert np.mean(p_sat) > np.mean(p_fresh)


def test_session_start_check_passes_for_identity(spec):
    """Full session-start check passes for identity_coherence domainpack."""
    from domainpacks.identity_coherence.run import (
        OMEGAS,
        _build_layer_map,
        run_session_start_check,
    )

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)
    imprint = ImprintState(m_k=np.zeros(n_osc), last_update=0.0)

    phases, report = run_session_start_check(spec, layer_map, OMEGAS, imprint)
    assert report.passed
    assert phases.shape == (n_osc,)
    assert "P" in report.quality_scores


def test_session_start_check_with_imprint(spec):
    """Session-start check reports imprint level when loaded."""
    from domainpacks.identity_coherence.run import (
        OMEGAS,
        _build_layer_map,
        run_session_start_check,
    )

    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)
    layer_map = _build_layer_map(spec)
    imprint = ImprintState(m_k=np.full(n_osc, 0.5), last_update=50.0)

    _, report = run_session_start_check(spec, layer_map, OMEGAS, imprint)
    assert report.passed
    assert abs(report.imprint_level - 0.5) < 1e-6
