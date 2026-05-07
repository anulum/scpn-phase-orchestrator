# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Domainpack validation tests

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.binding.types import VALID_SAFETY_TIERS
from scpn_phase_orchestrator.supervisor import (
    ValueAlignmentGuard,
    value_alignment_policy_from_binding_spec,
)

DOMAINPACKS_DIR = Path(__file__).resolve().parent.parent / "domainpacks"

ALL_PACKS = sorted(p.parent.name for p in DOMAINPACKS_DIR.glob("*/binding_spec.yaml"))
NCHANNEL_EXAMPLES = {
    "agent_coordination",
    "swarm_robotics",
    "traffic_flow",
}


@pytest.fixture(params=ALL_PACKS)
def pack_name(request):
    return request.param


@pytest.fixture
def spec(pack_name):
    return load_binding_spec(DOMAINPACKS_DIR / pack_name / "binding_spec.yaml")


def test_spec_loads(spec):
    assert spec.name


def test_spec_validates(spec):
    errors = validate_binding_spec(spec)
    assert errors == [], f"{spec.name}: {errors}"


def test_has_layers(spec):
    assert len(spec.layers) >= 1


def test_layer_indices_contiguous(spec):
    indices = [layer.index for layer in spec.layers]
    assert indices == list(range(len(spec.layers)))


def test_each_layer_has_oscillators(spec):
    for layer in spec.layers:
        assert len(layer.oscillator_ids) >= 1, f"layer {layer.name} has no oscillators"


def test_objectives_reference_valid_layers(spec):
    valid = {layer.index for layer in spec.layers}
    for idx in spec.objectives.good_layers:
        assert idx in valid, f"good_layer {idx} not in layers"
    for idx in spec.objectives.bad_layers:
        assert idx in valid, f"bad_layer {idx} not in layers"


def test_good_bad_disjoint(spec):
    overlap = set(spec.objectives.good_layers) & set(spec.objectives.bad_layers)
    assert not overlap, f"overlap in good/bad: {overlap}"


def test_safety_tier_valid(spec):
    assert spec.safety_tier in VALID_SAFETY_TIERS


def test_boundaries_have_at_least_one_limit(spec):
    for b in spec.boundaries:
        assert b.lower is not None or b.upper is not None, (
            f"boundary {b.name} has no limit"
        )


def test_actuators_have_valid_limits(spec):
    for a in spec.actuators:
        lo, hi = a.limits
        assert lo < hi, f"actuator {a.name}: lo={lo} >= hi={hi}"


def test_policy_file_exists(pack_name):
    policy_path = DOMAINPACKS_DIR / pack_name / "policy.yaml"
    spec_text = (DOMAINPACKS_DIR / pack_name / "binding_spec.yaml").read_text(
        encoding="utf-8"
    )
    if spec_text.find("policy:") != -1:
        assert policy_path.exists(), f"{pack_name}/policy.yaml missing"


def test_run_file_exists(pack_name):
    run_path = DOMAINPACKS_DIR / pack_name / "run.py"
    assert run_path.exists(), f"{pack_name}/run.py missing"


@pytest.mark.parametrize("pack_name", sorted(NCHANNEL_EXAMPLES))
def test_nchannel_examples_are_declared_and_wired(pack_name):
    spec = load_binding_spec(DOMAINPACKS_DIR / pack_name / "binding_spec.yaml")

    assert len(spec.used_channels()) > 3
    assert spec.channels
    assert spec.channel_groups
    assert spec.cross_channel_couplings
    assert any(channel.derived_from for channel in spec.channels.values())
    assert validate_binding_spec(spec) == []


def test_cardiac_value_alignment_template_blocks_excessive_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "cardiac_rhythm" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    guard = ValueAlignmentGuard(policy)
    unsafe = ControlAction(
        knob="K",
        scope="global",
        value=1.2,
        ttl_s=5.0,
        justification="review candidate exceeds cardiac coupling prior",
    )

    decision = guard.evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-drug-coupling"
    assert decision.actions_to_apply[0].knob == "zeta"
    assert decision.actions_to_apply[0].value == 0.0
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_cardiac_value_alignment_template_allows_bounded_review_action():
    spec = load_binding_spec(DOMAINPACKS_DIR / "cardiac_rhythm" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    decision = ValueAlignmentGuard(policy).evaluate(
        [
            ControlAction(
                knob="K",
                scope="global",
                value=0.2,
                ttl_s=5.0,
                justification="bounded review candidate",
            )
        ]
    )

    assert decision.satisfied
    assert not decision.violations
    assert decision.actions_to_apply == decision.approved_actions


def test_power_grid_value_alignment_blocks_excessive_load_shedding():
    spec = load_binding_spec(DOMAINPACKS_DIR / "power_grid" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="alpha",
        scope="layer_3",
        value=0.9,
        ttl_s=5.0,
        justification="review candidate exceeds load-shed prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-load-shed-step"
    assert decision.actions_to_apply[0].justification == "grid value guard safe hold"
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_power_grid_value_alignment_allows_bounded_governor_action():
    spec = load_binding_spec(DOMAINPACKS_DIR / "power_grid" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.4,
        ttl_s=5.0,
        justification="bounded governor droop review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_autonomous_vehicle_value_alignment_blocks_excessive_throttle():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "autonomous_vehicles" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.9,
        ttl_s=1.0,
        justification="review candidate exceeds throttle prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-throttle-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "autonomous vehicle value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_autonomous_vehicle_value_alignment_allows_bounded_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "autonomous_vehicles" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.4,
        ttl_s=1.0,
        justification="bounded platoon coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_satellite_value_alignment_blocks_excessive_beam_steering():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "satellite_constellation" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.8,
        ttl_s=1.0,
        justification="review candidate exceeds beam steering prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-beam-steering-step"
    assert decision.actions_to_apply[0].justification == (
        "satellite value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_satellite_value_alignment_allows_bounded_pll_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "satellite_constellation" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.35,
        ttl_s=1.0,
        justification="bounded PLL coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_power_safety_value_alignment_blocks_excessive_substation_lag():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "power_safety_nchannel" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="alpha",
        scope="layer_1",
        value=0.7,
        ttl_s=1.0,
        justification="review candidate exceeds substation lag prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-substation-lag-step"
    assert decision.actions_to_apply[0].justification == (
        "power safety value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_power_safety_value_alignment_allows_bounded_grid_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "power_safety_nchannel" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.3,
        ttl_s=1.0,
        justification="bounded N-channel grid coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_network_security_value_alignment_blocks_excessive_defense_drive():
    spec = load_binding_spec(DOMAINPACKS_DIR / "network_security" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=1.0,
        ttl_s=1.0,
        justification="review candidate exceeds defense drive prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-defense-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "network security value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_network_security_value_alignment_allows_bounded_firewall_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "network_security" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.4,
        ttl_s=1.0,
        justification="bounded firewall coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_financial_value_alignment_blocks_excessive_rebalance_lag():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "financial_markets" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="alpha",
        scope="layer_0",
        value=0.6,
        ttl_s=10.0,
        justification="review candidate exceeds rebalance lag prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-equity-rebalance-lag-step"
    assert (
        decision.actions_to_apply[0].justification == "financial value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_financial_value_alignment_allows_bounded_cross_asset_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "financial_markets" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.3,
        ttl_s=10.0,
        justification="bounded cross-asset coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_chemical_reactor_value_alignment_blocks_excessive_feed_rate():
    spec = load_binding_spec(DOMAINPACKS_DIR / "chemical_reactor" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.9,
        ttl_s=1.0,
        justification="review candidate exceeds feed-rate prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-feed-rate-step"
    assert decision.actions_to_apply[0].justification == (
        "chemical reactor value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_chemical_reactor_value_alignment_allows_bounded_coolant_flow():
    spec = load_binding_spec(DOMAINPACKS_DIR / "chemical_reactor" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.4,
        ttl_s=1.0,
        justification="bounded coolant-flow review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_manufacturing_value_alignment_blocks_excessive_sensor_lag():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "manufacturing_spc" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="alpha",
        scope="layer_0",
        value=0.7,
        ttl_s=1.0,
        justification="review candidate exceeds sensor lag prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-sensor-lag-step"
    assert decision.actions_to_apply[0].justification == (
        "manufacturing value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_manufacturing_value_alignment_allows_bounded_station_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "manufacturing_spc" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.3,
        ttl_s=1.0,
        justification="bounded station coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_robotic_cpg_value_alignment_blocks_excessive_stride_frequency():
    spec = load_binding_spec(DOMAINPACKS_DIR / "robotic_cpg" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=1.1,
        ttl_s=1.0,
        justification="review candidate exceeds stride-frequency prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-stride-frequency-step"
    assert decision.actions_to_apply[0].justification == (
        "robotic CPG value guard safe hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_robotic_cpg_value_alignment_allows_bounded_gait_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "robotic_cpg" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.3,
        ttl_s=1.0,
        justification="bounded gait coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_swarm_robotics_value_alignment_blocks_excessive_obstacle_avoidance():
    spec = load_binding_spec(DOMAINPACKS_DIR / "swarm_robotics" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="alpha",
        scope="layer_0",
        value=1.45,
        ttl_s=1.0,
        justification="review candidate exceeds obstacle-avoidance prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-obstacle-avoidance-step"
    assert decision.actions_to_apply[0].justification == (
        "swarm robotics value guard formation hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_swarm_robotics_value_alignment_allows_bounded_alignment_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "swarm_robotics" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.6,
        ttl_s=1.0,
        justification="bounded alignment coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_traffic_flow_value_alignment_blocks_excessive_signal_split():
    spec = load_binding_spec(DOMAINPACKS_DIR / "traffic_flow" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="alpha",
        scope="layer_0",
        value=1.3,
        ttl_s=1.0,
        justification="review candidate exceeds signal-split prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-signal-split-step"
    assert decision.actions_to_apply[0].justification == (
        "traffic flow value guard offset hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_traffic_flow_value_alignment_allows_bounded_cycle_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "traffic_flow" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.7,
        ttl_s=1.0,
        justification="bounded cycle-coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_plasma_control_value_alignment_blocks_excessive_damping_drive():
    spec = load_binding_spec(DOMAINPACKS_DIR / "plasma_control" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=1.1,
        ttl_s=0.5,
        justification="review candidate exceeds plasma damping prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-feedback-damping-step"
    assert decision.actions_to_apply[0].justification == (
        "plasma value guard damping hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_plasma_control_value_alignment_allows_bounded_transport_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "plasma_control" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.6,
        ttl_s=0.5,
        justification="bounded transport-coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_fusion_equilibrium_value_alignment_blocks_excessive_auxiliary_drive():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "fusion_equilibrium" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.95,
        ttl_s=0.5,
        justification="review candidate exceeds auxiliary-drive prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-auxiliary-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "fusion equilibrium value guard drive hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_fusion_equilibrium_value_alignment_allows_bounded_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "fusion_equilibrium" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.55,
        ttl_s=0.5,
        justification="bounded equilibrium-coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_neuroscience_eeg_value_alignment_blocks_excessive_entrainment():
    spec = load_binding_spec(DOMAINPACKS_DIR / "neuroscience_eeg" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.80,
        ttl_s=0.5,
        justification="review candidate exceeds entrainment-drive prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-entrainment-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "neuroscience EEG value guard stimulus hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_neuroscience_eeg_value_alignment_allows_bounded_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "neuroscience_eeg" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.45,
        ttl_s=0.5,
        justification="bounded EEG coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_brain_connectome_value_alignment_blocks_excessive_neuromodulation():
    spec = load_binding_spec(DOMAINPACKS_DIR / "brain_connectome" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.75,
        ttl_s=0.5,
        justification="review candidate exceeds neuromodulation-drive prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-neuromodulation-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "brain connectome value guard neuromodulation hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_brain_connectome_value_alignment_allows_bounded_coupling():
    spec = load_binding_spec(DOMAINPACKS_DIR / "brain_connectome" / "binding_spec.yaml")
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.42,
        ttl_s=0.5,
        justification="bounded connectome coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_sleep_architecture_value_alignment_blocks_excessive_circadian_drive():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "sleep_architecture" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.70,
        ttl_s=0.5,
        justification="review candidate exceeds circadian-drive prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-circadian-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "sleep architecture value guard circadian hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_sleep_architecture_value_alignment_allows_bounded_coupling():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "sleep_architecture" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.35,
        ttl_s=0.5,
        justification="bounded sleep coupling review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


def test_circadian_biology_value_alignment_blocks_excessive_zeitgeber_drive():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "circadian_biology" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    unsafe = ControlAction(
        knob="zeta",
        scope="global",
        value=0.90,
        ttl_s=0.5,
        justification="review candidate exceeds zeitgeber-drive prior",
    )
    decision = ValueAlignmentGuard(policy).evaluate([unsafe])

    assert not decision.satisfied
    assert decision.blocked_actions == (unsafe,)
    assert decision.violations[0].constraint == "limit-zeitgeber-drive-step"
    assert decision.actions_to_apply[0].justification == (
        "circadian value guard zeitgeber hold"
    )
    assert decision.to_audit_record()["violations"][0]["counterfactual"] == (
        "blocked_action_prevents_constraint_violation"
    )


def test_circadian_biology_value_alignment_allows_bounded_behavioral_lag():
    spec = load_binding_spec(
        DOMAINPACKS_DIR / "circadian_biology" / "binding_spec.yaml"
    )
    policy = value_alignment_policy_from_binding_spec(spec)

    assert policy is not None
    action = ControlAction(
        knob="alpha",
        scope="layer_3",
        value=0.45,
        ttl_s=0.5,
        justification="bounded behavioral lag review candidate",
    )
    decision = ValueAlignmentGuard(policy).evaluate([action])

    assert decision.satisfied
    assert decision.approved_actions == (action,)
    assert not decision.violations


# Pipeline wiring: domainpack validation tested via real domainpack loading and
# schema enforcement. TestDomainpackLoading (above) proves domainpacks are functional.
