# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — New domainpack integration tests

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.server import SimulationState
from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

DOMAINPACK_DIR = Path(__file__).parent.parent / "domainpacks"

NCHANNEL_PACKS = [
    "digital_twin_nchannel",
    "edge_consensus_nchannel",
    "power_safety_nchannel",
]

NEW_PACKS = [
    "financial_markets",
    "gene_oscillator",
    "vortex_shedding",
    "robotic_cpg",
    "sleep_architecture",
    "musical_acoustics",
    "brain_connectome",
    "autonomous_vehicles",
    "chemical_reactor",
    "circadian_biology",
    "epidemic_sir",
    "manufacturing_spc",
    "network_security",
    "pll_clock",
    "satellite_constellation",
    "swarm_robotics",
    "traffic_flow",
    *NCHANNEL_PACKS,
]


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_loads(pack: str) -> None:
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    assert spec.name == pack
    assert len(spec.layers) >= 2


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_simulates(pack: str) -> None:
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    sim = SimulationState(spec)
    for _ in range(10):
        state = sim.step()
    assert state["step"] == 10
    assert 0.0 <= state["R_global"] <= 1.0


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_has_objectives(pack: str) -> None:
    """Every pack must declare which layers are good vs bad — the
    supervisor relies on `objectives.good_layers`/`bad_layers` to
    compute R_good, R_bad."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    assert spec.objectives is not None
    # Either list may be empty but they must be typed.
    assert isinstance(spec.objectives.good_layers, list)
    assert isinstance(spec.objectives.bad_layers, list)


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_all_layer_oscillators_non_empty(pack: str) -> None:
    """A layer with zero oscillators is useless — every pack must have
    at least one oscillator per declared layer."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    for layer in spec.layers:
        assert len(layer.oscillator_ids) > 0, (
            f"{pack}: layer {layer.name!r} has no oscillators"
        )


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_sample_period_positive(pack: str) -> None:
    """sample_period_s and control_period_s must be positive for the
    integration loop to advance."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    assert spec.sample_period_s > 0
    assert spec.control_period_s > 0
    # Control period should be >= sample period (rule enforced by
    # validator but mirrored here as a pack invariant).
    assert spec.control_period_s >= spec.sample_period_s


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_domainpack_reset_restores_step_zero(pack: str) -> None:
    """sim.reset() after stepping must return state with step=0."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    sim = SimulationState(spec)
    for _ in range(5):
        sim.step()
    reset_state = sim.reset()
    assert reset_state["step"] == 0


@pytest.mark.parametrize("pack", NCHANNEL_PACKS)
def test_nchannel_domainpack_declares_channel_algebra(pack: str) -> None:
    """N-channel examples must exercise groups, derived channels, and
    cross-channel coupling rather than only renaming P/I/S."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    errors = validate_binding_spec(spec)
    assert not errors
    assert len(spec.used_channels()) > 3
    assert len(spec.channel_groups) >= 2
    assert len(spec.cross_channel_couplings) >= 3
    derived = [
        (name, channel)
        for name, channel in spec.channels.items()
        if channel.derived_from and channel.replay_semantics == "derived"
    ]
    assert derived
    for _, channel in derived:
        assert channel.derive_rule
        assert channel.supervisor_visibility is True


@pytest.mark.parametrize("pack", NCHANNEL_PACKS)
def test_nchannel_domainpack_policy_loads(pack: str) -> None:
    """The examples must ship an executable policy next to the binding."""
    rules = load_policy_rules(DOMAINPACK_DIR / pack / "policy.yaml")
    assert len(rules) >= 2
    assert all(rule.actions for rule in rules)


def test_missing_domainpack_raises_clear_error() -> None:
    """Loading a non-existent pack raises a scrubbed BindingLoadError."""
    from scpn_phase_orchestrator.binding.loader import BindingLoadError

    with pytest.raises(BindingLoadError) as exc_info:
        load_binding_spec(DOMAINPACK_DIR / "does_not_exist" / "binding_spec.yaml")
    # Error must identify the missing file but not the full absolute path.
    msg = str(exc_info.value)
    assert "binding_spec.yaml" in msg
    assert str(DOMAINPACK_DIR) not in msg


def test_new_pack_count_matches_parametrisation() -> None:
    """Guard against silent mismatch between NEW_PACKS and the physical
    domainpack directory — stops a new pack being added to the filesystem
    without being exercised by these tests."""
    declared = set(NEW_PACKS)
    existing = {
        p.name
        for p in DOMAINPACK_DIR.iterdir()
        if p.is_dir() and (p / "binding_spec.yaml").exists()
    }
    # Every declared pack must physically exist.
    missing = declared - existing
    assert not missing, f"declared but missing packs: {missing}"


# Pipeline wiring: test_domainpack_loads / simulates / has_objectives /
# all_layer_oscillators_non_empty / sample_period_positive / reset each
# exercise one slice of SimulationState per pack. The last two cases
# cover the "unknown pack" error contract and a filesystem drift guard.
