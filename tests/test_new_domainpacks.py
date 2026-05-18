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

from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.runtime.server import SimulationState
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
    """Every pack must declare which layers are good vs bad."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    assert spec.objectives is not None
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


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_new_domainpack_channel_objective_actuator_invariants(pack: str) -> None:
    """Hard-check schema invariants that must hold for stable runtime contracts."""
    spec = load_binding_spec(DOMAINPACK_DIR / pack / "binding_spec.yaml")
    assert validate_binding_spec(spec) == []

    valid_layer_indices = {layer.index for layer in spec.layers}
    valid_scopes = {"global"} | {f"layer_{idx}" for idx in valid_layer_indices}
    assert not (set(spec.objectives.good_layers) & set(spec.objectives.bad_layers))
    for idx in spec.objectives.good_layers:
        assert idx in valid_layer_indices
    for idx in spec.objectives.bad_layers:
        assert idx in valid_layer_indices
    assert spec.objectives.good_layers or spec.objectives.bad_layers

    for action in spec.actuators:
        assert action.scope in valid_scopes
        lo, hi = action.limits
        assert lo < hi

    for channel in spec.channels.values():
        if channel.derived_from:
            assert channel.replay_semantics == "derived"
            assert channel.derive_rule


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


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_new_domainpack_policy_schema_and_presence(pack: str) -> None:
    """Every new pack must carry a policy file that parses to policy rules."""
    policy_path = DOMAINPACK_DIR / pack / "policy.yaml"
    assert policy_path.exists()
    rules = load_policy_rules(policy_path)
    for rule in rules:
        assert rule.name
        assert rule.regimes
        assert rule.actions
        assert rule.condition is not None


@pytest.mark.parametrize("pack", NEW_PACKS)
def test_new_domainpack_bundle_integrity(pack: str) -> None:
    """Guard against domainpack drift for newly introduced pack contents."""
    pack_dir = DOMAINPACK_DIR / pack
    assert (pack_dir / "binding_spec.yaml").exists()
    assert (pack_dir / "run.py").exists()
    assert (pack_dir / "policy.yaml").exists()


def test_new_pack_count_matches_parametrisation() -> None:
    """Guard against silent mismatch between NEW_PACKS and checked folders."""
    declared = set(NEW_PACKS)
    existing = {
        p.name
        for p in DOMAINPACK_DIR.iterdir()
        if p.is_dir() and (p / "binding_spec.yaml").exists()
    }
    missing = declared - existing
    assert not missing, f"declared but missing packs: {sorted(missing)}"


def test_new_domainpack_simulation_reset_is_deterministic() -> None:
    """Reset after stepping returns a deterministic initial snapshot."""
    spec = load_binding_spec(DOMAINPACK_DIR / "financial_markets" / "binding_spec.yaml")
    sim = SimulationState(spec)
    baseline = sim.snapshot()

    for _ in range(4):
        sim.step()
    sim.reset()
    restored = sim.snapshot()

    assert restored["step"] == 0
    assert restored["R_global"] == baseline["R_global"]
    assert restored["layers"] == baseline["layers"]


def test_new_domainpack_step_and_observation_bounds() -> None:
    """Ensure bounded progress and observable invariants at each step."""
    spec = load_binding_spec(DOMAINPACK_DIR / "gene_oscillator" / "binding_spec.yaml")
    sim = SimulationState(spec)
    state = sim.snapshot()
    assert state["step"] == 0

    for expected_step in range(1, 4):
        state = sim.step()
        assert state["step"] == expected_step
        assert 0.0 <= state["R_global"] <= 1.0

    reset_state = sim.reset()
    assert reset_state["step"] == 0
    assert 0.0 <= reset_state["R_global"] <= 1.0


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("this is not yaml: [", "YAML parse error"),
        (
            """
version: 0.1.0
name: missing-layers
safety_tier: production
sample_period_s: 0.1
control_period_s: 0.2
oscillator_families: {}
coupling:
  base_strength: 0.1
  decay_alpha: 0.1
drivers:
  physical: {}
objectives:
  good_layers: [0]
  bad_layers: []
""",
            "missing required key 'layers'",
        ),
    ],
)
def test_new_domainpack_invalid_binding_spec_rejects_payload(
    tmp_path: Path,
    payload: str,
    match: str,
) -> None:
    path = tmp_path / "binding_spec.yaml"
    path.write_text(payload.strip() + "\n", encoding="utf-8")
    with pytest.raises(BindingLoadError, match=match):
        load_binding_spec(path)


def test_new_domainpack_invalid_policy_schema_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        """
rules:
  - name: missing-action
    regime: [NOMINAL]
    condition:
      metric: R
      layer: 0
      op: '>'
      threshold: 0.1
""",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError, match="invalid policy rules: rule missing required key 'action'"
    ):
        load_policy_rules(path)


def test_new_domainpack_invalid_policy_parse_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text("rules: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="policy rules YAML parse error"):
        load_policy_rules(path)


def test_missing_domainpack_raises_clear_error() -> None:
    """Loading a non-existent pack raises a scrubbed BindingLoadError."""
    with pytest.raises(BindingLoadError) as exc_info:
        load_binding_spec(DOMAINPACK_DIR / "does_not_exist" / "binding_spec.yaml")
    msg = str(exc_info.value)
    assert "binding_spec.yaml" in msg
    assert str(DOMAINPACK_DIR) not in msg
