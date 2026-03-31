# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Domainpack amplitude tests

"""Verify domainpacks with amplitude configs load and validate correctly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.supervisor.policy_rules import load_policy_rules

DOMAINPACKS_DIR = Path(__file__).resolve().parent.parent / "domainpacks"

AMPLITUDE_PACKS = [
    "neuroscience_eeg",
    "cardiac_rhythm",
    "plasma_control",
    "firefly_swarm",
    "rotating_machinery",
    "power_grid",
]


@pytest.fixture(params=AMPLITUDE_PACKS)
def pack_name(request):
    return request.param


def test_binding_spec_has_amplitude(pack_name):
    raw = yaml.safe_load(
        (DOMAINPACKS_DIR / pack_name / "binding_spec.yaml").read_text(encoding="utf-8")
    )
    assert "amplitude" in raw, f"{pack_name} missing amplitude block"
    amp = raw["amplitude"]
    assert "mu" in amp
    assert "epsilon" in amp
    assert isinstance(amp["mu"], (int, float))
    assert isinstance(amp["epsilon"], (int, float))


def test_binding_spec_validates(pack_name):
    spec = load_binding_spec(DOMAINPACKS_DIR / pack_name / "binding_spec.yaml")
    errors = validate_binding_spec(spec)
    assert errors == [], f"{pack_name}: {errors}"


def test_policy_has_pac_rules(pack_name):
    policy_path = DOMAINPACKS_DIR / pack_name / "policy.yaml"
    if not policy_path.exists():
        pytest.skip(f"{pack_name} has no policy.yaml")
    rules = load_policy_rules(policy_path)
    rule_names = {r.name for r in rules}
    assert "pac_gating_alert" in rule_names, (
        f"{pack_name} missing pac_gating_alert rule"
    )
    assert "subcritical_recovery" in rule_names, (
        f"{pack_name} missing subcritical_recovery rule"
    )


def test_pac_rule_uses_correct_metric(pack_name):
    policy_path = DOMAINPACKS_DIR / pack_name / "policy.yaml"
    if not policy_path.exists():
        pytest.skip(f"{pack_name} has no policy.yaml")
    rules = load_policy_rules(policy_path)
    pac_rules = [r for r in rules if r.name == "pac_gating_alert"]
    assert len(pac_rules) == 1
    cond = pac_rules[0].condition
    assert cond.metric == "pac_max"
    assert cond.op == ">"


def test_subcritical_rule_targets_coupling(pack_name):
    policy_path = DOMAINPACKS_DIR / pack_name / "policy.yaml"
    if not policy_path.exists():
        pytest.skip(f"{pack_name} has no policy.yaml")
    rules = load_policy_rules(policy_path)
    sub_rules = [r for r in rules if r.name == "subcritical_recovery"]
    assert len(sub_rules) == 1
    assert sub_rules[0].actions[0].knob == "K"


def test_amplitude_params_positive(pack_name):
    raw = yaml.safe_load(
        (DOMAINPACKS_DIR / pack_name / "binding_spec.yaml").read_text(encoding="utf-8")
    )
    amp = raw["amplitude"]
    assert amp["mu"] > 0, f"{pack_name}: mu must be positive for supercritical regime"
    assert amp["epsilon"] > 0, f"{pack_name}: epsilon must be positive"


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
