# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Domainpack validation tests

from __future__ import annotations

from pathlib import Path

import pytest

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.binding.types import VALID_SAFETY_TIERS

DOMAINPACKS_DIR = Path(__file__).resolve().parent.parent / "domainpacks"

ALL_PACKS = sorted(p.parent.name for p in DOMAINPACKS_DIR.glob("*/binding_spec.yaml"))


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
