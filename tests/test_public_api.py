# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public Python API tests

from __future__ import annotations

import numpy as np
import pytest
import yaml

from scpn import Orchestrator
from scpn_phase_orchestrator import Orchestrator as LongNameOrchestrator


def test_short_python_api_runs_binding_spec_deterministically(tmp_path):
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(_minimal_spec()), encoding="utf-8")

    orch = Orchestrator.from_yaml(spec_path)
    first = orch.run(steps=8, seed=7)
    second = orch.run(steps=8, seed=7)

    assert first.spec_name == "public-api-minimal"
    assert first.steps == 8
    assert first.phases.shape == (4,)
    assert first.knm.shape == (4, 4)
    assert first.alpha.shape == (4, 4)
    assert 0.0 <= first.order_parameter <= 1.0
    assert np.allclose(first.phases, second.phases)


def test_long_package_exports_same_orchestrator_surface():
    assert LongNameOrchestrator is Orchestrator


def test_python_api_rejects_non_research_execution(tmp_path):
    data = _minimal_spec()
    data["safety_tier"] = "pilot"
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="safety_tier"):
        Orchestrator.from_yaml(spec_path)


def test_python_api_rejects_invalid_step_count(tmp_path):
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(_minimal_spec()), encoding="utf-8")
    orch = Orchestrator.from_yaml(spec_path)

    with pytest.raises(ValueError, match="steps"):
        orch.run(steps=-1)


def _minimal_spec() -> dict[str, object]:
    return {
        "name": "public-api-minimal",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [
            {
                "name": "L0",
                "index": 0,
                "oscillator_ids": ["o0", "o1", "o2", "o3"],
                "omegas": [1.0, 1.0, 1.0, 1.0],
            }
        ],
        "oscillator_families": {
            "default": {
                "channel": "P",
                "extractor_type": "hilbert",
                "config": {},
            }
        },
        "coupling": {
            "base_strength": 0.4,
            "decay_alpha": 0.3,
            "templates": {},
        },
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
        },
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
