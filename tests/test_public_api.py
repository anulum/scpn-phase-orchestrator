# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Public Python API tests

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from scpn import Orchestrator
from scpn_phase_orchestrator import Orchestrator as LongNameOrchestrator
from scpn_phase_orchestrator import api as api_mod
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.binding.types import AmplitudeSpec, BindingSpec


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


def _load_minimal_binding_spec(tmp_path: Path) -> BindingSpec:
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(_minimal_spec()), encoding="utf-8")
    return load_binding_spec(spec_path)


def test_orchestrator_state_to_record_returns_scalar_summary() -> None:
    state = api_mod.OrchestratorState(
        spec_name="probe",
        steps=4,
        phases=np.zeros(4, dtype=np.float64),
        omegas=np.zeros(4, dtype=np.float64),
        knm=np.zeros((4, 4), dtype=np.float64),
        alpha=np.zeros((4, 4), dtype=np.float64),
        order_parameter=0.5,
        mean_phase=0.1,
        sample_period_s=0.01,
    )

    assert state.to_record() == {
        "spec_name": "probe",
        "steps": 4,
        "oscillator_count": 4,
        "order_parameter": 0.5,
        "mean_phase": 0.1,
        "sample_period_s": 0.01,
    }


def test_orchestrator_rejects_non_binding_spec() -> None:
    with pytest.raises(TypeError, match="spec must be BindingSpec"):
        Orchestrator(cast(BindingSpec, "not-a-spec"))


def test_validate_executable_spec_reports_validation_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _load_minimal_binding_spec(tmp_path)
    monkeypatch.setattr(api_mod, "validate_binding_spec", lambda _spec: ["bad knm"])

    with pytest.raises(ValueError, match="binding spec validation failed: bad knm"):
        Orchestrator._validate_executable_spec(spec)


def test_validate_executable_spec_rejects_non_research_tier(tmp_path: Path) -> None:
    spec = dataclasses.replace(
        _load_minimal_binding_spec(tmp_path), safety_tier="clinical"
    )

    with pytest.raises(ValueError, match="safety_tier='clinical' is not executable"):
        Orchestrator._validate_executable_spec(spec)


def test_validate_executable_spec_rejects_amplitude_mode_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = dataclasses.replace(
        _load_minimal_binding_spec(tmp_path),
        amplitude=cast(AmplitudeSpec, object()),
    )
    monkeypatch.setattr(api_mod, "validate_binding_spec", lambda _spec: [])

    with pytest.raises(ValueError, match="supports Kuramoto binding specs"):
        Orchestrator._validate_executable_spec(spec)


@pytest.mark.parametrize("bad", [True, 1.5])
def test_nonnegative_int_rejects_bool_and_non_integer(bad: object) -> None:
    with pytest.raises(ValueError, match="steps must be a non-negative integer"):
        api_mod._nonnegative_int(bad, name="steps")


def test_oscillator_count_requires_at_least_one_oscillator(tmp_path: Path) -> None:
    spec = dataclasses.replace(_load_minimal_binding_spec(tmp_path), layers=[])

    with pytest.raises(ValueError, match="must define at least one oscillator"):
        api_mod._oscillator_count(spec)


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
