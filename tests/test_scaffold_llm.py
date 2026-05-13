# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — LLM-guided scaffold tests

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.scaffold.llm import (
    StaticJSONScaffoldProvider,
    propose_domainpack_from_description,
)


def _traffic_grid_payload() -> dict[str, object]:
    return {
        "name": "traffic_grid",
        "sample_period_s": 1.0,
        "control_period_s": 5.0,
        "safety_tier": "production",
        "oscillators": [
            {
                "id": "north_south",
                "channel": "I",
                "extractor_type": "event",
                "omega": 0.9,
            },
            {
                "id": "east_west",
                "channel": "I",
                "extractor_type": "event",
                "omega": 1.1,
            },
            {
                "id": "pedestrian_crossing",
                "channel": "S",
                "extractor_type": "ring",
                "omega": 0.7,
            },
            {
                "id": "queue_pressure",
                "channel": "P",
                "extractor_type": "physical",
                "omega": 1.0,
            },
        ],
        "coupling": {"base_strength": 0.22, "decay_alpha": 0.18},
        "boundaries": [
            {
                "name": "queue_pressure_limit",
                "variable": "queue_pressure",
                "lower": 0.0,
                "upper": 1.0,
                "severity": "hard",
            }
        ],
        "actuators": [
            {
                "name": "signal_coupling",
                "knob": "K",
                "scope": "global",
                "limits": [0.0, 1.5],
            }
        ],
    }


def test_llm_scaffold_provider_generates_valid_binding() -> None:
    provider = StaticJSONScaffoldProvider(
        json.dumps(_traffic_grid_payload()),
        provider_name="fixture",
    )
    proposal = propose_domainpack_from_description(
        "I am modelling traffic lights in a 4-intersection grid",
        project_name="traffic_grid",
        provider=provider,
    )

    assert proposal.validation_errors == ()
    assert proposal.provenance["provider"] == "fixture"
    assert proposal.provenance["input_family"] == "llm_scaffold"
    raw = yaml.safe_load(proposal.yaml_text)
    assert raw["name"] == "traffic_grid"
    assert len(raw["layers"]) == 4
    assert raw["layers"][0]["oscillator_ids"] == ["north_south"]
    assert raw["coupling"]["base_strength"] == pytest.approx(0.22)


def test_llm_scaffold_cli_writes_valid_domainpack(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    response_path = tmp_path / "response.json"
    response_path.write_text(json.dumps(_traffic_grid_payload()), encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "scaffold",
            "traffic_grid",
            "--llm",
            "--description",
            "I am modelling traffic lights in a 4-intersection grid",
            "--llm-response-json",
            str(response_path),
        ],
    )

    assert result.exit_code == 0, result.output
    spec_path = tmp_path / "domainpacks" / "traffic_grid" / "binding_spec.yaml"
    readme_path = tmp_path / "domainpacks" / "traffic_grid" / "README.md"
    audit_path = tmp_path / "domainpacks" / "traffic_grid" / "llm_scaffold_audit.json"
    spec = load_binding_spec(spec_path)
    assert validate_binding_spec(spec) == []
    assert "LLM-assisted domainpack scaffold" in readme_path.read_text(encoding="utf-8")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["validation_errors"] == []
    assert audit["provider"] == "static-json"


def test_llm_scaffold_cli_fails_closed_without_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SPO_LLM_ENDPOINT", raising=False)
    monkeypatch.delenv("SPO_LLM_MODEL", raising=False)

    result = CliRunner().invoke(
        main,
        [
            "scaffold",
            "traffic_grid",
            "--llm",
            "--description",
            "I am modelling traffic lights in a 4-intersection grid",
        ],
    )

    assert result.exit_code != 0
    assert "LLM scaffold provider is not configured" in result.output
    assert not (tmp_path / "domainpacks" / "traffic_grid").exists()


def test_llm_scaffold_rejects_invalid_output() -> None:
    provider = StaticJSONScaffoldProvider(
        json.dumps({"name": "traffic_grid", "oscillators": []})
    )

    with pytest.raises(ValueError, match="at least one oscillator"):
        propose_domainpack_from_description(
            "I am modelling traffic lights",
            project_name="traffic_grid",
            provider=provider,
        )
