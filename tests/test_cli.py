from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

from scpn_phase_orchestrator.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def valid_spec_path(tmp_path):
    spec = {
        "name": "cli-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
        ],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": [1]},
        "boundaries": [],
        "actuators": [],
    }
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(path)


@pytest.fixture
def invalid_spec_path(tmp_path):
    spec = {
        "name": "",
        "version": "bad",
        "safety_tier": "unknown",
        "sample_period_s": -1,
        "control_period_s": 0.1,
        "layers": [],
        "oscillator_families": {},
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [], "bad_layers": []},
    }
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.dump(spec), encoding="utf-8")
    return str(path)


@pytest.fixture
def audit_log_path(tmp_path):
    log = tmp_path / "audit.jsonl"
    entries = [
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.8,
            "layers": [{"R": 0.8, "psi": 1.0}],
        },
        {
            "step": 1,
            "regime": "nominal",
            "stability": 0.9,
            "layers": [{"R": 0.9, "psi": 1.1}],
        },
        {"event": "boundary_violation", "detail": "test"},
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return str(log)


def test_validate_valid(runner, valid_spec_path):
    result = runner.invoke(main, ["validate", valid_spec_path])
    assert result.exit_code == 0
    assert "Valid" in result.output


def test_validate_invalid(runner, invalid_spec_path):
    result = runner.invoke(main, ["validate", invalid_spec_path])
    assert result.exit_code != 0
    assert "ERROR" in result.output


def test_run_simulation(runner, valid_spec_path):
    result = runner.invoke(main, ["run", valid_spec_path, "--steps", "10"])
    assert result.exit_code == 0
    assert "R_good" in result.output
    assert "R_bad" in result.output


def test_run_invalid_spec(runner, invalid_spec_path):
    result = runner.invoke(main, ["run", invalid_spec_path])
    assert result.exit_code != 0


def test_replay_command(runner, audit_log_path):
    result = runner.invoke(main, ["replay", audit_log_path])
    assert result.exit_code == 0
    assert "Steps logged: 2" in result.output
    assert "Events logged: 1" in result.output
    assert "Final regime: nominal" in result.output


def test_report_text(runner, audit_log_path):
    result = runner.invoke(main, ["report", audit_log_path])
    assert result.exit_code == 0
    assert "Steps: 2" in result.output
    assert "Layers: 1" in result.output
    assert "Final regime: nominal" in result.output
    assert "L0:" in result.output
    assert "Regime distribution:" in result.output
    assert "Hash chain:" in result.output


def test_report_json(runner, audit_log_path):
    result = runner.invoke(main, ["report", audit_log_path, "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["steps"] == 2
    assert data["layers"] == 1
    assert data["final_regime"] == "nominal"
    assert len(data["layer_r_mean"]) == 1
    assert data["hash_chain_ok"] is True


def test_scaffold_creates_structure(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["scaffold", "test_domain"])
    assert result.exit_code == 0
    assert (tmp_path / "domainpacks" / "test_domain" / "binding_spec.yaml").exists()
    assert (tmp_path / "domainpacks" / "test_domain" / "README.md").exists()


def test_scaffold_idempotent(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["scaffold", "test_domain"])
    result = runner.invoke(main, ["scaffold", "test_domain"])
    assert result.exit_code == 0


def test_run_applies_k_actions(runner, tmp_path):
    """Run with boundaries that trigger DEGRADED → supervisor emits K action."""
    spec = {
        "name": "k-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.1,
        "layers": [
            {"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]},
            {"name": "L2", "index": 1, "oscillator_ids": ["o2", "o3"]},
        ],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": [1]},
        "boundaries": [
            {
                "name": "R_low",
                "variable": "R",
                "lower": 0.8,
                "upper": None,
                "severity": "hard",
            },
        ],
        "actuators": [],
    }
    path = tmp_path / "k_spec.yaml"
    path.write_text(yaml.dump(spec), encoding="utf-8")
    result = runner.invoke(main, ["run", str(path), "--steps", "20"])
    assert result.exit_code == 0
    assert "R_good" in result.output
    assert "regime=" in result.output
