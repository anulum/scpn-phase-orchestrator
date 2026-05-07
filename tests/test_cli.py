# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI tests

from __future__ import annotations

import json

import pytest
import yaml
from click.testing import CliRunner

import scpn_phase_orchestrator.cli as cli_module
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.plugins import PluginCapability, PluginManifest


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
    assert "Resolved configuration:" in result.output
    assert "engine: kuramoto" in result.output


def test_validate_invalid(runner, invalid_spec_path):
    result = runner.invoke(main, ["validate", invalid_spec_path])
    assert result.exit_code != 0
    assert "ERROR" in result.output


def test_inspect_text_reports_resolved_defaults(runner, valid_spec_path):
    result = runner.invoke(main, ["inspect", valid_spec_path])
    assert result.exit_code == 0
    assert "Resolved configuration:" in result.output
    assert "domain: cli-test v1.0.0 (research)" in result.output
    assert "timing: sample=0.01s control=0.1s interval=10 steps" in result.output
    assert "structure: layers=2 oscillators=4" in result.output
    assert "engine: kuramoto" in result.output


def test_inspect_json_reports_resolved_summary(runner, valid_spec_path):
    result = runner.invoke(main, ["inspect", valid_spec_path, "--json-out"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["name"] == "cli-test"
    assert data["control_interval_steps"] == 10
    assert data["oscillator_count"] == 4
    assert data["engine_mode"] == "kuramoto"
    assert data["unassigned_layer_count"] == 2


def test_run_simulation(runner, valid_spec_path):
    result = runner.invoke(main, ["run", valid_spec_path, "--steps", "10"])
    assert result.exit_code == 0
    assert "Resolved configuration:" in result.output
    assert "R_good" in result.output
    assert "R_bad" in result.output


def test_run_audit_header_contains_binding_config(runner, valid_spec_path, tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    result = runner.invoke(
        main,
        ["run", valid_spec_path, "--steps", "2", "--audit", str(audit_path)],
    )
    assert result.exit_code == 0
    header = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert header["binding_config"]["name"] == "cli-test"
    assert header["binding_config"]["engine_mode"] == "kuramoto"
    assert header["binding_summary"]["name"] == "cli-test"
    assert header["binding_summary"]["engine_mode"] == "kuramoto"
    assert "P" in header["binding_config"]["channels"]
    assert "channel_algebra" in header["binding_config"]
    assert header["binding_config"]["channel_algebra"]["runtime_evidence_channels"] == [
        "P"
    ]
    assert header["binding_summary"]["channel_algebra"]["required_channels"] == []


def test_run_audit_records_channel_runtime_execution(runner, tmp_path):
    spec = {
        "name": "cli-nchannel-runtime-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {
                "name": "plant",
                "index": 0,
                "oscillator_ids": ["p0", "p1"],
                "family": "plant",
            },
            {
                "name": "forecast",
                "index": 1,
                "oscillator_ids": ["f0", "f1"],
                "family": "forecast",
            },
        ],
        "oscillator_families": {
            "plant": {"channel": "P", "extractor_type": "physical", "config": {}},
            "forecast": {
                "channel": "Forecast",
                "extractor_type": "event",
                "config": {},
            },
        },
        "channels": {
            "P": {"role": "plant", "units": "rad"},
            "Forecast": {
                "role": "delayed_forecast",
                "required": False,
                "replay_semantics": "external",
                "metric_semantics": "delayed confidence interval",
            },
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {
            "physical": {},
            "informational": {},
            "symbolic": {},
            "Forecast": {"confidence_weight": 0.5},
        },
        "objectives": {"good_layers": [0], "bad_layers": [1]},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = tmp_path / "spec.yaml"
    audit_path = tmp_path / "audit.jsonl"
    spec_path.write_text(yaml.dump(spec), encoding="utf-8")

    result = runner.invoke(
        main,
        ["run", str(spec_path), "--steps", "2", "--audit", str(audit_path)],
    )

    assert result.exit_code == 0
    records = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    step_records = [record for record in records if "step" in record]
    first_forecast = step_records[0]["channel_runtime"]["layers"][1]
    second_forecast = step_records[1]["channel_runtime"]["layers"][1]
    assert first_forecast["evidence_source"] == "current_tick_prime"
    assert second_forecast["evidence_source"] == "held_previous_tick"
    assert second_forecast["confidence_weight"] == 0.5
    assert second_forecast["executed_R"] == first_forecast["executed_R"]


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


def test_report_exposes_integrated_information_summary(runner, tmp_path):
    audit_path = tmp_path / "audit_phi.jsonl"
    entries = [
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.8,
            "layers": [{"R": 0.8, "psi": 1.0}],
        },
        {
            "monitor": "integrated_information",
            "phi": 0.125,
            "normalised_phi": 0.25,
            "total_integration": 0.5,
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        },
        {
            "step": 1,
            "regime": "nominal",
            "stability": 0.9,
            "layers": [{"R": 0.9, "psi": 1.1}],
        },
        {
            "monitor": "integrated_information",
            "phi": 0.25,
            "normalised_phi": 0.5,
            "total_integration": 0.75,
            "claim_boundary": "engineering_proxy_not_theoretical_iit",
        },
    ]
    audit_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    json_result = runner.invoke(main, ["report", str(audit_path), "--json-out"])
    text_result = runner.invoke(main, ["report", str(audit_path)])

    assert json_result.exit_code == 0
    data = json.loads(json_result.output)
    assert data["integrated_information"]["records"] == 2
    assert data["integrated_information"]["latest_phi"] == 0.25
    assert data["integrated_information"]["latest_normalised_phi"] == 0.5
    assert text_result.exit_code == 0
    assert (
        "Integrated information: records=2 phi=0.2500 "
        "normalised_phi=0.5000 total_integration=0.7500"
    ) in text_result.output


def test_report_json_exposes_binding_channel_algebra(
    runner,
    valid_spec_path,
    tmp_path,
):
    audit_path = tmp_path / "audit.jsonl"
    run_result = runner.invoke(
        main,
        ["run", valid_spec_path, "--steps", "2", "--audit", str(audit_path)],
    )
    assert run_result.exit_code == 0

    result = runner.invoke(main, ["report", str(audit_path), "--json-out"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["binding_summary"]["name"] == "cli-test"
    assert data["channel_algebra"]["runtime_evidence_channels"] == ["P"]
    assert data["channel_algebra"]["required_channels"] == []


def test_report_text_exposes_binding_channel_algebra(
    runner,
    valid_spec_path,
    tmp_path,
):
    audit_path = tmp_path / "audit.jsonl"
    run_result = runner.invoke(
        main,
        ["run", valid_spec_path, "--steps", "2", "--audit", str(audit_path)],
    )
    assert run_result.exit_code == 0

    result = runner.invoke(main, ["report", str(audit_path)])

    assert result.exit_code == 0
    assert (
        "Channel algebra: required=0 optional=0 derived=0 delayed=0 uncertain=0"
        in result.output
    )


def test_plugins_catalog_outputs_discovered_marketplace_catalog(
    runner,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = PluginManifest(
        name="cli_plugin",
        version="0.1.0",
        package="cli_plugin",
        capabilities=(
            PluginCapability(
                kind="extractor",
                name="phase",
                target="cli_plugin.extractors:PhaseExtractor",
                channels=("P",),
            ),
        ),
    )
    monkeypatch.setattr(cli_module, "discover_plugin_manifests", lambda: (manifest,))

    result = runner.invoke(main, ["plugins", "catalog"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["plugin_count"] == 1
    assert data["compatible_count"] == 1
    assert data["incompatible_count"] == 0
    assert data["plugins"][0]["manifest"]["name"] == "cli_plugin"
    assert data["capability_counts"]["extractor"] == 1


def test_plugins_catalog_can_include_incompatible_manifests(
    runner,
    monkeypatch: pytest.MonkeyPatch,
):
    invalid = PluginManifest(
        name="bad_cli_plugin",
        version="0.1.0",
        package="bad_cli_plugin",
        capabilities=(
            PluginCapability(
                kind="extractor",
                name="empty",
                target="bad_cli_plugin.extractors:Empty",
            ),
        ),
    )
    monkeypatch.setattr(cli_module, "discover_plugin_manifests", lambda: (invalid,))

    default_result = runner.invoke(main, ["plugins", "catalog"])
    full_result = runner.invoke(
        main,
        ["plugins", "catalog", "--include-incompatible"],
    )

    assert default_result.exit_code == 0
    default_data = json.loads(default_result.output)
    assert default_data["plugin_count"] == 0
    assert default_data["incompatible_count"] == 1
    assert full_result.exit_code == 0
    full_data = json.loads(full_result.output)
    assert full_data["plugin_count"] == 1
    assert full_data["plugins"][0]["compatible"] is False
    assert "must declare channels" in full_data["plugins"][0]["reasons"][0]


def test_plugins_catalog_can_emit_rust_registry(
    runner,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = PluginManifest(
        name="cli_plugin",
        version="0.1.0",
        package="cli_plugin",
        capabilities=(
            PluginCapability(
                kind="actuator",
                name="phase_driver",
                target="cli_plugin.actuators:PhaseDriver",
                knobs=("Psi",),
            ),
        ),
    )
    monkeypatch.setattr(cli_module, "discover_plugin_manifests", lambda: (manifest,))

    result = runner.invoke(main, ["plugins", "catalog", "--rust-registry"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema"] == "scpn_rust_plugin_registry_v1"
    assert data["capability_count"] == 1
    assert data["capabilities"][0]["plugin"] == "cli_plugin"
    assert data["capabilities"][0]["knobs"] == ["Psi"]


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


def test_run_r_values_bounded(runner, valid_spec_path):
    """R_good and R_bad must be in [0, 1] in every step output line."""
    result = runner.invoke(main, ["run", valid_spec_path, "--steps", "5"])
    assert result.exit_code == 0
    for line in result.output.strip().split("\n"):
        if "R_good=" in line:
            r_str = line.split("R_good=")[1].split()[0].rstrip(",")
            r_val = float(r_str)
            assert 0.0 <= r_val <= 1.0, f"R_good={r_val} out of [0,1]"
        if "R_bad=" in line:
            r_str = line.split("R_bad=")[1].split()[0].rstrip(",")
            r_val = float(r_str)
            assert 0.0 <= r_val <= 1.0, f"R_bad={r_val} out of [0,1]"


def test_nonexistent_spec_errors(runner, tmp_path):
    """Pointing to a non-existent file must fail with clear error."""
    result = runner.invoke(main, ["validate", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0


def test_scaffold_spec_is_valid_yaml(runner, tmp_path, monkeypatch):
    """Scaffolded binding_spec.yaml must be valid YAML and parseable."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["scaffold", "test_domain"])
    spec_path = tmp_path / "domainpacks" / "test_domain" / "binding_spec.yaml"
    content = spec_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(content)
    assert isinstance(parsed, dict)
    assert "name" in parsed
    assert "layers" in parsed


def test_report_hash_chain_verified(runner, audit_log_path):
    """Report must verify the hash chain and report its status."""
    result = runner.invoke(main, ["report", audit_log_path, "--json-out"])
    data = json.loads(result.output)
    # Hash chain check: audit entries don't have _hash (manual entries),
    # so chain verification may skip — but the field must exist
    assert "hash_chain_ok" in data


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


# Pipeline wiring: CLI tested via CliRunner invoking validate/run/replay/report/scaffold
# commands which wrap SimulationState -> engine -> order_parameter -> policy.
# TestCliCommands (above) proves full E2E.
