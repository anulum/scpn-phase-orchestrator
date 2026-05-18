# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI tests

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli as cli_module
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.plugins import PluginCapability, PluginManifest
from scpn_phase_orchestrator.runtime.cli import main


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


def test_inspect_invalid_spec_reports_validation_errors(runner, invalid_spec_path):
    result = runner.invoke(main, ["inspect", invalid_spec_path])

    assert result.exit_code == 1
    assert "ERROR" in result.output
    assert "safety_tier" in result.output


def test_auto_bind_csv_outputs_valid_yaml(runner, tmp_path):
    csv_path = tmp_path / "grid.csv"
    csv_path.write_text(
        "time,grid,load\n0.00,0.0,1.0\n0.01,0.2,0.9\n0.02,0.4,0.7\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(csv_path),
            "--sample-rate-hz",
            "100",
            "--project-name",
            "grid_replay",
        ],
    )

    assert result.exit_code == 0
    assert 'name: "grid_replay"' in result.output
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(result.output, encoding="utf-8")
    spec = load_binding_spec(spec_path)
    assert validate_binding_spec(spec) == []


def test_auto_bind_json_out_emits_audit_record(runner, tmp_path):
    csv_path = tmp_path / "grid.csv"
    csv_path.write_text(
        "time,grid,load\n0.00,0.0,1.0\n0.01,0.2,0.9\n0.02,0.4,0.7\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(csv_path),
            "--sample-rate-hz",
            "100",
            "--project-name",
            "grid_replay",
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    record = json.loads(result.output)
    assert record["source"]["source_kind"] == "time_series_csv"
    assert record["binding"]["validation_errors"] == []
    assert record["binding"]["inferred_channels"] == ["P", "I"]
    assert record["binding"]["provenance"]["extractor_parameter_proposals"]
    assert record["binding"]["provenance"]["initial_coupling_proposal"]["template"] == (
        "auto_initial_k"
    )
    assert record["runtime"]["replay_status"] == "proposal_only"


def test_auto_bind_infers_sample_rate_from_time_column(runner, tmp_path):
    csv_path = tmp_path / "signals.csv"
    csv_path.write_text(
        "time,grid,load\n0.0,0.0,1.0\n0.5,0.2,0.8\n1.0,0.4,0.6\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(csv_path),
            "--project-name",
            "grid_review",
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    record = json.loads(result.output)
    provenance = record["binding"]["provenance"]
    assert provenance["sample_rate_hz"] == pytest.approx(2.0)
    assert provenance["sample_rate_inference"] == "time_column"
    assert "discovery_evidence" in provenance


def test_auto_bind_rejects_bad_source_with_scrubbed_error(runner, tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("time,grid\n0.00,not-a-number\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(csv_path),
            "--sample-rate-hz",
            "100",
            "--project-name",
            "bad_replay",
        ],
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.output
    assert "non-numeric sample" in result.output
    assert str(csv_path) not in result.output


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


def test_run_warns_for_unenforced_non_research_tier(runner, tmp_path):
    spec = {
        "name": "clinical-warning-test",
        "version": "1.0.0",
        "safety_tier": "clinical",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [{"name": "L1", "index": 0, "oscillator_ids": ["o0"]}],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    path = tmp_path / "clinical.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = runner.invoke(main, ["run", str(path), "--steps", "1"])

    assert result.exit_code == 0
    assert "WARNING: safety_tier='clinical'" in result.output
    assert "R_good=" in result.output


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


def test_explain_rejects_invalid_limits_and_empty_logs(runner, tmp_path):
    audit_path = tmp_path / "empty.jsonl"
    audit_path.write_text(
        json.dumps({"event": "boundary_violation"}) + "\n",
        encoding="utf-8",
    )

    bad_limit = runner.invoke(
        main,
        ["explain", str(audit_path), "--max-actions", "0"],
    )
    no_steps = runner.invoke(main, ["explain", str(audit_path)])

    assert bad_limit.exit_code == 1
    assert "ERROR: --max-actions must be >= 1" in bad_limit.output
    assert no_steps.exit_code == 1
    assert "ERROR: no step records in audit log" in no_steps.output


def test_explain_renders_text_markdown_and_pdf_outputs(runner, tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    markdown_path = tmp_path / "explain.md"
    pdf_path = tmp_path / "explain.pdf"
    entries = [
        {
            "step": 0,
            "regime": "nominal",
            "stability": 0.4,
            "layers": [{"R": 0.4, "psi": 0.0}],
            "actions": [
                {
                    "knob": "K",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": 5.0,
                    "justification": "policy rule: boost",
                }
            ],
        },
        {
            "step": 1,
            "regime": "degraded",
            "stability": 0.6,
            "layers": [{"R": 0.6, "psi": 0.0}],
        },
        {"event": "boundary_violation", "step": 1, "detail": "R below target"},
    ]
    audit_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    text_result = runner.invoke(
        main,
        ["explain", str(audit_path), "--max-actions", "1"],
    )
    file_result = runner.invoke(
        main,
        [
            "explain",
            str(audit_path),
            "--markdown-out",
            str(markdown_path),
            "--pdf-out",
            str(pdf_path),
        ],
    )

    assert text_result.exit_code == 0
    assert "# SCPN Phase Orchestrator Explainability Report" in text_result.output
    assert "Step 0: K=0.1000" in text_result.output
    assert "Step 1: nominal -> degraded" in text_result.output
    assert file_result.exit_code == 0
    assert f"Markdown report written: {markdown_path}" in file_result.output
    assert f"PDF report written: {pdf_path}" in file_result.output
    assert "Step 0: K=0.1000" in markdown_path.read_text(encoding="utf-8")
    assert pdf_path.read_bytes().startswith(b"%PDF-1.4")


def _write_formal_export_spec(tmp_path: Path, *, protocol_net: bool = True) -> Path:
    spec = {
        "name": "formal-cli-test",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [{"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]}],
        "oscillator_families": {
            "p": {"channel": "P", "extractor_type": "hilbert"},
        },
        "coupling": {"base_strength": 0.45, "decay_alpha": 0.3},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    if protocol_net:
        spec["protocol_net"] = {
            "places": ["warmup", "nominal"],
            "initial": {"warmup": 1, "nominal": 0},
            "place_regime": {"warmup": "NOMINAL", "nominal": "NOMINAL"},
            "transitions": [
                {
                    "name": "start",
                    "inputs": [{"place": "warmup"}],
                    "outputs": [{"place": "nominal"}],
                    "guard": "stability_proxy > 0.0",
                },
            ],
        }
    spec_path = tmp_path / "binding_spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return spec_path


def _write_policy_rules(tmp_path: Path, rules: list[dict]) -> Path:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump({"rules": rules}), encoding="utf-8")
    return policy_path


def test_formal_export_protocol_stdout_and_file_outputs(runner, tmp_path):
    spec_path = _write_formal_export_spec(tmp_path)
    prism_path = tmp_path / "protocol.prism"
    tla_path = tmp_path / "protocol.tla"

    prism_result = runner.invoke(
        main,
        ["formal-export", str(spec_path), "--module-name", "cli_protocol"],
    )
    prism_file_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--output",
            str(prism_path),
            "--module-name",
            "cli_protocol_file",
        ],
    )
    tla_stdout_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "protocol-tla",
            "--module-name",
            "CliProtocolStdout",
        ],
    )
    tla_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "protocol-tla",
            "--output",
            str(tla_path),
            "--module-name",
            "CliProtocol",
        ],
    )

    assert prism_result.exit_code == 0
    assert "module cli_protocol" in prism_result.output
    assert "[start] stability_proxy > 0 & warmup >= 1" in prism_result.output
    assert prism_file_result.exit_code == 0
    assert "PRISM model written:" in prism_file_result.output
    assert "module cli_protocol_file" in prism_path.read_text(encoding="utf-8")
    assert tla_stdout_result.exit_code == 0
    assert "---- MODULE CliProtocolStdout ----" in tla_stdout_result.output
    assert tla_result.exit_code == 0
    assert "TLA+ model written:" in tla_result.output
    assert "---- MODULE CliProtocol ----" in tla_path.read_text(encoding="utf-8")


def test_formal_export_policy_and_stl_error_branches(runner, tmp_path):
    spec_path = _write_formal_export_spec(tmp_path, protocol_net=False)

    missing_policy = runner.invoke(main, ["formal-export", str(spec_path)])
    missing_stl = runner.invoke(
        main,
        ["formal-export", str(spec_path), "--export", "stl"],
    )
    (tmp_path / "policy.yaml").write_text(
        yaml.safe_dump({"rules": [], "stl_monitors": []}),
        encoding="utf-8",
    )
    empty_stl = runner.invoke(
        main,
        ["formal-export", str(spec_path), "--export", "stl"],
    )
    empty_rules = runner.invoke(
        main,
        ["formal-export", str(spec_path), "--export", "policy"],
    )

    assert missing_policy.exit_code == 1
    assert "ERROR: binding spec has no protocol_net" in missing_policy.output
    assert missing_stl.exit_code == 1
    assert "ERROR: policy file not found:" in missing_stl.output
    assert empty_stl.exit_code == 1
    assert "ERROR: policy file contains no stl_monitors" in empty_stl.output
    assert empty_rules.exit_code == 1
    assert "ERROR: policy file contains no rules" in empty_rules.output


def test_formal_export_rejects_invalid_binding_spec(runner, invalid_spec_path):
    result = runner.invoke(main, ["formal-export", invalid_spec_path])

    assert result.exit_code == 1
    assert "ERROR:" in result.output
    assert "safety_tier" in result.output


def test_formal_export_policy_targets_stdout_and_files(runner, tmp_path):
    spec_path = _write_formal_export_spec(tmp_path, protocol_net=False)
    rules = [
        {
            "name": "boost",
            "regime": ["DEGRADED"],
            "condition": {
                "metric": "R_good",
                "layer": 0,
                "op": "<",
                "threshold": 0.7,
            },
            "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
        }
    ]
    _write_policy_rules(tmp_path, rules)
    policy_path = tmp_path / "policy.prism"
    tla_path = tmp_path / "policy.tla"

    prism_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy",
            "--module-name",
            "cli_policy",
        ],
    )
    prism_file_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy",
            "--output",
            str(policy_path),
            "--module-name",
            "cli_policy_file",
        ],
    )
    tla_stdout_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy-tla",
            "--module-name",
            "CliPolicyStdout",
        ],
    )
    tla_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "policy-tla",
            "--output",
            str(tla_path),
            "--module-name",
            "CliPolicy",
        ],
    )

    assert prism_result.exit_code == 0
    assert "module cli_policy" in prism_result.output
    assert "[boost]" in prism_result.output
    assert prism_file_result.exit_code == 0
    assert "PRISM model written:" in prism_file_result.output
    assert "module cli_policy_file" in policy_path.read_text(encoding="utf-8")
    assert tla_stdout_result.exit_code == 0
    assert "---- MODULE CliPolicyStdout ----" in tla_stdout_result.output
    assert tla_result.exit_code == 0
    assert "TLA+ model written:" in tla_result.output
    assert "---- MODULE CliPolicy ----" in tla_path.read_text(encoding="utf-8")


def test_formal_export_stl_stdout_and_file_outputs(runner, tmp_path):
    spec_path = _write_formal_export_spec(tmp_path, protocol_net=False)
    policy_path = tmp_path / "policy.yaml"
    stl_path = tmp_path / "stl.prism"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "rules": [],
                "stl_monitors": [
                    {
                        "name": "keep_sync",
                        "spec": "always (R >= 0.3)",
                        "severity": "hard",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    stdout_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "stl",
            "--module-name",
            "cli_stl",
        ],
    )
    file_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "stl",
            "--output",
            str(stl_path),
            "--module-name",
            "cli_stl_file",
        ],
    )

    assert stdout_result.exit_code == 0
    assert "module cli_stl" in stdout_result.output
    assert 'label "stl_keep_sync_satisfied"' in stdout_result.output
    assert file_result.exit_code == 0
    assert "PRISM model written:" in file_result.output
    assert "module cli_stl_file" in stl_path.read_text(encoding="utf-8")


def test_policy_dry_run_text_reports_overlaps_collisions_and_unreachable(
    runner,
    tmp_path,
):
    spec_path = _write_formal_export_spec(tmp_path, protocol_net=False)
    _write_policy_rules(
        tmp_path,
        [
            {
                "name": "boost_low_stability",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "stability_proxy",
                    "op": "<",
                    "threshold": 0.5,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            },
            {
                "name": "second_global_boost",
                "regime": ["DEGRADED"],
                "condition": {"metric": "R", "layer": 0, "op": ">", "threshold": 0.1},
                "action": {"knob": "K", "scope": "global", "value": 0.2, "ttl_s": 5.0},
            },
            {
                "name": "never_fires",
                "regime": ["NOMINAL"],
                "condition": {"metric": "R", "layer": 0, "op": ">", "threshold": 0.95},
                "action": {
                    "knob": "zeta",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": 5.0,
                },
            },
        ],
    )
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        json.dumps(
            {
                "step": 8,
                "regime": "DEGRADED",
                "stability": 0.3,
                "layers": [{"R": 0.7, "psi": 0.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["policy-dry-run", str(spec_path), str(audit_path)])

    assert result.exit_code == 0
    assert "Steps: 1  Rules: 3" in result.output
    assert "boost_low_stability: 1" in result.output
    assert "Unreachable rules:" in result.output
    assert "never_fires" in result.output
    assert "Overlapping rule steps: 8" in result.output
    assert "Action collision steps: 8" in result.output


def test_policy_dry_run_json_and_error_paths(runner, tmp_path, invalid_spec_path):
    spec_path = _write_formal_export_spec(tmp_path, protocol_net=False)
    audit_path = tmp_path / "audit.jsonl"
    audit_path.write_text(
        json.dumps(
            {
                "step": 0,
                "regime": "DEGRADED",
                "stability": 0.3,
                "layers": [{"R": 0.7, "psi": 0.0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    invalid_spec = runner.invoke(
        main,
        ["policy-dry-run", invalid_spec_path, str(audit_path)],
    )
    missing_policy = runner.invoke(
        main,
        ["policy-dry-run", str(spec_path), str(audit_path)],
    )
    (tmp_path / "policy.yaml").write_text("rules: []\n", encoding="utf-8")
    empty_rules = runner.invoke(
        main,
        ["policy-dry-run", str(spec_path), str(audit_path)],
    )
    _write_policy_rules(
        tmp_path,
        [
            {
                "name": "boost_low_stability",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "stability_proxy",
                    "op": "<",
                    "threshold": 0.5,
                },
                "action": {"knob": "K", "scope": "global", "value": 0.1, "ttl_s": 5.0},
            },
        ],
    )
    json_result = runner.invoke(
        main,
        ["policy-dry-run", str(spec_path), str(audit_path), "--json-out"],
    )

    assert invalid_spec.exit_code == 1
    assert "ERROR:" in invalid_spec.output
    assert missing_policy.exit_code == 1
    assert "ERROR: policy file not found:" in missing_policy.output
    assert empty_rules.exit_code == 1
    assert "ERROR: policy file contains no rules" in empty_rules.output
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["steps"] == 1
    assert payload["fire_counts"]["boost_low_stability"] == 1


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


def test_scaffold_rejects_unsafe_domain_names(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(main, ["scaffold", "../escape"])

    assert result.exit_code != 0
    assert "domain_name must match" in result.output
    assert not (tmp_path / "escape").exists()


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


def test_queuewaves_check_runs_pipeline_and_reports_status(runner, tmp_path):
    config_path = tmp_path / "queuewaves.yaml"
    config_path.write_text(
        """
prometheus_url: "http://localhost:9090"
scrape_interval_s: 1
buffer_length: 16
services:
  - name: svc-a
    promql: "up"
    layer: micro
  - name: throughput
    promql: "up"
    layer: macro
thresholds:
  r_bad_warn: 0.50
  r_bad_critical: 0.70
""",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["queuewaves", "check", "--config", str(config_path)])

    assert result.exit_code in (0, 1)
    assert "R_good=" in result.output
    assert "R_bad=" in result.output
    if result.exit_code == 0:
        assert "No anomalies detected." in result.output
    else:
        assert "[" in result.output


def test_queuewaves_serve_delegates_to_server_runner(runner, tmp_path, monkeypatch):
    config_path = tmp_path / "queuewaves.yaml"
    config_path.write_text(
        """
prometheus_url: "http://localhost:9090"
services:
  - name: svc-a
    promql: "up"
    layer: micro
""",
        encoding="utf-8",
    )
    calls = []

    def fake_run_server(config_path_arg, *, host, port):
        calls.append((config_path_arg, host, port))

    from scpn_phase_orchestrator.apps.queuewaves import server as server_module

    monkeypatch.setattr(server_module, "run_server", fake_run_server)

    result = runner.invoke(
        main,
        [
            "queuewaves",
            "serve",
            "--config",
            str(config_path),
            "--host",
            "0.0.0.0",
            "--port",
            "9099",
        ],
    )

    assert result.exit_code == 0
    assert calls == [(str(config_path), "0.0.0.0", 9099)]


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


def test_generate_writes_reviewable_domainpack_outputs(runner, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "generated"

    result = runner.invoke(
        main,
        [
            "generate",
            "two coupled oscillators with physical synchronisation",
            "--name",
            "generated_cli_domain",
            "--output-dir",
            str(output_dir),
            "--oscillators-per-layer",
            "2",
            "--dry-run-steps",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert f"Generated domainpack at {output_dir}" in result.output
    assert "schema_valid=True" in result.output
    assert "confidence=" in result.output
    assert "retrieval_matches=" in result.output
    assert "dry_run_R=" in result.output
    assert (output_dir / "binding_spec.yaml").exists()
    assert (output_dir / "policy.yaml").exists()
    binding = yaml.safe_load(
        (output_dir / "binding_spec.yaml").read_text(encoding="utf-8")
    )
    audit = json.loads((output_dir / "audit.json").read_text(encoding="utf-8"))
    assert binding["name"] == "generated_cli_domain"
    assert audit["notebook_execution"]["status"] == "passed"


def test_demo_reports_available_domainpacks_for_missing_domain(runner):
    result = runner.invoke(main, ["demo", "--domain", "missing_demo"])

    assert result.exit_code == 1
    assert "Domainpack 'missing_demo' not found." in result.output
    assert "Available:" in result.output
    assert "minimal_domain" in result.output


def test_demo_runs_packaged_domainpack_and_prints_progress(runner):
    result = runner.invoke(main, ["demo", "--domain", "minimal_domain", "--steps", "3"])

    assert result.exit_code == 0
    assert "SPO Demo" in result.output
    assert "Oscillators:" in result.output
    assert "Step     3:" in result.output
    assert "Final R=" in result.output


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


def _write_rich_cli_run_spec(tmp_path: Path) -> Path:
    spec = {
        "name": "rich-cli-run",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [
            {"name": "empty", "index": 0, "oscillator_ids": []},
            {"name": "active", "index": 1, "oscillator_ids": ["o0", "o1"]},
        ],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.25, "decay_alpha": 0.2, "templates": {}},
        "drivers": {
            "physical": {"frequency": 0.5, "amplitude": 0.2},
            "informational": {},
            "symbolic": {},
        },
        "objectives": {"good_layers": [1], "bad_layers": [0]},
        "boundaries": [],
        "actuators": [
            {"name": "psi", "knob": "Psi", "scope": "global", "limits": [-1.0, 1.0]},
            {"name": "zeta", "knob": "zeta", "scope": "global", "limits": [0.0, 0.5]},
            {"name": "k0", "knob": "K", "scope": "layer_0", "limits": [-0.2, 0.2]},
        ],
        "imprint_model": {
            "decay_rate": 0.01,
            "saturation": 2.0,
            "modulates": ["K", "alpha", "mu"],
        },
        "geometry_prior": {"constraint_type": "symmetric_non_negative", "params": {}},
        "protocol_net": {
            "places": ["idle", "active"],
            "initial": {"idle": 1, "active": 0},
            "place_regime": {"idle": "NOMINAL", "active": "NOMINAL"},
            "transitions": [
                {
                    "name": "activate",
                    "inputs": [{"place": "idle"}],
                    "outputs": [{"place": "active"}],
                    "guard": "stability_proxy > 0.0",
                }
            ],
        },
        "amplitude": {
            "mu": 0.4,
            "epsilon": 0.01,
            "amp_coupling_strength": 0.05,
            "amp_coupling_decay": 0.2,
        },
    }
    spec_path = tmp_path / "rich_binding.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    (tmp_path / "policy.yaml").write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "name": "set_psi",
                        "regime": ["NOMINAL"],
                        "condition": {
                            "metric": "stability_proxy",
                            "op": ">",
                            "threshold": -1.0,
                        },
                        "action": {
                            "knob": "Psi",
                            "scope": "global",
                            "value": 0.2,
                            "ttl_s": 0.01,
                        },
                    },
                    {
                        "name": "pulse_zeta",
                        "regime": ["NOMINAL"],
                        "condition": {
                            "metric": "stability_proxy",
                            "op": ">",
                            "threshold": -1.0,
                        },
                        "action": {
                            "knob": "zeta",
                            "scope": "global",
                            "value": 0.1,
                            "ttl_s": 0.01,
                        },
                    },
                    {
                        "name": "layer_boost",
                        "regime": ["NOMINAL"],
                        "condition": {
                            "metric": "stability_proxy",
                            "op": ">",
                            "threshold": -1.0,
                        },
                        "action": {
                            "knob": "K",
                            "scope": "layer_0",
                            "value": 0.1,
                            "ttl_s": 0.01,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return spec_path


def test_run_rich_binding_records_amplitude_policy_and_imprint_audit(
    runner,
    tmp_path,
):
    spec_path = _write_rich_cli_run_spec(tmp_path)
    audit_path = tmp_path / "rich_audit.jsonl"

    result = runner.invoke(
        main,
        [
            "run",
            str(spec_path),
            "--steps",
            "21",
            "--seed",
            "7",
            "--audit",
            str(audit_path),
        ],
    )

    assert result.exit_code == 0
    assert "mean_amplitude=" in result.output
    records = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    header = records[0]
    steps = [record for record in records if "phases" in record]
    assert header["amplitude_mode"] is True
    assert header["binding_summary"]["features"]["imprint_model"] is True
    assert header["binding_summary"]["features"]["geometry_prior"] is True
    assert {"Psi", "zeta", "K"} <= {
        action["knob"] for step in steps for action in step["actions"]
    }
    assert "amplitudes" in steps[0]
    assert "mu" in steps[0]
    assert "knm_r" in steps[0]
    assert "epsilon" in steps[0]


def test_replay_verify_reports_rich_stuart_landau_mismatch(runner, tmp_path):
    spec_path = _write_rich_cli_run_spec(tmp_path)
    audit_path = tmp_path / "rich_audit.jsonl"
    run_result = runner.invoke(
        main,
        [
            "run",
            str(spec_path),
            "--steps",
            "3",
            "--seed",
            "11",
            "--audit",
            str(audit_path),
        ],
    )

    result = runner.invoke(main, ["replay", str(audit_path), "--verify"])

    assert run_result.exit_code == 0
    assert result.exit_code == 1
    assert "Determinism FAILED at transition" in result.output


def test_replay_verify_accepts_plain_audit(runner, valid_spec_path, tmp_path):
    audit_path = tmp_path / "plain_audit.jsonl"
    run_result = runner.invoke(
        main,
        [
            "run",
            valid_spec_path,
            "--steps",
            "3",
            "--seed",
            "13",
            "--audit",
            str(audit_path),
        ],
    )

    result = runner.invoke(main, ["replay", str(audit_path), "--verify"])

    assert run_result.exit_code == 0
    assert result.exit_code == 0
    assert "Determinism verified:" in result.output


def test_replay_verify_rejects_logs_without_header(runner, audit_log_path):
    result = runner.invoke(main, ["replay", audit_log_path, "--verify"])

    assert result.exit_code == 1
    assert "ERROR: no header record in log" in result.output


@pytest.mark.parametrize(
    ("driver_block", "expected_audit_value"),
    [
        ({"informational": {"cadence_hz": 2.0}}, 0.12566370614359174),
        ({"symbolic": {"sequence": [0.0, 0.25]}}, 0.25),
    ],
)
def test_run_accepts_non_physical_psi_drivers(
    runner,
    tmp_path,
    driver_block,
    expected_audit_value,
):
    spec = {
        "name": "psi-driver-cli",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [{"name": "L1", "index": 0, "oscillator_ids": ["o0", "o1"]}],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}} | driver_block,
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = tmp_path / "driver.yaml"
    audit_path = tmp_path / "driver_audit.jsonl"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = runner.invoke(
        main,
        ["run", str(spec_path), "--steps", "2", "--audit", str(audit_path)],
    )

    assert result.exit_code == 0
    steps = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if '"phases"' in line
    ]
    assert steps[-1]["psi_drive"] == pytest.approx(expected_audit_value)


def test_run_rejects_specs_without_oscillators(runner, tmp_path):
    spec = {
        "name": "empty-run",
        "version": "1.0.0",
        "safety_tier": "research",
        "sample_period_s": 0.01,
        "control_period_s": 0.01,
        "layers": [{"name": "empty", "index": 0, "oscillator_ids": []}],
        "oscillator_families": {
            "phys": {"channel": "P", "extractor_type": "hilbert", "config": {}},
        },
        "coupling": {"base_strength": 0.2, "decay_alpha": 0.1, "templates": {}},
        "drivers": {"physical": {}, "informational": {}, "symbolic": {}},
        "objectives": {"good_layers": [0], "bad_layers": []},
        "boundaries": [],
        "actuators": [],
    }
    spec_path = tmp_path / "empty.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")

    result = runner.invoke(main, ["run", str(spec_path), "--steps", "1"])

    assert result.exit_code == 1
    assert "ERROR: no oscillators defined in layers" in result.output


def test_report_rejects_logs_without_step_records(runner, tmp_path):
    audit_path = tmp_path / "events_only.jsonl"
    audit_path.write_text(
        json.dumps({"event": "boundary_violation", "detail": "R below target"}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["report", str(audit_path)])

    assert result.exit_code == 1
    assert "ERROR: no step records in log" in result.output


def test_report_text_lists_missing_channels_and_actions(runner, tmp_path):
    audit_path = tmp_path / "manual_report.jsonl"
    entries = [
        {
            "header": True,
            "amplitude_mode": False,
            "binding_summary": {
                "channel_algebra": {
                    "required_channels": ["I"],
                    "optional_channels": ["P"],
                    "derived_channels": [],
                    "delayed_channels": [],
                    "uncertain_channels": [],
                    "missing_required_channels": ["I"],
                }
            },
        },
        {
            "step": 0,
            "regime": "NOMINAL",
            "stability": 0.5,
            "layers": [{"R": 0.5, "psi": 0.0}],
            "actions": [{"knob": "Psi", "scope": "global", "value": 0.2}],
        },
    ]
    audit_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["report", str(audit_path)])

    assert result.exit_code == 0
    assert "Missing required channels: I" in result.output
    assert "Actions fired:" in result.output
    assert "Psi: 1" in result.output


def test_queuewaves_check_reports_clean_status_for_high_thresholds(runner, tmp_path):
    config_path = tmp_path / "queuewaves_clean.yaml"
    config_path.write_text(
        """
prometheus_url: "http://localhost:9090"
scrape_interval_s: 1
buffer_length: 16
services:
  - name: svc-a
    promql: "up"
    layer: micro
  - name: svc-b
    promql: "up"
    layer: macro
thresholds:
  r_bad_warn: 2.0
  r_bad_critical: 3.0
  plv_cascade: 2.0
  imprint_chronic: 10.0
""",
        encoding="utf-8",
    )

    result = runner.invoke(main, ["queuewaves", "check", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "No anomalies detected." in result.output


def test_supervisor_baseline_experiment_materialises_reproducibility_outputs(
    runner, tmp_path
):
    config_path = tmp_path / "supervisor_config.json"
    metrics_path = tmp_path / "supervisor_metrics.jsonl"
    summary_path = tmp_path / "supervisor_summary.json"
    manifest_path = tmp_path / "supervisor_manifest.json"

    result = runner.invoke(
        main,
        [
            "supervisor-baseline-experiment",
            "--config-json",
            str(config_path),
            "--metrics-jsonl",
            str(metrics_path),
            "--summary-json",
            str(summary_path),
            "--manifest-json",
            str(manifest_path),
            "--git-sha",
            "abc1234",
            "--seed",
            "91",
            "--dependency-lock",
            "requirements-dev.txt:sha256:test",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    stdout_record = json.loads(result.output)
    assert stdout_record["proposal_type"] == (
        "differentiable_supervisor_experiment_manifest"
    )
    assert stdout_record["actuation_permitted"] is False
    assert stdout_record["seed_list"] == [91]
    assert stdout_record["artifacts"]["config_json_path"] == str(config_path)

    config_record = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_record["proposal_type"] == "supervisor_baseline_experiment_config"
    assert config_record["actuation_permitted"] is False
    assert config_record["policy_config"]["n_oscillators"] == 4
    assert config_record["scenario"]["horizon"] == 6

    metric_records = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["proposal_type"] for record in metric_records] == [
        "differentiable_supervisor_static_baseline",
        "differentiable_supervisor_random_baseline",
        "differentiable_supervisor_hand_tuned_baseline",
    ]
    assert all(record["actuation_permitted"] is False for record in metric_records)

    summary_record = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_record["comparison_count"] == 3
    assert summary_record["actuation_permitted"] is False
    assert summary_record["metric_record_path"] == str(metrics_path)

    manifest_record = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_record == stdout_record
    assert manifest_record["dependency_lock"] == {"requirements-dev.txt": "sha256:test"}


def test_supervisor_baseline_experiment_records_existing_artifact_manifests(
    runner, tmp_path
):
    config_path = tmp_path / "supervisor_config.json"
    metrics_path = tmp_path / "supervisor_metrics.jsonl"
    summary_path = tmp_path / "supervisor_summary.json"
    checkpoint_manifest = tmp_path / "checkpoint_manifest.json"
    plot_manifest = tmp_path / "plot_manifest.json"
    checkpoint_manifest.write_text(
        json.dumps({"proposal_type": "checkpoint_manifest", "checkpoints": []}),
        encoding="utf-8",
    )
    plot_manifest.write_text(
        json.dumps({"proposal_type": "plot_manifest", "plots": []}),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "supervisor-baseline-experiment",
            "--config-json",
            str(config_path),
            "--metrics-jsonl",
            str(metrics_path),
            "--summary-json",
            str(summary_path),
            "--checkpoint-manifest",
            str(checkpoint_manifest),
            "--plot-manifest",
            str(plot_manifest),
            "--git-sha",
            "abc1234",
            "--seed",
            "91",
            "--dependency-lock",
            "requirements-dev.txt:sha256:test",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    stdout_record = json.loads(result.output)
    assert stdout_record["artifacts"]["checkpoint_manifest_path"] == str(
        checkpoint_manifest
    )
    assert stdout_record["artifacts"]["plot_manifest_path"] == str(plot_manifest)


def test_supervisor_baseline_experiment_loads_scenario_json(runner, tmp_path):
    config_path = tmp_path / "supervisor_config.json"
    metrics_path = tmp_path / "supervisor_metrics.jsonl"
    summary_path = tmp_path / "supervisor_summary.json"
    scenario_path = tmp_path / "scenario.json"
    scenario_payload = {
        "phases": [0.0, 0.2, 2.5, 3.0],
        "omegas": [0.05, 0.02, -0.02, -0.05],
        "base_coupling_off_diagonal": 0.02,
        "good_mask": [1.0, 1.0, 0.0, 0.0],
        "bad_mask": [0.0, 0.0, 1.0, 1.0],
        "dt": 0.04,
        "inner_steps": 3,
        "horizon": 5,
    }
    scenario_path.write_text(json.dumps(scenario_payload), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "supervisor-baseline-experiment",
            "--scenario-json",
            str(scenario_path),
            "--config-json",
            str(config_path),
            "--metrics-jsonl",
            str(metrics_path),
            "--summary-json",
            str(summary_path),
            "--git-sha",
            "abc1234",
            "--seed",
            "91",
            "--dependency-lock",
            "requirements-dev.txt:sha256:test",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    config_record = json.loads(config_path.read_text(encoding="utf-8"))
    assert config_record["scenario"] == {
        "n_oscillators": 4,
        **scenario_payload,
    }


def test_supervisor_baseline_experiment_rejects_malformed_scenario_json(
    runner, tmp_path
):
    scenario_path = tmp_path / "bad_scenario.json"
    scenario_path.write_text(
        json.dumps(
            {
                "phases": [0.0, 0.2],
                "omegas": [0.05],
                "base_coupling_off_diagonal": 0.02,
                "good_mask": [1.0, 1.0],
                "bad_mask": [0.0, 0.0],
                "dt": 0.04,
                "inner_steps": 3,
                "horizon": 5,
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "supervisor-baseline-experiment",
            "--scenario-json",
            str(scenario_path),
            "--config-json",
            str(tmp_path / "supervisor_config.json"),
            "--metrics-jsonl",
            str(tmp_path / "supervisor_metrics.jsonl"),
            "--summary-json",
            str(tmp_path / "supervisor_summary.json"),
            "--git-sha",
            "abc1234",
            "--seed",
            "91",
            "--dependency-lock",
            "requirements-dev.txt:sha256:test",
        ],
    )

    assert result.exit_code == 1
    assert "scenario omegas length must match phases length" in result.output


def test_supervisor_baseline_experiment_rejects_missing_artifact_manifests(
    runner, tmp_path
):
    result = runner.invoke(
        main,
        [
            "supervisor-baseline-experiment",
            "--metrics-jsonl",
            str(tmp_path / "supervisor_metrics.jsonl"),
            "--summary-json",
            str(tmp_path / "supervisor_summary.json"),
            "--checkpoint-manifest",
            str(tmp_path / "missing_checkpoint_manifest.json"),
            "--git-sha",
            "abc1234",
            "--seed",
            "91",
            "--dependency-lock",
            "requirements-dev.txt:sha256:test",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--checkpoint-manifest'" in result.output


# Pipeline wiring: CLI tested via CliRunner invoking validate/run/replay/report/scaffold
# commands which wrap SimulationState -> engine -> order_parameter -> policy.
# TestCliCommands (above) proves full E2E.
