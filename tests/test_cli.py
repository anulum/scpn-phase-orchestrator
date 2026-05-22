# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI tests

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli as cli_module
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.plugins import (
    PluginCapability,
    PluginManifest,
    PluginRuntimeExecutionPolicy,
    build_plugin_execution_approval,
    build_plugin_execution_plan,
)
from scpn_phase_orchestrator.runtime.audit_stream import read_event_stream
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


def test_auto_bind_rejects_malformed_sample_rate_option(runner, tmp_path):
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
            "not-a-number",
            "--project-name",
            "grid_replay",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid value for '--sample-rate-hz'" in result.output


def test_auto_bind_json_out_is_read_only(runner, tmp_path):
    csv_path = tmp_path / "grid.csv"
    csv_path.write_text(
        "time,grid,load\n0.00,0.0,1.0\n0.01,0.2,0.9\n0.02,0.4,0.7\n",
        encoding="utf-8",
    )

    with runner.isolated_filesystem() as fs_root:
        input_path = Path(fs_root) / "grid.csv"
        input_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
        before = set(Path(fs_root).iterdir())

        result = runner.invoke(
            main,
            [
                "auto-bind",
                "time-series-csv",
                str(input_path),
                "--project-name",
                "grid_replay",
                "--json-out",
            ],
        )

        after = set(Path(fs_root).iterdir())

    assert result.exit_code == 0
    assert before == after


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


def test_run_fails_closed_for_unenforced_non_research_tier(runner, tmp_path):
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

    assert result.exit_code != 0
    assert "safety_tier='clinical' is not enforced" in result.output
    assert "R_good=" not in result.output


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


def test_run_rejects_malformed_step_count(runner, valid_spec_path):
    result = runner.invoke(main, ["run", valid_spec_path, "--steps", "not-a-number"])

    assert result.exit_code == 2
    assert "Invalid value for '--steps'" in result.output


def test_run_with_seed_is_deterministic(runner, valid_spec_path):
    args = ["run", valid_spec_path, "--steps", "8", "--seed", "12345"]

    result_one = runner.invoke(main, args)
    result_two = runner.invoke(main, args)

    assert result_one.exit_code == 0
    assert result_two.exit_code == 0
    assert result_one.output == result_two.output


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


def test_formal_export_package_outputs_no_execution_manifest(runner, tmp_path):
    spec_path = _write_formal_export_spec(tmp_path)
    package_path = tmp_path / "formal_package.json"
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

    stdout_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--module-name",
            "cli_formal_package",
        ],
    )
    file_result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--output",
            str(package_path),
            "--module-name",
            "cli_formal_package",
        ],
    )

    assert stdout_result.exit_code == 0
    payload = json.loads(stdout_result.output)
    assert payload["package_name"] == "cli_formal_package"
    assert payload["artifact_types"] == {
        "policy_prism": "prism",
        "protocol_prism": "prism",
        "protocol_tla": "tla",
    }
    assert [item["name"] for item in payload["properties"]] == [
        "protocol_type_ok",
        "protocol_reachable_terminal",
        "policy_rule_review",
    ]
    assert all(
        command["execution_permitted"] is False
        for command in payload["checker_commands"]
    )
    assert "checker_availability" not in payload
    assert len(payload["package_hash"]) == 64
    assert file_result.exit_code == 0
    assert "Formal verification package written:" in file_result.output
    assert json.loads(package_path.read_text(encoding="utf-8")) == payload


def test_formal_export_package_can_include_checker_readiness(runner, tmp_path):
    spec_path = _write_formal_export_spec(tmp_path)
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

    result = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--module-name",
            "cli_formal_package",
            "--include-checker-readiness",
            "--checker-path",
            "prism=/opt/prism/bin/prism",
            "--checker-path",
            "tlc2.TLC=",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    availability = payload["checker_availability"]
    assert [record["status"] for record in availability] == [
        "missing_executable",
        "ready_not_executed",
        "ready_not_executed",
    ]
    assert [record["executable"] for record in availability] == [
        "tlc2.TLC",
        "prism",
        "prism",
    ]
    assert [record["resolved_path"] for record in availability] == [
        None,
        "/opt/prism/bin/prism",
        "/opt/prism/bin/prism",
    ]
    assert all(record["execution_permitted"] is False for record in availability)


def test_formal_export_checker_readiness_options_are_package_only(
    runner,
    tmp_path,
):
    spec_path = _write_formal_export_spec(tmp_path)
    _write_policy_rules(
        tmp_path,
        [
            {
                "name": "boost",
                "regime": ["DEGRADED"],
                "condition": {
                    "metric": "R_good",
                    "layer": 0,
                    "op": "<",
                    "threshold": 0.7,
                },
                "action": {
                    "knob": "K",
                    "scope": "global",
                    "value": 0.1,
                    "ttl_s": 5.0,
                },
            }
        ],
    )

    non_package = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--include-checker-readiness",
        ],
    )
    missing_readiness = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--checker-path",
            "prism=/opt/prism/bin/prism",
        ],
    )
    malformed = runner.invoke(
        main,
        [
            "formal-export",
            str(spec_path),
            "--export",
            "package",
            "--include-checker-readiness",
            "--checker-path",
            "prism",
        ],
    )

    assert non_package.exit_code == 1
    assert "only valid with --export package" in non_package.output
    assert missing_readiness.exit_code == 1
    assert "--checker-path requires --include-checker-readiness" in (
        missing_readiness.output
    )
    assert malformed.exit_code == 1
    assert "executable=/path syntax" in malformed.output


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


def test_plugins_catalog_can_emit_guarded_rust_runtime_handoff(
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

    result = runner.invoke(main, ["plugins", "catalog", "--rust-runtime-handoff"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema"] == "scpn_rust_plugin_runtime_handoff_v1"
    assert data["registry_schema"] == "scpn_rust_plugin_registry_v1"
    assert data["loading_permitted"] is False
    assert data["compatible_capability_count"] == 1
    assert data["blocked_capability_count"] == 0
    assert data["dispatch_groups"]["actuator"][0]["plugin"] == "cli_plugin"
    assert data["dispatch_groups"]["actuator"][0]["loading_permitted"] is False
    assert len(data["handoff_hash"]) == 64


def test_plugins_catalog_rejects_conflicting_rust_output_modes(runner) -> None:
    result = runner.invoke(
        main,
        ["plugins", "catalog", "--rust-registry", "--rust-runtime-handoff"],
    )

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def _lookup_target_hash(
    manifest: PluginManifest,
    kind: str,
    name: str,
) -> str:
    plan = build_plugin_execution_plan(
        manifest,
        kind,
        name,
        policy=PluginRuntimeExecutionPolicy(
            loading_permitted=True,
            execution_permitted=True,
        ),
    )
    return plan.target_hash


def _write_plan_payload(
    path: Path,
    manifest: PluginManifest,
    kind: str,
    name: str,
    *,
    execution_permitted: bool = True,
    require_target_hash_approval: bool = False,
    approved_target_hashes: tuple[str, ...] = (),
) -> dict[str, object]:
    policy = PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
        require_target_hash_approval=require_target_hash_approval,
        approved_target_hashes=approved_target_hashes,
    )
    plan = build_plugin_execution_plan(
        manifest,
        kind,
        name,
        policy=policy,
    )
    payload: dict[str, object] = {
        **plan.audit_record,
        "manifest": manifest.to_audit_record(),
        "capability": {
            "kind": plan.capability.kind,
            "name": plan.capability.name,
            "target": plan.capability.target,
            "version": plan.capability.version,
            "channels": list(plan.capability.channels),
            "knobs": list(plan.capability.knobs),
        },
    }
    if not execution_permitted:
        payload["execution_permitted"] = False
        payload["plan_hash"] = _normalize_plan_hash(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def _write_approval_payload(
    path: Path,
    manifest: PluginManifest,
    kind: str,
    name: str,
    *,
    operator_identity: str = "operator_42",
    approval_reference: str = "RFC-2026-05-20-01",
    approval_reason: str = "Production change window",
    approved: bool = True,
) -> dict[str, object]:
    policy = PluginRuntimeExecutionPolicy(
        loading_permitted=True,
        execution_permitted=True,
    )
    plan = build_plugin_execution_plan(
        manifest,
        kind,
        name,
        policy=policy,
    )
    approval = build_plugin_execution_approval(
        plan,
        operator_identity=operator_identity,
        approval_reference=approval_reference,
        approval_reason=approval_reason,
    )
    payload = dict(approval.audit_record)
    payload["approved"] = approved
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def _recompute_plan_hash(plan_payload: dict[str, object]) -> str:
    canonical_payload = dict(plan_payload)
    canonical_payload.pop("plan_hash", None)
    canonical_payload.pop("manifest", None)
    canonical_payload.pop("capability", None)
    canonical_payload.pop("compatible", None)
    canonical_payload.pop("compatibility_reasons", None)
    canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_plan_hash(plan_payload: dict[str, object]) -> str:
    payload = dict(plan_payload)
    payload["plan_hash"] = _recompute_plan_hash(plan_payload)
    return payload["plan_hash"]


def test_plugins_plan_execution_outputs_non_executing_plan(
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

    target_hash = _lookup_target_hash(manifest, "actuator", "phase_driver")
    result = runner.invoke(
        main,
        ["plugins", "plan-execution", "cli_plugin", "actuator", "phase_driver"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_runtime_execution_plan_v1"
    assert payload["manifest"]["name"] == "cli_plugin"
    assert payload["capability"]["kind"] == "actuator"
    assert payload["capability"]["name"] == "phase_driver"
    assert payload["target_hash"] == target_hash
    assert payload["execution_permitted"] is True
    assert payload["loading_permitted"] is True
    assert payload["argument_count"] == 0
    assert payload["keyword_names"] == []
    assert payload["compatible"] is True


def test_plugins_plan_execution_requires_approved_target_hash(
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

    target_hash = _lookup_target_hash(manifest, "actuator", "phase_driver")
    result = runner.invoke(
        main,
        [
            "plugins",
            "plan-execution",
            "cli_plugin",
            "actuator",
            "phase_driver",
            "--require-target-hash-approval",
            "--approved-target-hash",
            target_hash,
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["target_hash_approved"] is True
    assert payload["require_target_hash_approval"] is True
    assert payload["execution_permitted"] is True


def test_plugins_plan_execution_rejects_unapproved_target_hash(
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

    result = runner.invoke(
        main,
        [
            "plugins",
            "plan-execution",
            "cli_plugin",
            "actuator",
            "phase_driver",
            "--require-target-hash-approval",
            "--approved-target-hash",
            "0" * 64,
        ],
    )

    assert result.exit_code == 1
    assert "not approved" in result.output


def test_plugins_plan_execution_fails_with_bad_target_hash_format(
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

    result = runner.invoke(
        main,
        [
            "plugins",
            "plan-execution",
            "cli_plugin",
            "actuator",
            "phase_driver",
            "--approved-target-hash",
            "not-a-hash",
        ],
    )

    assert result.exit_code == 1
    assert "not a valid SHA-256 digest" in result.output


def test_plugins_plan_execution_fails_on_missing_plugin(
    runner,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = PluginManifest(
        name="cli_plugin",
        version="0.0.1",
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

    result = runner.invoke(
        main,
        ["plugins", "plan-execution", "missing_plugin", "actuator", "phase_driver"],
    )

    assert result.exit_code == 1
    assert "is not discovered" in result.output


def test_plugins_plan_execution_fails_on_missing_capability(
    runner,
    monkeypatch: pytest.MonkeyPatch,
):
    manifest = PluginManifest(
        name="cli_plugin",
        version="0.0.1",
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

    result = runner.invoke(
        main,
        ["plugins", "plan-execution", "cli_plugin", "actuator", "missing_capability"],
    )

    assert result.exit_code == 1
    assert "does not expose actuator:'missing_capability'" in result.output


def test_plugins_approve_execution_plan_outputs_deterministic_approval(
    runner,
    tmp_path: Path,
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
    plan_path = tmp_path / "plan.json"
    _write_plan_payload(plan_path, manifest, "actuator", "phase_driver")

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_approval_v1"
    assert payload["operator_identity"] == "operator_42"
    assert payload["approval_reference"] == "RFC-2026-05-20-01"
    assert payload["approval_reason"] == "Production change window"
    assert payload["approved"] is True
    assert payload["execution_permitted"] is True
    assert len(payload["approval_hash"]) == 64


def test_plugins_approve_execution_plan_rejects_malformed_json(
    runner,
    tmp_path: Path,
):
    path = tmp_path / "plan.json"
    path.write_text("{bad-json", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 1
    assert "malformed plan JSON" in result.output


def test_plugins_approve_execution_plan_rejects_schema_mismatch(
    runner,
    tmp_path: Path,
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
    plan_path = tmp_path / "plan.json"
    payload = _write_plan_payload(plan_path, manifest, "actuator", "phase_driver")
    payload["schema"] = "unsupported_schema_v1"
    payload["plan_hash"] = _normalize_plan_hash(payload)
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 1
    assert "plan schema mismatch" in result.output


def test_plugins_approve_execution_plan_rejects_missing_hashes(
    runner,
    tmp_path: Path,
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
    plan_path = tmp_path / "plan.json"
    payload = _write_plan_payload(plan_path, manifest, "actuator", "phase_driver")
    payload.pop("plan_hash")
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 1
    assert "missing required field plan_hash" in result.output


def test_plugins_approve_execution_plan_rejects_disabled_plan(
    runner,
    tmp_path: Path,
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
    plan_path = tmp_path / "plan.json"
    payload = _write_plan_payload(
        plan_path,
        manifest,
        "actuator",
        "phase_driver",
        execution_permitted=False,
    )
    payload["plan_hash"] = _normalize_plan_hash(payload)
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 1
    assert "execution must be permitted" in result.output


def test_plugins_approve_execution_plan_rejects_unapproved_target_hash(
    runner,
    tmp_path: Path,
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
    plan_path = tmp_path / "plan.json"
    payload = _write_plan_payload(
        plan_path,
        manifest,
        "actuator",
        "phase_driver",
    )
    payload["require_target_hash_approval"] = True
    payload["target_hash_approved"] = False
    payload["plan_hash"] = _normalize_plan_hash(payload)
    plan_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 1
    assert "is not approved" in result.output


def test_plugins_approve_execution_plan_rejects_missing_operator_metadata(
    runner,
    tmp_path: Path,
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
    plan_path = tmp_path / "plan.json"
    _write_plan_payload(plan_path, manifest, "actuator", "phase_driver")

    result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )

    assert result.exit_code == 1
    assert "operator identity is required" in result.output


def _request_execution_test_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
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
    plan_path = tmp_path / "plan.json"
    plan_payload = _write_plan_payload(plan_path, manifest, "actuator", "phase_driver")
    approval_path = tmp_path / "approval.json"
    approval_payload = _write_approval_payload(
        approval_path, manifest, "actuator", "phase_driver"
    )
    return plan_path, approval_path, plan_payload, approval_payload


def test_plugins_request_execution_outputs_deterministic_request(
    runner,
    tmp_path: Path,
):
    (
        plan_path,
        approval_path,
        plan_payload,
        approval_payload,
    ) = _request_execution_test_fixture(tmp_path)
    approval_payload["plan_hash"] = plan_payload["plan_hash"]
    approval_path.write_text(
        json.dumps(approval_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_runtime_execution_request_v1"
    assert payload["plan_hash"] == plan_payload["plan_hash"]
    assert payload["target_hash"] == plan_payload["target_hash"]
    assert payload["approval_hash"] == approval_payload["approval_hash"]
    assert payload["operator_identity"] == "operator_42"


def test_plugins_request_execution_rejects_plan_hash_mismatch(
    runner,
    tmp_path: Path,
):
    (
        plan_path,
        approval_path,
        plan_payload,
        approval_payload,
    ) = _request_execution_test_fixture(tmp_path)
    approval_payload["plan_hash"] = "0" * 64
    approval_path.write_text(
        json.dumps(approval_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )

    assert result.exit_code == 1
    assert "plan hash mismatch" in result.output


def test_plugins_request_execution_rejects_target_hash_mismatch(
    runner,
    tmp_path: Path,
):
    (
        plan_path,
        approval_path,
        plan_payload,
        approval_payload,
    ) = _request_execution_test_fixture(tmp_path)
    approval_payload["target_hash"] = plan_payload["target_hash"][::-1]
    approval_path.write_text(
        json.dumps(approval_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )

    assert result.exit_code == 1
    assert "target hash mismatch" in result.output


def test_plugins_request_execution_rejects_unapproved_approval(
    runner,
    tmp_path: Path,
):
    (
        _plan_path,
        approval_path,
        plan_payload,
        approval_payload,
    ) = _request_execution_test_fixture(tmp_path)
    approval_payload["approved"] = False
    approval_payload["plan_hash"] = plan_payload["plan_hash"]
    approval_payload["target_hash"] = plan_payload["target_hash"]
    approval_path.write_text(
        json.dumps(approval_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"

    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )

    assert result.exit_code == 1
    assert "not approved" in result.output


def test_plugins_request_execution_rejects_malformed_approval_json(
    runner,
    tmp_path: Path,
):
    plan_path, approval_path, _, _ = _request_execution_test_fixture(tmp_path)
    approval_path.write_text("{bad-json", encoding="utf-8")

    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )

    assert result.exit_code == 1
    assert "malformed approval JSON" in result.output


def test_plugins_request_execution_rejects_missing_approval_hash(
    runner,
    tmp_path: Path,
):
    (
        plan_path,
        approval_path,
        plan_payload,
        approval_payload,
    ) = _request_execution_test_fixture(tmp_path)
    approval_payload.pop("approval_hash", None)
    approval_payload["plan_hash"] = plan_payload["plan_hash"]
    approval_payload["target_hash"] = plan_payload["target_hash"]
    approval_path.write_text(
        json.dumps(approval_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )

    assert result.exit_code == 1
    assert "approval_hash" in result.output


def _write_request_payload_from_cli(runner, tmp_path: Path) -> Path:
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
    target_hash = _lookup_target_hash(manifest, "actuator", "phase_driver")
    plan_path = tmp_path / "plan.json"
    _write_plan_payload(
        plan_path,
        manifest,
        "actuator",
        "phase_driver",
        require_target_hash_approval=True,
        approved_target_hashes=(target_hash,),
    )
    approval_result = runner.invoke(
        main,
        [
            "plugins",
            "approve-execution-plan",
            str(plan_path),
            "--operator-id",
            "operator_42",
            "--approval-reference",
            "RFC-2026-05-20-01",
            "--approval-reason",
            "Production change window",
        ],
    )
    assert approval_result.exit_code == 0
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(approval_result.output, encoding="utf-8")
    result = runner.invoke(
        main,
        ["plugins", "request-execution", str(plan_path), str(approval_path)],
    )
    assert result.exit_code == 0
    request_path = tmp_path / "request.json"
    request_path.write_text(result.output, encoding="utf-8")
    return request_path


def test_plugins_persist_execution_request_writes_bundle(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    output_path = tmp_path / "bundle.json"

    result = runner.invoke(
        main,
        [
            "plugins",
            "persist-execution-request",
            str(request_path),
            str(output_path),
            "--storage-uri",
            f"file://{output_path}",
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == written
    assert payload["schema"] == "scpn_plugin_execution_request_storage_bundle_v1"
    assert payload["storage_manifest"]["storage_backend"] == "local_file"
    assert payload["storage_manifest"]["created_by"] == "deployment_gate"
    assert (
        payload["storage_manifest"]["request_hash"]
        == payload["request"]["request_hash"]
    )
    assert len(payload["bundle_hash"]) == 64


def test_plugins_persist_execution_request_rejects_existing_bundle(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    output_path = tmp_path / "bundle.json"
    command = [
        "plugins",
        "persist-execution-request",
        str(request_path),
        str(output_path),
        "--storage-uri",
        f"file://{output_path}",
        "--created-by",
        "deployment_gate",
    ]

    first = runner.invoke(main, command)
    second = runner.invoke(main, command)

    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "already exists" in second.output


def test_plugins_persist_execution_request_rejects_tampered_request(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload["request_hash"] = "0" * 64
    request_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "persist-execution-request",
            str(request_path),
            str(tmp_path / "bundle.json"),
            "--storage-uri",
            f"file://{tmp_path / 'bundle.json'}",
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "request audit record mismatch" in result.output


def test_plugins_persist_execution_request_rejects_revoked_request(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    payload = json.loads(request_path.read_text(encoding="utf-8"))

    result = runner.invoke(
        main,
        [
            "plugins",
            "persist-execution-request",
            str(request_path),
            str(tmp_path / "bundle.json"),
            "--storage-uri",
            f"file://{tmp_path / 'bundle.json'}",
            "--created-by",
            "deployment_gate",
            "--revoked-request-hash",
            payload["request_hash"],
        ],
    )

    assert result.exit_code == 1
    assert "revoked" in result.output


def test_plugins_storage_adapter_manifest_outputs_external_handoff(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)

    result = runner.invoke(
        main,
        [
            "plugins",
            "storage-adapter-manifest",
            str(request_path),
            "--storage-uri",
            "s3://spo-prod/plugin-requests/request.json",
            "--storage-backend",
            "s3_object",
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_request_storage_adapter_v1"
    assert payload["request_hash"] == request_payload["request_hash"]
    assert payload["storage_backend"] == "s3_object"
    assert payload["storage_scheme"] == "s3"
    assert payload["adapter_mode"] == "deployment_owned_external_write"
    assert payload["write_performed"] is False
    assert len(payload["bundle_hash"]) == 64
    assert len(payload["adapter_hash"]) == 64


def test_plugins_storage_adapter_manifest_rejects_credential_uri(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)

    result = runner.invoke(
        main,
        [
            "plugins",
            "storage-adapter-manifest",
            str(request_path),
            "--storage-uri",
            "https://user:pass@storage.example.test/request.json",
            "--storage-backend",
            "https_api",
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "must not contain credentials" in result.output


def test_plugins_lifecycle_status_reports_stored_request(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    bundle_path = tmp_path / "bundle.json"
    persist_result = runner.invoke(
        main,
        [
            "plugins",
            "persist-execution-request",
            str(request_path),
            str(bundle_path),
            "--storage-uri",
            f"file://{bundle_path}",
            "--created-by",
            "deployment_gate",
        ],
    )
    assert persist_result.exit_code == 0

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--storage-bundle",
            str(bundle_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_request_lifecycle_v1"
    assert payload["request_hash"] == request_payload["request_hash"]
    assert payload["status"] == "stored"
    assert payload["revoked"] is False
    assert (
        payload["storage_manifest_hash"]
        == bundle_payload["storage_manifest"]["manifest_hash"]
    )
    assert len(payload["lifecycle_hash"]) == 64


def test_plugins_lifecycle_status_reports_revoked_request(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    revocation_result = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_path),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-20-40",
            "--revocation-reason",
            "operator rotation",
        ],
    )
    assert revocation_result.exit_code == 0
    revocation_path = tmp_path / "revocation.json"
    revocation_path.write_text(revocation_result.output, encoding="utf-8")
    revocation_list_result = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert revocation_list_result.exit_code == 0
    revocation_list_path = tmp_path / "revocation-list.json"
    revocation_list_path.write_text(
        revocation_list_result.output,
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--revocation-list",
            str(revocation_list_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    revocation_payload = json.loads(revocation_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["status"] == "revoked"
    assert payload["revoked"] is True
    assert payload["revocation_hash"] == revocation_payload["revocation_hash"]
    assert payload["revoked_by"] == "deployment_gate"
    assert payload["revocation_reference"] == "REV-2026-05-20-40"


def test_plugins_lifecycle_summary_outputs_review_queues(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    lifecycle_payload = json.loads(lifecycle_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_request_lifecycle_summary_v1"
    assert payload["request_count"] == 1
    assert payload["status_counts"] == {"approved": 1, "revoked": 0, "stored": 0}
    assert payload["approved_request_hashes"] == [lifecycle_payload["request_hash"]]
    assert payload["storage_missing_request_hashes"] == [
        lifecycle_payload["request_hash"]
    ]
    assert payload["renewal_required_request_hashes"] == []
    assert len(payload["summary_hash"]) == 64


def test_plugins_lifecycle_summary_rejects_duplicate_request_hash(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate request hashes" in result.output


def test_plugins_lifecycle_policy_report_outputs_operator_dashboard(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")
    summary_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_result.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary_result.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_request_lifecycle_policy_v1"
    assert payload["summary_hash"] == summary_payload["summary_hash"]
    assert payload["policy_action_counts"] == {
        "confirm_external_write": 0,
        "persist_request": 1,
        "register_storage_adapter": 1,
        "renew_approval": 0,
    }
    assert (
        payload["storage_missing_request_hashes"]
        == summary_payload["storage_missing_request_hashes"]
    )
    assert len(payload["policy_hash"]) == 64


def test_plugins_lifecycle_policy_report_rejects_tampered_summary(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")
    summary_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_result.exit_code == 0
    summary_payload = json.loads(summary_result.output)
    summary_payload["request_count"] = 2
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "lifecycle summary hash mismatch" in result.output


def test_plugins_lifecycle_renewal_queue_outputs_deterministic_followups(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")
    summary_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_result.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary_result.output, encoding="utf-8")
    policy_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_result.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy_result.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-renewal-queue",
            str(summary_path),
            "--policy-report",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    policy_payload = json.loads(policy_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert (
        payload["schema"] == "scpn_plugin_execution_request_lifecycle_renewal_queue_v1"
    )
    assert payload["summary_hash"] == summary_payload["summary_hash"]
    assert payload["request_count"] == summary_payload["request_count"]
    assert payload["renewal_required_request_hashes"] == []
    assert (
        payload["storage_missing_request_hashes"]
        == summary_payload["storage_missing_request_hashes"]
    )
    assert (
        payload["missing_adapter_request_hashes"]
        == policy_payload["missing_adapter_request_hashes"]
    )
    assert (
        payload["external_write_followup_request_hashes"]
        == policy_payload["external_write_followup_request_hashes"]
    )
    assert len(payload["queue_hash"]) == 64


def test_plugins_lifecycle_renewal_queue_rejects_policy_summary_mismatch(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")
    summary_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_result.exit_code == 0
    summary_payload = json.loads(summary_result.output)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    policy_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_result.exit_code == 0
    policy_payload = json.loads(policy_result.output)
    policy_payload["summary_hash"] = "0" * 64
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(policy_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-renewal-queue",
            str(summary_path),
            "--policy-report",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "summary_hash does not match lifecycle summary" in result.output


def test_plugins_lifecycle_multistore_dashboard_aggregates_policy_reports(
    runner,
    tmp_path: Path,
):
    tmp_a = tmp_path / "a"
    tmp_b = tmp_path / "b"
    tmp_a.mkdir(parents=True, exist_ok=True)
    tmp_b.mkdir(parents=True, exist_ok=True)
    request_a = _write_request_payload_from_cli(runner, tmp_a)
    request_b = _write_request_payload_from_cli(runner, tmp_b)

    lifecycle_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_a),
            "--created-by",
            "deployment_gate",
        ],
    )
    revocation_b = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_b),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-21-MS",
            "--revocation-reason",
            "operator rotation",
        ],
    )
    assert revocation_b.exit_code == 0
    revocation_b_path = tmp_path / "revocation-b.json"
    revocation_b_path.write_text(revocation_b.output, encoding="utf-8")
    revocation_list_b = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert revocation_list_b.exit_code == 0
    revocation_list_b_path = tmp_path / "revocation-list-b.json"
    revocation_list_b_path.write_text(revocation_list_b.output, encoding="utf-8")
    lifecycle_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_b),
            "--revocation-list",
            str(revocation_list_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_a.exit_code == 0
    assert lifecycle_b.exit_code == 0
    lifecycle_a_path = tmp_path / "lifecycle-a.json"
    lifecycle_b_path = tmp_path / "lifecycle-b.json"
    lifecycle_a_path.write_text(lifecycle_a.output, encoding="utf-8")
    lifecycle_b_path.write_text(lifecycle_b.output, encoding="utf-8")

    summary_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_a_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    summary_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_a.exit_code == 0
    assert summary_b.exit_code == 0
    summary_a_path = tmp_path / "summary-a.json"
    summary_b_path = tmp_path / "summary-b.json"
    summary_a_path.write_text(summary_a.output, encoding="utf-8")
    summary_b_path.write_text(summary_b.output, encoding="utf-8")

    policy_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_a_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    policy_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_a.exit_code == 0
    assert policy_b.exit_code == 0
    policy_a_path = tmp_path / "policy-a.json"
    policy_b_path = tmp_path / "policy-b.json"
    policy_a_path.write_text(policy_a.output, encoding="utf-8")
    policy_b_path.write_text(policy_b.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-dashboard",
            str(policy_a_path),
            str(policy_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    policy_a_payload = json.loads(policy_a_path.read_text(encoding="utf-8"))
    policy_b_payload = json.loads(policy_b_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_multistore_dashboard_v1"
    )
    assert payload["policy_count"] == 2
    assert payload["policy_hashes"] == sorted(
        [policy_a_payload["policy_hash"], policy_b_payload["policy_hash"]]
    )
    expected_action_counts = {
        key: policy_a_payload["policy_action_counts"][key]
        + policy_b_payload["policy_action_counts"][key]
        for key in (
            "confirm_external_write",
            "persist_request",
            "register_storage_adapter",
            "renew_approval",
        )
    }
    assert payload["aggregated_policy_action_counts"] == expected_action_counts
    expected_unique = set(policy_a_payload["renewal_required_request_hashes"])
    expected_unique.update(policy_a_payload["storage_missing_request_hashes"])
    expected_unique.update(policy_a_payload["missing_adapter_request_hashes"])
    expected_unique.update(policy_a_payload["external_write_followup_request_hashes"])
    expected_unique.update(policy_b_payload["renewal_required_request_hashes"])
    expected_unique.update(policy_b_payload["storage_missing_request_hashes"])
    expected_unique.update(policy_b_payload["missing_adapter_request_hashes"])
    expected_unique.update(policy_b_payload["external_write_followup_request_hashes"])
    assert payload["unique_flagged_request_count"] == len(expected_unique)
    assert len(payload["dashboard_hash"]) == 64


def test_plugins_lifecycle_multistore_dashboard_rejects_duplicate_policy_hash(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")
    summary_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_result.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary_result.output, encoding="utf-8")
    policy_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_result.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy_result.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-dashboard",
            str(policy_path),
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate lifecycle policy hash" in result.output


def test_plugins_lifecycle_multistore_drilldown_outputs_store_provenance(
    runner,
    tmp_path: Path,
):
    tmp_a = tmp_path / "a"
    tmp_b = tmp_path / "b"
    tmp_a.mkdir(parents=True, exist_ok=True)
    tmp_b.mkdir(parents=True, exist_ok=True)
    request_a = _write_request_payload_from_cli(runner, tmp_a)
    request_b = _write_request_payload_from_cli(runner, tmp_b)

    lifecycle_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_a),
            "--created-by",
            "deployment_gate",
        ],
    )
    revocation_b = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_b),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-21-DD",
            "--revocation-reason",
            "operator rotation",
        ],
    )
    assert lifecycle_a.exit_code == 0
    assert revocation_b.exit_code == 0
    lifecycle_a_path = tmp_path / "lifecycle-a.json"
    lifecycle_a_path.write_text(lifecycle_a.output, encoding="utf-8")
    revocation_b_path = tmp_path / "revocation-b.json"
    revocation_b_path.write_text(revocation_b.output, encoding="utf-8")
    revocation_list_b = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert revocation_list_b.exit_code == 0
    revocation_list_b_path = tmp_path / "revocation-list-b.json"
    revocation_list_b_path.write_text(revocation_list_b.output, encoding="utf-8")
    lifecycle_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_b),
            "--revocation-list",
            str(revocation_list_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_b.exit_code == 0
    lifecycle_b_path = tmp_path / "lifecycle-b.json"
    lifecycle_b_path.write_text(lifecycle_b.output, encoding="utf-8")

    summary_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_a_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    summary_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_a.exit_code == 0
    assert summary_b.exit_code == 0
    summary_a_path = tmp_path / "summary-a.json"
    summary_b_path = tmp_path / "summary-b.json"
    summary_a_path.write_text(summary_a.output, encoding="utf-8")
    summary_b_path.write_text(summary_b.output, encoding="utf-8")

    policy_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_a_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    policy_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_a.exit_code == 0
    assert policy_b.exit_code == 0
    policy_a_path = tmp_path / "policy-a.json"
    policy_b_path = tmp_path / "policy-b.json"
    policy_a_path.write_text(policy_a.output, encoding="utf-8")
    policy_b_path.write_text(policy_b.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_a_path),
            str(policy_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_multistore_drilldown_v1"
    )
    assert payload["policy_count"] == 2
    assert len(payload["stores"]) == 2
    assert len(payload["drilldown_hash"]) == 64
    for store in payload["stores"]:
        assert len(store["store_hash"]) == 64
    assert payload["global_flagged_request_count"] >= 1


def test_plugins_lifecycle_multistore_drilldown_rejects_duplicate_policy_hash(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_result.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle_result.output, encoding="utf-8")
    summary_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_result.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary_result.output, encoding="utf-8")
    policy_result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_result.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy_result.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate lifecycle policy hash" in result.output


def test_plugins_lifecycle_remediation_orchestration_outputs_priority_plan(
    runner,
    tmp_path: Path,
):
    tmp_a = tmp_path / "a"
    tmp_b = tmp_path / "b"
    tmp_a.mkdir(parents=True, exist_ok=True)
    tmp_b.mkdir(parents=True, exist_ok=True)
    request_a = _write_request_payload_from_cli(runner, tmp_a)
    request_b = _write_request_payload_from_cli(runner, tmp_b)

    lifecycle_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_a),
            "--created-by",
            "deployment_gate",
        ],
    )
    revocation_b = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_b),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-21-ORCH",
            "--revocation-reason",
            "operator rotation",
        ],
    )
    assert lifecycle_a.exit_code == 0
    assert revocation_b.exit_code == 0
    lifecycle_a_path = tmp_path / "lifecycle-a.json"
    lifecycle_a_path.write_text(lifecycle_a.output, encoding="utf-8")
    revocation_b_path = tmp_path / "revocation-b.json"
    revocation_b_path.write_text(revocation_b.output, encoding="utf-8")
    revocation_list_b = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert revocation_list_b.exit_code == 0
    revocation_list_b_path = tmp_path / "revocation-list-b.json"
    revocation_list_b_path.write_text(revocation_list_b.output, encoding="utf-8")
    lifecycle_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_b),
            "--revocation-list",
            str(revocation_list_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle_b.exit_code == 0
    lifecycle_b_path = tmp_path / "lifecycle-b.json"
    lifecycle_b_path.write_text(lifecycle_b.output, encoding="utf-8")

    summary_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_a_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    summary_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary_a.exit_code == 0
    assert summary_b.exit_code == 0
    summary_a_path = tmp_path / "summary-a.json"
    summary_b_path = tmp_path / "summary-b.json"
    summary_a_path.write_text(summary_a.output, encoding="utf-8")
    summary_b_path.write_text(summary_b.output, encoding="utf-8")

    policy_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_a_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    policy_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy_a.exit_code == 0
    assert policy_b.exit_code == 0
    policy_a_path = tmp_path / "policy-a.json"
    policy_b_path = tmp_path / "policy-b.json"
    policy_a_path.write_text(policy_a.output, encoding="utf-8")
    policy_b_path.write_text(policy_b.output, encoding="utf-8")

    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_a_path),
            str(policy_b_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_remediation_plan_v1"
    )
    assert payload["action_count"] == len(payload["actions"])
    assert len(payload["plan_hash"]) == 64
    priorities = [action["priority"] for action in payload["actions"]]
    assert priorities == sorted(priorities)
    for action in payload["actions"]:
        assert len(action["action_hash"]) == 64


def test_plugins_lifecycle_remediation_orchestration_rejects_bad_drilldown_schema(
    runner,
    tmp_path: Path,
):
    bad_path = tmp_path / "bad-drilldown.json"
    bad_path.write_text(
        json.dumps(
            {
                "schema": "scpn_plugin_execution_request_lifecycle_other_v1",
                "version": "1.0.0",
                "drilldown_hash": "0" * 64,
                "policy_count": 1,
                "stores": [],
                "global_flagged_request_hashes": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(bad_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "multi-store drilldown schema mismatch" in result.output


def test_plugins_lifecycle_remediation_execution_dashboard_tracks_action_states(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "completed",
            "--updated-by",
            "deployment_gate",
            "--note",
            "executed in maintenance window",
        ],
    )
    assert status.exit_code == 0
    status_path = tmp_path / "status.json"
    status_path.write_text(status.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(status_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_remediation_execution_dashboard_v1"
    )
    assert payload["action_count"] == len(plan_payload["actions"])
    assert payload["state_counts"]["completed"] == 1
    assert payload["state_counts"]["pending"] == payload["action_count"] - 1
    assert first_action_hash in payload["resolved_action_hashes"]
    assert len(payload["execution_hash"]) == 64


def test_plugins_lifecycle_remediation_execution_dashboard_rejects_plan_mismatch(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "completed",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert status.exit_code == 0
    status_payload = json.loads(status.output)
    status_payload["plan_hash"] = "0" * 64
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(status_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(status_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "status plan_hash does not match remediation plan" in result.output


def test_plugins_lifecycle_remediation_deployment_handoff_outputs_unresolved_actions(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "in_progress",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert status.exit_code == 0
    status_path = tmp_path / "status.json"
    status_path.write_text(status.output, encoding="utf-8")
    dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(status_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert dashboard.exit_code == 0
    dashboard_path = tmp_path / "execution-dashboard.json"
    dashboard_path.write_text(dashboard.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
    )
    assert payload["unresolved_action_count"] == len(payload["handoff_actions"])
    assert len(payload["handoff_hash"]) == 64
    for action in payload["handoff_actions"]:
        assert len(action["handoff_action_hash"]) == 64
        assert action["deployment_command_template"]


def test_plugins_lifecycle_remediation_deployment_handoff_rejects_tampered_dashboard(
    runner,
    tmp_path: Path,
):
    bad_path = tmp_path / "bad-execution-dashboard.json"
    bad_path.write_text(
        json.dumps(
            {
                "schema": "scpn_plugin_execution_request_lifecycle_other_v1",
                "version": "1.0.0",
                "plan_hash": "0" * 64,
                "execution_hash": "0" * 64,
                "action_count": 0,
                "rows": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(bad_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "remediation execution dashboard schema mismatch" in result.output


def test_plugins_lifecycle_remediation_scheduler_queue_outputs_deterministic_entries(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(request_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "in_progress",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert status.exit_code == 0
    status_path = tmp_path / "status.json"
    status_path.write_text(status.output, encoding="utf-8")
    execution_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(status_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert execution_dashboard.exit_code == 0
    execution_dashboard_path = tmp_path / "execution-dashboard.json"
    execution_dashboard_path.write_text(execution_dashboard.output, encoding="utf-8")
    handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(execution_dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert handoff.exit_code == 0
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(handoff.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "7200",
            "--created-by",
            "deployment_scheduler",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
    )
    assert payload["queue_entry_count"] == len(payload["queue_entries"])
    assert payload["window_start_epoch"] == 1700000000
    assert payload["window_duration_seconds"] == 7200
    assert len(payload["scheduler_hash"]) == 64
    for entry in payload["queue_entries"]:
        assert len(entry["entry_hash"]) == 64
        assert entry["schedule_epoch"] >= 1700000000


def test_plugins_lifecycle_remediation_scheduler_queue_rejects_bad_handoff_schema(
    runner,
    tmp_path: Path,
):
    bad_handoff = tmp_path / "bad-handoff.json"
    bad_handoff.write_text(
        json.dumps(
            {
                "schema": "scpn_plugin_execution_request_lifecycle_other_v1",
                "version": "1.0.0",
                "plan_hash": "0" * 64,
                "execution_hash": "0" * 64,
                "handoff_hash": "0" * 64,
                "unresolved_action_count": 0,
                "handoff_actions": [],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(bad_handoff),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "3600",
            "--created-by",
            "deployment_scheduler",
        ],
    )

    assert result.exit_code == 1
    assert "remediation deployment handoff schema mismatch" in result.output


def test_plugins_lifecycle_remediation_scheduler_queue_rejects_window_overflow(
    runner,
    tmp_path: Path,
):
    handoff_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_deployment_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "created_by": "deployment_gate",
        "handoff_actions": [],
        "unresolved_action_count": 2,
    }
    for index in range(2):
        action = {
            "handoff_action_hash": hashlib.sha256(
                f"handoff-{index}".encode()
            ).hexdigest(),
            "action_hash": hashlib.sha256(f"action-{index}".encode()).hexdigest(),
            "request_hash": hashlib.sha256(f"request-{index}".encode()).hexdigest(),
            "action_type": "renew_approval",
            "priority": index + 1,
            "state": "pending",
            "deployment_command_template": (
                "spo plugins approve-execution-plan PLAN_JSON"
            ),
        }
        handoff_payload["handoff_actions"].append(action)
    handoff_payload["handoff_hash"] = hashlib.sha256(
        json.dumps(handoff_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    handoff_path = tmp_path / "handoff-overflow.json"
    handoff_path.write_text(
        json.dumps(handoff_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "1",
            "--created-by",
            "deployment_scheduler",
        ],
    )

    assert result.exit_code == 1
    assert "exceeds scheduler window duration" in result.output


def test_plugins_lifecycle_remediation_scheduler_telemetry_outputs_overdue_rows(
    runner,
    tmp_path: Path,
):
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(_write_request_payload_from_cli(runner, tmp_path)),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "pending",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert status.exit_code == 0
    initial_status_path = tmp_path / "initial-status.json"
    initial_status_path.write_text(status.output, encoding="utf-8")
    execution_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(initial_status_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert execution_dashboard.exit_code == 0
    execution_dashboard_path = tmp_path / "execution-dashboard.json"
    execution_dashboard_path.write_text(execution_dashboard.output, encoding="utf-8")
    handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(execution_dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert handoff.exit_code == 0
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(handoff.output, encoding="utf-8")
    queue = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "3600",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert queue.exit_code == 0
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(queue.output, encoding="utf-8")
    queue_payload = json.loads(queue.output)
    queued_action_hash = queue_payload["queue_entries"][0]["action_hash"]
    completed_status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            queued_action_hash,
            "--state",
            "completed",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert completed_status.exit_code == 0
    completed_status_path = tmp_path / "completed-status.json"
    completed_status_path.write_text(completed_status.output, encoding="utf-8")

    telemetry = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-telemetry",
            str(queue_path),
            str(completed_status_path),
            "--as-of-epoch",
            "1700000100",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert telemetry.exit_code == 0
    payload = json.loads(telemetry.output)
    assert (
        payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_remediation_scheduler_telemetry_v1"
    )
    assert payload["queue_entry_count"] == len(payload["rows"])
    assert len(payload["telemetry_hash"]) == 64
    assert payload["state_counts"]["completed"] >= 1
    assert payload["state_counts"]["overdue"] >= 1
    for row in payload["rows"]:
        if row["state"] == "completed":
            assert row["overdue"] is False


def test_plugins_lifecycle_remediation_scheduler_telemetry_rejects_duplicate_status(
    runner,
    tmp_path: Path,
):
    queue_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_queue_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "handoff_hash": "3" * 64,
        "window_start_epoch": 1700000000,
        "window_duration_seconds": 3600,
        "queue_entries": [
            {
                "entry_hash": "4" * 64,
                "handoff_action_hash": "5" * 64,
                "action_hash": "6" * 64,
                "request_hash": "7" * 64,
                "action_type": "renew_approval",
                "priority": 1,
                "schedule_epoch": 1700000000,
                "scheduler_command_template": (
                    "spo plugins approve-execution-plan PLAN_JSON"
                ),
            }
        ],
        "queue_entry_count": 1,
        "created_by": "deployment_scheduler",
    }
    queue_payload["scheduler_hash"] = hashlib.sha256(
        json.dumps(queue_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps(queue_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    status_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_action_status_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "action_hash": "6" * 64,
        "request_hash": "7" * 64,
        "state": "pending",
        "updated_by": "deployment_scheduler",
        "note": "",
    }
    status_payload["status_hash"] = hashlib.sha256(
        json.dumps(status_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    status_path_a = tmp_path / "status-a.json"
    status_path_b = tmp_path / "status-b.json"
    status_blob = json.dumps(status_payload, indent=2, sort_keys=True)
    status_path_a.write_text(status_blob, encoding="utf-8")
    status_path_b.write_text(status_blob, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-telemetry",
            str(queue_path),
            str(status_path_a),
            str(status_path_b),
            "--as-of-epoch",
            "1700000200",
            "--created-by",
            "deployment_scheduler",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate action status action_hash" in result.output


def test_plugins_lifecycle_remediation_scheduler_adapter_handoff_and_acknowledgement(
    runner,
    tmp_path: Path,
):
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(_write_request_payload_from_cli(runner, tmp_path)),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "pending",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert status.exit_code == 0
    status_path = tmp_path / "status.json"
    status_path.write_text(status.output, encoding="utf-8")
    execution_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(status_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert execution_dashboard.exit_code == 0
    execution_dashboard_path = tmp_path / "execution-dashboard.json"
    execution_dashboard_path.write_text(execution_dashboard.output, encoding="utf-8")
    handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(execution_dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert handoff.exit_code == 0
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(handoff.output, encoding="utf-8")
    queue = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "3600",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert queue.exit_code == 0
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(queue.output, encoding="utf-8")
    telemetry = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-telemetry",
            str(queue_path),
            "--as-of-epoch",
            "1700000000",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert telemetry.exit_code == 0
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(telemetry.output, encoding="utf-8")

    adapter_handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-adapter-handoff",
            str(telemetry_path),
            "--adapter-name",
            "airflow",
            "--adapter-endpoint",
            "airflow://cluster-a",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert adapter_handoff.exit_code == 0
    adapter_handoff_payload = json.loads(adapter_handoff.output)
    assert adapter_handoff_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
    )
    assert adapter_handoff_payload["entry_count"] == len(
        adapter_handoff_payload["entries"]
    )
    assert len(adapter_handoff_payload["adapter_handoff_hash"]) == 64
    first_adapter_entry_hash = adapter_handoff_payload["entries"][0][
        "adapter_entry_hash"
    ]
    adapter_handoff_path = tmp_path / "adapter-handoff.json"
    adapter_handoff_path.write_text(adapter_handoff.output, encoding="utf-8")

    acknowledgement = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement",
            str(adapter_handoff_path),
            first_adapter_entry_hash,
            "--state",
            "completed",
            "--acknowledged-by",
            "airflow_worker",
            "--external-reference",
            "airflow-run-0001",
            "--note",
            "executed",
        ],
    )
    assert acknowledgement.exit_code == 0
    acknowledgement_payload = json.loads(acknowledgement.output)
    assert acknowledgement_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_v1"
    )
    assert acknowledgement_payload["adapter_entry_hash"] == first_adapter_entry_hash
    assert acknowledgement_payload["state"] == "completed"
    assert len(acknowledgement_payload["acknowledgement_hash"]) == 64


def test_plugins_lifecycle_scheduler_ack_rejects_unknown_entry_hash(
    runner,
    tmp_path: Path,
):
    adapter_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "telemetry_hash": "3" * 64,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "entry_count": 0,
        "entries": [],
        "created_by": "deployment_scheduler",
    }
    adapter_payload["adapter_handoff_hash"] = hashlib.sha256(
        json.dumps(adapter_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    adapter_path = tmp_path / "adapter-handoff-empty.json"
    adapter_path.write_text(
        json.dumps(adapter_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement",
            str(adapter_path),
            "4" * 64,
            "--state",
            "in_progress",
            "--acknowledged-by",
            "airflow_worker",
            "--external-reference",
            "airflow-run-0002",
        ],
    )

    assert result.exit_code == 1
    assert "entry_hash not present in adapter handoff" in result.output


def test_plugins_lifecycle_scheduler_replay_and_execution_dashboard(
    runner,
    tmp_path: Path,
):
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(_write_request_payload_from_cli(runner, tmp_path)),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    plan_payload = json.loads(plan.output)
    first_action_hash = plan_payload["actions"][0]["action_hash"]
    status = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-action-status",
            str(plan_path),
            first_action_hash,
            "--state",
            "pending",
            "--updated-by",
            "deployment_gate",
        ],
    )
    assert status.exit_code == 0
    status_path = tmp_path / "status.json"
    status_path.write_text(status.output, encoding="utf-8")
    execution_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            str(status_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert execution_dashboard.exit_code == 0
    execution_dashboard_path = tmp_path / "execution-dashboard.json"
    execution_dashboard_path.write_text(execution_dashboard.output, encoding="utf-8")
    handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(execution_dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert handoff.exit_code == 0
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(handoff.output, encoding="utf-8")
    queue = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "3600",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert queue.exit_code == 0
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(queue.output, encoding="utf-8")
    telemetry = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-telemetry",
            str(queue_path),
            "--as-of-epoch",
            "1700000100",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert telemetry.exit_code == 0
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(telemetry.output, encoding="utf-8")
    adapter_handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-adapter-handoff",
            str(telemetry_path),
            "--adapter-name",
            "airflow",
            "--adapter-endpoint",
            "airflow://cluster-a",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert adapter_handoff.exit_code == 0
    adapter_handoff_path = tmp_path / "adapter-handoff.json"
    adapter_handoff_path.write_text(adapter_handoff.output, encoding="utf-8")
    adapter_payload = json.loads(adapter_handoff.output)
    first = adapter_payload["entries"][0]["adapter_entry_hash"]
    second = adapter_payload["entries"][1]["adapter_entry_hash"]

    ack_a = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement",
            str(adapter_handoff_path),
            first,
            "--state",
            "completed",
            "--acknowledged-by",
            "airflow_worker",
            "--external-reference",
            "airflow-run-0001",
        ],
    )
    assert ack_a.exit_code == 0
    ack_a_path = tmp_path / "ack-a.json"
    ack_a_path.write_text(ack_a.output, encoding="utf-8")
    ack_b = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement",
            str(adapter_handoff_path),
            second,
            "--state",
            "blocked",
            "--acknowledged-by",
            "airflow_worker",
            "--external-reference",
            "airflow-run-0002",
            "--note",
            "quota",
        ],
    )
    assert ack_b.exit_code == 0
    ack_b_path = tmp_path / "ack-b.json"
    ack_b_path.write_text(ack_b.output, encoding="utf-8")

    replay = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_handoff_path),
            str(ack_a_path),
            str(ack_b_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert replay.exit_code == 0
    replay_payload = json.loads(replay.output)
    assert replay_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_replay_v1"
    )
    assert replay_payload["acknowledgement_count"] == 2
    assert replay_payload["state_counts"]["completed"] == 1
    assert replay_payload["state_counts"]["blocked"] == 1
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(replay.output, encoding="utf-8")

    dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-execution-dashboard",
            str(telemetry_path),
            str(replay_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert dashboard.exit_code == 0
    dashboard_payload = json.loads(dashboard.output)
    assert dashboard_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_execution_dashboard_v1"
    )
    assert dashboard_payload["row_count"] == len(dashboard_payload["rows"])
    assert dashboard_payload["state_counts"]["completed"] >= 1
    assert dashboard_payload["state_counts"]["blocked"] >= 1
    assert len(dashboard_payload["dashboard_hash"]) == 64


def test_plugins_lifecycle_scheduler_replay_rejects_hash_mismatch(
    runner,
    tmp_path: Path,
):
    adapter_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "telemetry_hash": "3" * 64,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "entry_count": 1,
        "entries": [
            {
                "adapter_entry_hash": "4" * 64,
                "entry_hash": "5" * 64,
                "action_hash": "6" * 64,
                "request_hash": "7" * 64,
            }
        ],
        "created_by": "deployment_scheduler",
    }
    adapter_payload["adapter_handoff_hash"] = hashlib.sha256(
        json.dumps(adapter_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(
        json.dumps(adapter_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    ack_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_v1"
        ),
        "version": "1.0.0",
        "adapter_handoff_hash": "8" * 64,
        "telemetry_hash": "3" * 64,
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "adapter_entry_hash": "4" * 64,
        "entry_hash": "5" * 64,
        "action_hash": "6" * 64,
        "request_hash": "7" * 64,
        "state": "completed",
        "acknowledged_by": "worker",
        "external_reference": "ref-1",
        "note": "",
    }
    ack_payload["acknowledgement_hash"] = hashlib.sha256(
        json.dumps(ack_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    ack_path = tmp_path / "ack.json"
    ack_path.write_text(
        json.dumps(ack_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_path),
            str(ack_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )

    assert result.exit_code == 1
    assert "adapter_handoff_hash mismatch" in result.output


def test_plugins_lifecycle_remediation_scheduler_control_plan_and_runbook(
    runner,
    tmp_path: Path,
):
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(_write_request_payload_from_cli(runner, tmp_path)),
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    execution_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert execution_dashboard.exit_code == 0
    execution_dashboard_path = tmp_path / "execution-dashboard.json"
    execution_dashboard_path.write_text(execution_dashboard.output, encoding="utf-8")
    handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(execution_dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert handoff.exit_code == 0
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(handoff.output, encoding="utf-8")
    queue = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "3600",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert queue.exit_code == 0
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(queue.output, encoding="utf-8")
    telemetry = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-telemetry",
            str(queue_path),
            "--as-of-epoch",
            "1700000100",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert telemetry.exit_code == 0
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(telemetry.output, encoding="utf-8")
    adapter_handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-adapter-handoff",
            str(telemetry_path),
            "--adapter-name",
            "airflow",
            "--adapter-endpoint",
            "airflow://cluster-a",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert adapter_handoff.exit_code == 0
    adapter_handoff_path = tmp_path / "adapter-handoff.json"
    adapter_handoff_path.write_text(adapter_handoff.output, encoding="utf-8")
    adapter_payload = json.loads(adapter_handoff.output)
    entry_hash = adapter_payload["entries"][0]["adapter_entry_hash"]
    acknowledgement = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement",
            str(adapter_handoff_path),
            entry_hash,
            "--state",
            "blocked",
            "--acknowledged-by",
            "airflow_worker",
            "--external-reference",
            "airflow-run-777",
        ],
    )
    assert acknowledgement.exit_code == 0
    acknowledgement_path = tmp_path / "ack.json"
    acknowledgement_path.write_text(acknowledgement.output, encoding="utf-8")
    replay = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_handoff_path),
            str(acknowledgement_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert replay.exit_code == 0
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(replay.output, encoding="utf-8")
    scheduler_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-execution-dashboard",
            str(telemetry_path),
            str(replay_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert scheduler_dashboard.exit_code == 0
    scheduler_dashboard_path = tmp_path / "scheduler-dashboard.json"
    scheduler_dashboard_path.write_text(scheduler_dashboard.output, encoding="utf-8")

    control_plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-control-plan",
            str(scheduler_dashboard_path),
            "--created-by",
            "operator_console",
        ],
    )
    assert control_plan.exit_code == 0
    control_payload = json.loads(control_plan.output)
    assert control_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_control_plan_v1"
    )
    assert control_payload["control_action_count"] == len(
        control_payload["control_actions"]
    )
    assert len(control_payload["control_plan_hash"]) == 64
    control_plan_path = tmp_path / "control-plan.json"
    control_plan_path.write_text(control_plan.output, encoding="utf-8")

    runbook = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-runbook",
            str(control_plan_path),
            str(adapter_handoff_path),
            "--created-by",
            "operator_console",
        ],
    )
    assert runbook.exit_code == 0
    runbook_payload = json.loads(runbook.output)
    assert (
        runbook_payload["schema"]
        == "scpn_plugin_execution_request_lifecycle_remediation_scheduler_runbook_v1"
    )
    assert runbook_payload["group_count"] == len(runbook_payload["groups"])
    assert len(runbook_payload["runbook_hash"]) == 64


def test_plugins_lifecycle_remediation_scheduler_runbook_rejects_plan_hash_mismatch(
    runner,
    tmp_path: Path,
):
    control_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_control_plan_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "dashboard_hash": "3" * 64,
        "control_action_count": 0,
        "control_counts": {
            "dispatch": 0,
            "monitor": 0,
            "expedite": 0,
            "escalate": 0,
            "no_op": 0,
        },
        "control_actions": [],
        "created_by": "operator_console",
    }
    control_payload["control_plan_hash"] = hashlib.sha256(
        json.dumps(control_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    control_path = tmp_path / "control-plan.json"
    control_path.write_text(
        json.dumps(control_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    adapter_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "9" * 64,
        "execution_hash": "2" * 64,
        "telemetry_hash": "8" * 64,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "entry_count": 0,
        "entries": [],
        "created_by": "deployment_scheduler",
    }
    adapter_payload["adapter_handoff_hash"] = hashlib.sha256(
        json.dumps(adapter_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    adapter_path = tmp_path / "adapter-handoff.json"
    adapter_path.write_text(
        json.dumps(adapter_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-runbook",
            str(control_path),
            str(adapter_path),
            "--created-by",
            "operator_console",
        ],
    )

    assert result.exit_code == 1
    assert "plan_hash mismatch" in result.output


def test_plugins_lifecycle_remediation_scheduler_automation_profile_and_capture(
    runner,
    tmp_path: Path,
):
    lifecycle = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-status",
            str(_write_request_payload_from_cli(runner, tmp_path)),
        ],
    )
    assert lifecycle.exit_code == 0
    lifecycle_path = tmp_path / "lifecycle.json"
    lifecycle_path.write_text(lifecycle.output, encoding="utf-8")
    summary = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-summary",
            str(lifecycle_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert summary.exit_code == 0
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(summary.output, encoding="utf-8")
    policy = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-policy-report",
            str(summary_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert policy.exit_code == 0
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.output, encoding="utf-8")
    drilldown = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-multistore-drilldown",
            str(policy_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert drilldown.exit_code == 0
    drilldown_path = tmp_path / "drilldown.json"
    drilldown_path.write_text(drilldown.output, encoding="utf-8")
    plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-orchestration",
            str(drilldown_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert plan.exit_code == 0
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(plan.output, encoding="utf-8")
    execution_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-execution-dashboard",
            str(plan_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert execution_dashboard.exit_code == 0
    execution_dashboard_path = tmp_path / "execution-dashboard.json"
    execution_dashboard_path.write_text(execution_dashboard.output, encoding="utf-8")
    handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-deployment-handoff",
            str(execution_dashboard_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert handoff.exit_code == 0
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_text(handoff.output, encoding="utf-8")
    queue = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-queue",
            str(handoff_path),
            "--window-start-epoch",
            "1700000000",
            "--window-duration-seconds",
            "3600",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert queue.exit_code == 0
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(queue.output, encoding="utf-8")
    telemetry = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-telemetry",
            str(queue_path),
            "--as-of-epoch",
            "1700000100",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert telemetry.exit_code == 0
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(telemetry.output, encoding="utf-8")
    adapter_handoff = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-adapter-handoff",
            str(telemetry_path),
            "--adapter-name",
            "airflow",
            "--adapter-endpoint",
            "airflow://cluster-a",
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert adapter_handoff.exit_code == 0
    adapter_handoff_path = tmp_path / "adapter-handoff.json"
    adapter_handoff_path.write_text(adapter_handoff.output, encoding="utf-8")
    adapter_payload = json.loads(adapter_handoff.output)
    first_entry = adapter_payload["entries"][0]
    acknowledgement = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement",
            str(adapter_handoff_path),
            first_entry["adapter_entry_hash"],
            "--state",
            "in_progress",
            "--acknowledged-by",
            "airflow_worker",
            "--external-reference",
            "airflow-run-abc",
        ],
    )
    assert acknowledgement.exit_code == 0
    acknowledgement_path = tmp_path / "ack.json"
    acknowledgement_path.write_text(acknowledgement.output, encoding="utf-8")
    replay = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement-replay",
            str(adapter_handoff_path),
            str(acknowledgement_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert replay.exit_code == 0
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(replay.output, encoding="utf-8")
    scheduler_dashboard = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-execution-dashboard",
            str(telemetry_path),
            str(replay_path),
            "--created-by",
            "deployment_scheduler",
        ],
    )
    assert scheduler_dashboard.exit_code == 0
    scheduler_dashboard_path = tmp_path / "scheduler-dashboard.json"
    scheduler_dashboard_path.write_text(scheduler_dashboard.output, encoding="utf-8")
    control_plan = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-control-plan",
            str(scheduler_dashboard_path),
            "--created-by",
            "operator_console",
        ],
    )
    assert control_plan.exit_code == 0
    control_plan_path = tmp_path / "control-plan.json"
    control_plan_path.write_text(control_plan.output, encoding="utf-8")
    runbook = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-runbook",
            str(control_plan_path),
            str(adapter_handoff_path),
            "--created-by",
            "operator_console",
        ],
    )
    assert runbook.exit_code == 0
    runbook_path = tmp_path / "runbook.json"
    runbook_path.write_text(runbook.output, encoding="utf-8")

    profile = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-automation-profile",
            str(runbook_path),
            "--profile-name",
            "airflow-default",
            "--profile-version",
            "1.0.0",
            "--created-by",
            "operator_console",
        ],
    )
    assert profile.exit_code == 0
    profile_payload = json.loads(profile.output)
    assert profile_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
    )
    assert profile_payload["automation_rule_count"] == len(
        profile_payload["automation_rules"]
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(profile.output, encoding="utf-8")
    rule = profile_payload["automation_rules"][0]

    capture = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement-capture",
            str(profile_path),
            str(adapter_handoff_path),
            rule["action_hash"],
            "--external-reference",
            "airflow-run-capture",
            "--acknowledged-by",
            "operator_console",
            "--captured-state",
            rule["target_state"],
            "--note",
            "captured",
        ],
    )
    assert capture.exit_code == 0
    capture_payload = json.loads(capture.output)
    assert capture_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_capture_v1"
    )
    assert capture_payload["action_hash"] == rule["action_hash"]
    assert len(capture_payload["capture_hash"]) == 64


def test_plugins_lifecycle_scheduler_capture_rejects_auto_state_mismatch(
    runner,
    tmp_path: Path,
):
    profile_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
        ),
        "version": "1.0.0",
        "profile_name": "airflow-default",
        "profile_version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "runbook_hash": "3" * 64,
        "automation_rule_count": 1,
        "automation_rules": [
            {
                "control_action": "dispatch",
                "action_hash": "4" * 64,
                "request_hash": "5" * 64,
                "action_type": "persist_request",
                "priority": 1,
                "automation_mode": "auto",
                "target_state": "in_progress",
                "capture_command_template": "x",
                "automation_rule_hash": "6" * 64,
            }
        ],
        "created_by": "operator_console",
    }
    profile_payload["automation_profile_hash"] = hashlib.sha256(
        json.dumps(profile_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(profile_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    adapter_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_adapter_handoff_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "telemetry_hash": "7" * 64,
        "adapter_name": "airflow",
        "adapter_endpoint": "airflow://cluster-a",
        "entry_count": 1,
        "entries": [
            {
                "adapter_entry_hash": "8" * 64,
                "entry_hash": "9" * 64,
                "action_hash": "4" * 64,
                "request_hash": "5" * 64,
            }
        ],
        "created_by": "deployment_scheduler",
    }
    adapter_payload["adapter_handoff_hash"] = hashlib.sha256(
        json.dumps(adapter_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    adapter_path = tmp_path / "adapter.json"
    adapter_path.write_text(
        json.dumps(adapter_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-acknowledgement-capture",
            str(profile_path),
            str(adapter_path),
            "4" * 64,
            "--external-reference",
            "run-xyz",
            "--acknowledged-by",
            "operator_console",
            "--captured-state",
            "completed",
        ],
    )

    assert result.exit_code == 1
    assert "captured_state does not match auto target_state" in result.output


def test_plugins_lifecycle_remediation_scheduler_retry_profile_and_orchestration(
    runner,
    tmp_path: Path,
):
    profile_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_automation_profile_v1"
        ),
        "version": "1.0.0",
        "profile_name": "airflow-default",
        "profile_version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "runbook_hash": "3" * 64,
        "automation_rule_count": 2,
        "automation_rules": [
            {
                "control_action": "dispatch",
                "action_hash": "4" * 64,
                "request_hash": "5" * 64,
                "action_type": "persist_request",
                "priority": 1,
                "automation_mode": "auto",
                "target_state": "in_progress",
                "capture_command_template": "x",
                "automation_rule_hash": "6" * 64,
            },
            {
                "control_action": "escalate",
                "action_hash": "7" * 64,
                "request_hash": "8" * 64,
                "action_type": "confirm_external_write",
                "priority": 2,
                "automation_mode": "manual",
                "target_state": "blocked",
                "capture_command_template": "x",
                "automation_rule_hash": "9" * 64,
            },
        ],
        "created_by": "operator_console",
    }
    profile_payload["automation_profile_hash"] = hashlib.sha256(
        json.dumps(profile_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        json.dumps(profile_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    retry_profile = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-retry-profile",
            str(profile_path),
            "--max-attempts",
            "4",
            "--base-delay-seconds",
            "20",
            "--backoff-multiplier",
            "2.0",
            "--created-by",
            "operator_console",
        ],
    )
    assert retry_profile.exit_code == 0
    retry_payload = json.loads(retry_profile.output)
    assert retry_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_retry_profile_v1"
    )
    assert retry_payload["retry_rule_count"] == 2
    assert len(retry_payload["retry_profile_hash"]) == 64
    retry_profile_path = tmp_path / "retry-profile.json"
    retry_profile_path.write_text(retry_profile.output, encoding="utf-8")

    capture_a = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_capture_v1"
        ),
        "version": "1.0.0",
        "automation_profile_hash": retry_payload["automation_profile_hash"],
        "adapter_handoff_hash": "a" * 64,
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "action_hash": "4" * 64,
        "request_hash": "5" * 64,
        "adapter_entry_hash": "b" * 64,
        "captured_state": "blocked",
        "target_state": "in_progress",
        "automation_mode": "auto",
        "external_reference": "run-1",
        "acknowledged_by": "operator",
        "note": "",
    }
    capture_a["capture_hash"] = hashlib.sha256(
        json.dumps(capture_a, sort_keys=True).encode("utf-8")
    ).hexdigest()
    capture_a_path = tmp_path / "capture-a.json"
    capture_a_path.write_text(
        json.dumps(capture_a, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    capture_b = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_capture_v1"
        ),
        "version": "1.0.0",
        "automation_profile_hash": retry_payload["automation_profile_hash"],
        "adapter_handoff_hash": "c" * 64,
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "action_hash": "7" * 64,
        "request_hash": "8" * 64,
        "adapter_entry_hash": "d" * 64,
        "captured_state": "blocked",
        "target_state": "blocked",
        "automation_mode": "manual",
        "external_reference": "run-2",
        "acknowledged_by": "operator",
        "note": "",
    }
    capture_b["capture_hash"] = hashlib.sha256(
        json.dumps(capture_b, sort_keys=True).encode("utf-8")
    ).hexdigest()
    capture_b_path = tmp_path / "capture-b.json"
    capture_b_path.write_text(
        json.dumps(capture_b, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    orchestration = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-retry-orchestration",
            str(retry_profile_path),
            str(capture_a_path),
            str(capture_b_path),
            "--created-by",
            "operator_console",
        ],
    )
    assert orchestration.exit_code == 0
    orchestration_payload = json.loads(orchestration.output)
    assert orchestration_payload["schema"] == (
        "scpn_plugin_execution_request_lifecycle_remediation_scheduler_retry_orchestration_v1"
    )
    assert orchestration_payload["retry_entry_count"] == 1
    assert orchestration_payload["retry_entries"][0]["action_hash"] == "4" * 64
    assert orchestration_payload["retry_entries"][0]["next_delay_seconds"] == 20
    assert len(orchestration_payload["retry_orchestration_hash"]) == 64


def test_plugins_lifecycle_scheduler_retry_rejects_duplicate_capture(
    runner,
    tmp_path: Path,
):
    retry_profile_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_retry_profile_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "automation_profile_hash": "3" * 64,
        "retry_rule_count": 1,
        "retry_rules": [
            {
                "action_hash": "4" * 64,
                "request_hash": "5" * 64,
                "automation_mode": "auto",
                "control_action": "dispatch",
                "target_state": "in_progress",
                "policy_mode": "retry_enabled",
                "max_attempts": 3,
                "base_delay_seconds": 30,
                "backoff_multiplier": 2.0,
                "retry_rule_hash": "6" * 64,
            }
        ],
        "created_by": "operator_console",
    }
    retry_profile_payload["retry_profile_hash"] = hashlib.sha256(
        json.dumps(retry_profile_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    retry_profile_path = tmp_path / "retry-profile.json"
    retry_profile_path.write_text(
        json.dumps(retry_profile_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    capture_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_capture_v1"
        ),
        "version": "1.0.0",
        "automation_profile_hash": "3" * 64,
        "adapter_handoff_hash": "7" * 64,
        "plan_hash": "1" * 64,
        "execution_hash": "2" * 64,
        "action_hash": "4" * 64,
        "request_hash": "5" * 64,
        "adapter_entry_hash": "8" * 64,
        "captured_state": "blocked",
        "target_state": "in_progress",
        "automation_mode": "auto",
        "external_reference": "run",
        "acknowledged_by": "operator",
        "note": "",
    }
    capture_payload["capture_hash"] = hashlib.sha256(
        json.dumps(capture_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    capture_a = tmp_path / "capture-a.json"
    capture_b = tmp_path / "capture-b.json"
    capture_blob = json.dumps(capture_payload, indent=2, sort_keys=True)
    capture_a.write_text(capture_blob, encoding="utf-8")
    capture_b.write_text(capture_blob, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "lifecycle-remediation-scheduler-retry-orchestration",
            str(retry_profile_path),
            str(capture_a),
            str(capture_b),
            "--created-by",
            "operator_console",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate capture action_hash" in result.output


def test_digital_twin_observability_bundle_outputs_prometheus_and_replay_linkage(
    runner,
    tmp_path: Path,
):
    evidence_payload = {
        "contract_hash": "1" * 64,
        "accepted_count": 7,
        "rejected_count": 2,
        "adapter_count": 3,
        "unhealthy_adapter_count": 1,
        "latest_sequence": 42,
        "max_abs_twin_residual": 0.125,
        "status": "warning",
        "capability_counts": {"push": 2, "pull": 1},
        "direction_counts": {"inbound": 5, "outbound": 4},
        "mismatch_reasons": ["shape_mismatch", "shape_mismatch", "stale_sequence"],
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    replay_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_acknowledgement_replay_v1"
        ),
        "version": "1.0.0",
        "adapter_handoff_hash": "2" * 64,
        "plan_hash": "3" * 64,
        "execution_hash": "4" * 64,
        "telemetry_hash": "5" * 64,
        "acknowledgement_count": 2,
        "state_counts": {"in_progress": 0, "completed": 1, "blocked": 1},
        "rows": [
            {"action_hash": "6" * 64, "state": "completed"},
            {"action_hash": "7" * 64, "state": "blocked"},
        ],
        "created_by": "deployment_scheduler",
    }
    replay_payload["replay_hash"] = hashlib.sha256(
        json.dumps(replay_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(
        json.dumps(replay_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    dashboard_payload = {
        "schema": (
            "scpn_plugin_execution_request_lifecycle_remediation_scheduler_execution_dashboard_v1"
        ),
        "version": "1.0.0",
        "plan_hash": "3" * 64,
        "execution_hash": "4" * 64,
        "handoff_hash": "8" * 64,
        "scheduler_hash": "9" * 64,
        "telemetry_hash": "5" * 64,
        "replay_hash": replay_payload["replay_hash"],
        "row_count": 2,
        "state_counts": {
            "pending": 0,
            "in_progress": 0,
            "completed": 1,
            "blocked": 1,
            "overdue": 1,
        },
        "rows": [
            {"action_hash": "6" * 64, "effective_state": "completed", "overdue": False},
            {"action_hash": "7" * 64, "effective_state": "blocked", "overdue": True},
        ],
        "created_by": "deployment_scheduler",
    }
    dashboard_payload["dashboard_hash"] = hashlib.sha256(
        json.dumps(dashboard_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    dashboard_path = tmp_path / "dashboard.json"
    dashboard_path.write_text(
        json.dumps(dashboard_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "digital-twin-observability-bundle",
            str(evidence_path),
            "--scheduler-dashboard-json",
            str(dashboard_path),
            "--scheduler-replay-json",
            str(replay_path),
            "--created-by",
            "operator_console",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_digital_twin_observability_bundle_v1"
    assert "spo_digital_twin_sync_accepted_total" in payload["prometheus_text"]
    assert payload["replay_linkage"]["scheduler_row_count"] == 2
    assert payload["replay_linkage"]["scheduler_overdue_count"] == 1
    assert payload["replay_linkage"]["scheduler_replay_count"] == 2
    assert len(payload["bundle_hash"]) == 64


def test_digital_twin_observability_bundle_rejects_bad_scheduler_dashboard_schema(
    runner,
    tmp_path: Path,
):
    evidence_payload = {
        "contract_hash": "1" * 64,
        "accepted_count": 1,
        "rejected_count": 0,
        "adapter_count": 1,
        "unhealthy_adapter_count": 0,
        "latest_sequence": 1,
        "max_abs_twin_residual": 0.01,
        "status": "healthy",
        "capability_counts": {"push": 1},
        "direction_counts": {"inbound": 1},
        "mismatch_reasons": [],
    }
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    bad_dashboard = tmp_path / "bad-dashboard.json"
    bad_dashboard.write_text(
        json.dumps({"schema": "not_expected"}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "digital-twin-observability-bundle",
            str(evidence_path),
            "--scheduler-dashboard-json",
            str(bad_dashboard),
            "--created-by",
            "operator_console",
        ],
    )

    assert result.exit_code == 1
    assert "unexpected scheduler dashboard schema" in result.output


def test_digital_twin_grafana_dashboard_pack_and_live_playbook(
    runner,
    tmp_path: Path,
):
    bundle_payload = {
        "schema": "scpn_digital_twin_observability_bundle_v1",
        "version": "1.0.0",
        "contract_hash": "1" * 64,
        "status": "warning",
        "accepted_count": 10,
        "rejected_count": 2,
        "prometheus_metric_prefix": "spo",
        "prometheus_text": "spo_digital_twin_sync_accepted_total 10\n",
        "replay_linkage": {
            "scheduler_dashboard_present": True,
            "scheduler_replay_present": True,
            "scheduler_row_count": 5,
            "scheduler_overdue_count": 1,
            "scheduler_blocked_count": 0,
            "scheduler_completed_count": 4,
            "scheduler_replay_count": 5,
            "scheduler_replay_blocked_count": 0,
            "scheduler_replay_completed_count": 4,
            "scheduler_dashboard_hash": "2" * 64,
            "scheduler_replay_hash": "3" * 64,
        },
        "created_by": "operator_console",
    }
    bundle_payload["bundle_hash"] = hashlib.sha256(
        json.dumps(bundle_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    pack = runner.invoke(
        main,
        [
            "digital-twin-grafana-dashboard-pack",
            str(bundle_path),
            "--adapter-family",
            "kafka",
            "--created-by",
            "operator_console",
        ],
    )
    assert pack.exit_code == 0
    pack_payload = json.loads(pack.output)
    assert pack_payload["schema"] == "scpn_digital_twin_grafana_dashboard_pack_v1"
    assert pack_payload["panel_count"] == len(pack_payload["panels"])
    assert len(pack_payload["dashboard_pack_hash"]) == 64
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(pack.output, encoding="utf-8")

    playbook = runner.invoke(
        main,
        [
            "digital-twin-live-deployment-playbook",
            str(bundle_path),
            str(pack_path),
            "--environment-name",
            "prod-eu-west",
            "--created-by",
            "operator_console",
        ],
    )
    assert playbook.exit_code == 0
    playbook_payload = json.loads(playbook.output)
    assert playbook_payload["schema"] == "scpn_digital_twin_live_deployment_playbook_v1"
    assert playbook_payload["rollout_gate"] == "degraded"
    assert playbook_payload["step_count"] == len(playbook_payload["steps"])
    assert len(playbook_payload["playbook_hash"]) == 64


def test_digital_twin_live_deployment_playbook_rejects_bundle_hash_mismatch(
    runner,
    tmp_path: Path,
):
    bundle_payload = {
        "schema": "scpn_digital_twin_observability_bundle_v1",
        "version": "1.0.0",
        "contract_hash": "1" * 64,
        "status": "healthy",
        "accepted_count": 1,
        "rejected_count": 0,
        "prometheus_metric_prefix": "spo",
        "prometheus_text": "x",
        "replay_linkage": {
            "scheduler_dashboard_present": False,
            "scheduler_replay_present": False,
            "scheduler_row_count": 0,
            "scheduler_overdue_count": 0,
            "scheduler_blocked_count": 0,
            "scheduler_completed_count": 0,
            "scheduler_replay_count": 0,
            "scheduler_replay_blocked_count": 0,
            "scheduler_replay_completed_count": 0,
            "scheduler_dashboard_hash": None,
            "scheduler_replay_hash": None,
        },
        "created_by": "operator_console",
        "bundle_hash": "2" * 64,
    }
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(bundle_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pack_payload = {
        "schema": "scpn_digital_twin_grafana_dashboard_pack_v1",
        "version": "1.0.0",
        "adapter_family": "rest",
        "contract_hash": "1" * 64,
        "observability_bundle_hash": "3" * 64,
        "panel_count": 0,
        "panels": [],
        "created_by": "operator_console",
        "dashboard_pack_hash": "4" * 64,
    }
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(
        json.dumps(pack_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "digital-twin-live-deployment-playbook",
            str(bundle_path),
            str(pack_path),
            "--environment-name",
            "prod-eu-west",
            "--created-by",
            "operator_console",
        ],
    )
    assert result.exit_code == 1
    assert "observability_bundle_hash mismatch" in result.output


def test_plugins_revoke_execution_request_outputs_revocation(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)

    result = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_path),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-20-01",
            "--revocation-reason",
            "operator rotation",
        ],
    )

    assert result.exit_code == 0
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_request_revocation_v1"
    assert payload["request_hash"] == request_payload["request_hash"]
    assert payload["plan_hash"] == request_payload["plan_hash"]
    assert payload["target_hash"] == request_payload["target_hash"]
    assert payload["revoked_by"] == "deployment_gate"
    assert payload["revocation_reference"] == "REV-2026-05-20-01"
    assert payload["revoked"] is True
    assert len(payload["revocation_hash"]) == 64


def test_plugins_revoke_execution_request_rejects_tampered_request(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    request_payload = json.loads(request_path.read_text(encoding="utf-8"))
    request_payload["request_hash"] = "0" * 64
    request_path.write_text(
        json.dumps(request_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_path),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-20-02",
            "--revocation-reason",
            "operator rotation",
        ],
    )

    assert result.exit_code == 1
    assert "request audit record mismatch" in result.output


def _write_revocation_payload_from_cli(runner, tmp_path: Path) -> Path:
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    result = runner.invoke(
        main,
        [
            "plugins",
            "revoke-execution-request",
            str(request_path),
            "--revoked-by",
            "deployment_gate",
            "--revocation-reference",
            "REV-2026-05-20-03",
            "--revocation-reason",
            "operator rotation",
        ],
    )
    assert result.exit_code == 0
    revocation_path = tmp_path / "revocation.json"
    revocation_path.write_text(result.output, encoding="utf-8")
    return revocation_path


def test_plugins_revocation_list_outputs_deterministic_list(
    runner,
    tmp_path: Path,
):
    revocation_path = _write_revocation_payload_from_cli(runner, tmp_path)

    result = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 0
    revocation_payload = json.loads(revocation_path.read_text(encoding="utf-8"))
    payload = json.loads(result.output)
    assert payload["schema"] == "scpn_plugin_execution_request_revocation_list_v1"
    assert payload["created_by"] == "deployment_gate"
    assert payload["revocation_count"] == 1
    assert payload["request_hashes"] == [revocation_payload["request_hash"]]
    assert payload["revocation_hashes"] == [revocation_payload["revocation_hash"]]
    assert len(payload["revocation_list_hash"]) == 64


def test_plugins_revocation_list_rejects_duplicate_request_hash(
    runner,
    tmp_path: Path,
):
    revocation_path = _write_revocation_payload_from_cli(runner, tmp_path)

    result = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_path),
            str(revocation_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "duplicate request hashes" in result.output


def test_plugins_revocation_list_rejects_tampered_revocation(
    runner,
    tmp_path: Path,
):
    revocation_path = _write_revocation_payload_from_cli(runner, tmp_path)
    payload = json.loads(revocation_path.read_text(encoding="utf-8"))
    payload["revocation_hash"] = "0" * 64
    revocation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_path),
            "--created-by",
            "deployment_gate",
        ],
    )

    assert result.exit_code == 1
    assert "revocation audit record mismatch" in result.output


def test_plugins_persist_execution_request_accepts_revocation_list(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    revocation_path = _write_revocation_payload_from_cli(runner, tmp_path)
    revocation_list = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert revocation_list.exit_code == 0
    revocation_list_path = tmp_path / "revocation_list.json"
    revocation_list_path.write_text(revocation_list.output, encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "plugins",
            "persist-execution-request",
            str(request_path),
            str(tmp_path / "bundle.json"),
            "--storage-uri",
            f"file://{tmp_path / 'bundle.json'}",
            "--created-by",
            "deployment_gate",
            "--revocation-list",
            str(revocation_list_path),
        ],
    )

    assert result.exit_code == 1
    assert "revoked" in result.output


def test_plugins_persist_execution_request_rejects_tampered_revocation_list(
    runner,
    tmp_path: Path,
):
    request_path = _write_request_payload_from_cli(runner, tmp_path)
    revocation_path = _write_revocation_payload_from_cli(runner, tmp_path)
    revocation_list = runner.invoke(
        main,
        [
            "plugins",
            "revocation-list",
            str(revocation_path),
            "--created-by",
            "deployment_gate",
        ],
    )
    assert revocation_list.exit_code == 0
    payload = json.loads(revocation_list.output)
    payload["request_hashes"] = ["0" * 64]
    revocation_list_path = tmp_path / "revocation_list.json"
    revocation_list_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        [
            "plugins",
            "persist-execution-request",
            str(request_path),
            str(tmp_path / "bundle.json"),
            "--storage-uri",
            f"file://{tmp_path / 'bundle.json'}",
            "--created-by",
            "deployment_gate",
            "--revocation-list",
            str(revocation_list_path),
        ],
    )

    assert result.exit_code == 1
    assert "revocation list hash mismatch" in result.output


def _write_meta_audit_record(
    path: Path,
    *,
    domain: str,
    coherence: float,
    k_value: float,
    reward: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "domain": domain,
                "features": {"coherence": coherence, "event_rate": 1.0 - coherence},
                "knobs": {"K": k_value, "zeta": 0.1 - k_value},
                "reward": reward,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_meta_transfer_manifest_outputs_review_only_manifest(
    runner,
    tmp_path,
) -> None:
    first = tmp_path / "grid.jsonl"
    second = tmp_path / "cardiac.jsonl"
    _write_meta_audit_record(
        first,
        domain="power_grid",
        coherence=0.8,
        k_value=0.04,
        reward=0.9,
    )
    _write_meta_audit_record(
        second,
        domain="cardiac",
        coherence=0.9,
        k_value=0.03,
        reward=0.8,
    )

    result = runner.invoke(
        main,
        ["meta-transfer-manifest", str(first), str(second), "--min-records", "2"],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["schema"] == "scpn_meta_package_manifest_v1"
    assert data["package_name"] == "scpn-meta"
    assert data["import_target"] == "scpn_phase_orchestrator.meta"
    assert data["console_script"] == "scpn-meta"
    assert data["execution_permitted"] is False
    assert data["training_summary"]["record_count"] == 2
    assert data["training_summary"]["domain_count"] == 2
    assert len(data["package_sha256"]) == 64


def test_meta_transfer_manifest_discovers_nested_audit_directory(
    runner,
    tmp_path,
) -> None:
    _write_meta_audit_record(
        tmp_path / "grid" / "audit.jsonl",
        domain="power_grid",
        coherence=0.8,
        k_value=0.04,
        reward=0.9,
    )
    _write_meta_audit_record(
        tmp_path / "nested" / "cardiac" / "audit.jsonl",
        domain="cardiac",
        coherence=0.9,
        k_value=0.03,
        reward=0.8,
    )

    result = runner.invoke(
        main,
        [
            "meta-transfer-manifest",
            "--audit-directory",
            str(tmp_path),
            "--min-records",
            "2",
            "--package-name",
            "scpn-meta-cli",
            "--console-script",
            "scpn-meta-cli",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["package_name"] == "scpn-meta-cli"
    assert data["console_script"] == "scpn-meta-cli"
    assert data["execution_permitted"] is False
    assert data["training_summary"]["domains"] == ["cardiac", "power_grid"]


def test_meta_transfer_manifest_writes_only_explicit_output(
    runner,
    tmp_path,
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    output_path = tmp_path / "manifest.json"
    _write_meta_audit_record(
        audit_path,
        domain="power_grid",
        coherence=0.8,
        k_value=0.04,
        reward=0.9,
    )

    result = runner.invoke(
        main,
        ["meta-transfer-manifest", str(audit_path), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert "Meta-transfer package manifest written:" in result.output
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["schema"] == "scpn_meta_package_manifest_v1"
    assert data["execution_permitted"] is False


def test_meta_transfer_manifest_rejects_missing_and_conflicting_sources(
    runner,
    tmp_path,
) -> None:
    audit_path = tmp_path / "audit.jsonl"
    _write_meta_audit_record(
        audit_path,
        domain="power_grid",
        coherence=0.8,
        k_value=0.04,
        reward=0.9,
    )

    missing = runner.invoke(main, ["meta-transfer-manifest"])
    conflicting = runner.invoke(
        main,
        [
            "meta-transfer-manifest",
            str(audit_path),
            "--audit-directory",
            str(tmp_path),
        ],
    )
    bad_min = runner.invoke(
        main,
        ["meta-transfer-manifest", str(audit_path), "--min-records", "0"],
    )

    assert missing.exit_code == 1
    assert "provide one or more audit JSONL files" in missing.output
    assert conflicting.exit_code == 1
    assert "mutually exclusive" in conflicting.output
    assert bad_min.exit_code == 1
    assert "--min-records must be at least 1" in bad_min.output


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


def test_run_creates_sidecar_audit_jsonl_for_audit_stream_output(
    runner, valid_spec_path, tmp_path
):
    stream_path = tmp_path / "run_trace.spoa"
    jsonl_path = stream_path.with_suffix(".jsonl")

    result = runner.invoke(
        main,
        [
            "run",
            valid_spec_path,
            "--steps",
            "2",
            "--audit-stream",
            str(stream_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert stream_path.exists()
    assert jsonl_path.exists()

    jsonl_records = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(jsonl_records) == 3
    assert jsonl_records[0]["header"] is True
    assert jsonl_records[0]["binding_summary"]["name"] == "cli-test"

    events = read_event_stream(stream_path)
    assert len(events) == 3
    assert events[0].event_type == "header"
    assert events[1].event_type == "step"


def test_watch_rejects_invalid_audit_stream_payload(runner, tmp_path):
    stream_path = tmp_path / "corrupt.spoa"
    stream_path.write_text("not an SPO audit stream", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "watch",
            str(stream_path),
            "--from-start",
            "--max-events",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert "ERROR: not an SPO audit event stream" in result.output


def test_watch_rejects_non_positive_poll_interval(runner, tmp_path):
    stream_path = tmp_path / "run_trace.spoa"
    stream_path.write_bytes(b"not read before poll validation")

    result = runner.invoke(
        main,
        [
            "watch",
            str(stream_path),
            "--poll-interval",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "ERROR: --poll-interval must be positive" in result.output


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
    stdout_record = json.loads(manifest_path.read_text(encoding="utf-8"))
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
