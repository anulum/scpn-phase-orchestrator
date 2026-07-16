# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — CLI audit-detector command tests

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.evaluation as cli_evaluation
from scpn_phase_orchestrator.runtime.cli import main


def _write_spec(path: Path, spec: object) -> Path:
    path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def _skilful_spec() -> dict[str, object]:
    rng = np.random.default_rng(0)
    return {
        "detector_name": "demo-cli",
        "event_scores": rng.normal(2.0, 1.0, 50).tolist(),
        "null_scores": rng.normal(0.0, 1.0, 300).tolist(),
    }


class TestCommandRegistration:
    def test_command_is_a_registered_click_command(self):
        command = cli_evaluation.audit_detector_command
        assert isinstance(command, click.Command)
        assert command.name == "audit-detector"
        assert "audit-detector" in main.commands


class TestAuditDetectorCommand:
    def test_bare_verdict_without_sealing(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main, ["audit-detector", str(spec), "--n-permutations", "500"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["detector_name"] == "demo-cli"
        assert payload["beats_chance"] is True
        assert payload["significance"]["p_value"] < 0.05
        # A bare verdict carries no seal.
        assert "content_hash" not in payload

    def test_sealed_record_when_provenance_given(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main,
            [
                "audit-detector",
                str(spec),
                "--n-permutations",
                "500",
                "--corpus-id",
                "demo-corpus",
                "--captured-at",
                "2026-07-07T15:00:00+02:00",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload["content_hash"]) == 64
        assert payload["corpus_id"] == "demo-corpus"
        assert payload["audit"]["detector_name"] == "demo-cli"

    def test_output_file_is_written(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        out = tmp_path / "verdict.json"
        result = CliRunner().invoke(
            main,
            [
                "audit-detector",
                str(spec),
                "--n-permutations",
                "200",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert json.loads(out.read_text())["detector_name"] == "demo-cli"

    def test_detector_name_defaults_when_absent(self, tmp_path):
        spec = _skilful_spec()
        del spec["detector_name"]
        path = _write_spec(tmp_path / "scores.json", spec)
        result = CliRunner().invoke(
            main, ["audit-detector", str(path), "--n-permutations", "200"]
        )
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["detector_name"] == "detector"

    def test_reproducible_seed(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        runs = [
            CliRunner().invoke(
                main,
                ["audit-detector", str(spec), "--n-permutations", "500", "--seed", "7"],
            )
            for _ in range(2)
        ]
        p_values = [json.loads(r.output)["significance"]["p_value"] for r in runs]
        assert p_values[0] == p_values[1]


class TestAuditDetectorErrors:
    def test_invalid_json_is_reported(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        result = CliRunner().invoke(main, ["audit-detector", str(bad)])
        assert result.exit_code != 0
        assert "not valid JSON" in result.output

    def test_non_object_payload_rejected(self, tmp_path):
        spec = _write_spec(tmp_path / "list.json", [1, 2, 3])
        result = CliRunner().invoke(main, ["audit-detector", str(spec)])
        assert result.exit_code != 0
        assert "must hold a JSON object" in result.output

    def test_missing_event_scores_rejected(self, tmp_path):
        spec = _write_spec(tmp_path / "s.json", {"null_scores": [0.0, 1.0]})
        result = CliRunner().invoke(main, ["audit-detector", str(spec)])
        assert result.exit_code != 0
        assert "'event_scores' must be a JSON array" in result.output

    def test_empty_scores_rejected(self, tmp_path):
        spec = _write_spec(
            tmp_path / "s.json", {"event_scores": [], "null_scores": [0.0]}
        )
        result = CliRunner().invoke(main, ["audit-detector", str(spec)])
        assert result.exit_code != 0
        assert "'event_scores' must not be empty" in result.output

    def test_non_numeric_score_rejected(self, tmp_path):
        spec = _write_spec(
            tmp_path / "s.json",
            {"event_scores": [1.0, "x"], "null_scores": [0.0, 1.0]},
        )
        result = CliRunner().invoke(main, ["audit-detector", str(spec)])
        assert result.exit_code != 0
        assert "must be a number" in result.output

    def test_bool_score_rejected(self, tmp_path):
        spec = _write_spec(
            tmp_path / "s.json",
            {"event_scores": [1.0, True], "null_scores": [0.0, 1.0]},
        )
        result = CliRunner().invoke(main, ["audit-detector", str(spec)])
        assert result.exit_code != 0
        assert "must be a number" in result.output

    def test_non_finite_score_rejected(self, tmp_path):
        # Infinity is legal JSON to Python's json module but not a finite score.
        (tmp_path / "s.json").write_text(
            '{"event_scores": [Infinity], "null_scores": [0.0, 1.0]}',
            encoding="utf-8",
        )
        result = CliRunner().invoke(main, ["audit-detector", str(tmp_path / "s.json")])
        assert result.exit_code != 0
        assert "must be finite" in result.output

    def test_nan_score_rejected(self, tmp_path):
        # NaN is legal to Python's json module but not a finite score.
        (tmp_path / "s.json").write_text(
            '{"event_scores": [NaN], "null_scores": [0.0, 1.0]}',
            encoding="utf-8",
        )
        result = CliRunner().invoke(main, ["audit-detector", str(tmp_path / "s.json")])
        assert result.exit_code != 0
        assert "must be finite" in result.output

    def test_half_provenance_rejected(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main,
            ["audit-detector", str(spec), "--corpus-id", "only-id"],
        )
        assert result.exit_code != 0
        assert "must be given together" in result.output

    def test_out_of_range_target_false_alarm_reported(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main,
            ["audit-detector", str(spec), "--target-false-alarm", "2.0"],
        )
        assert result.exit_code != 0
        assert "target_fa must be in" in result.output


class TestSignedSeal:
    def test_sign_emits_a_verifiable_signature(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPO_AUDIT_KEY", "cli-signing-key")
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main,
            [
                "audit-detector",
                str(spec),
                "--corpus-id",
                "demo-corpus",
                "--captured-at",
                "2026-07-07T15:00:00+02:00",
                "--sign",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload["signature"]) == 64
        assert payload["signature_algorithm"] == "HMAC-SHA256"
        assert "signing_key_id" in payload

    def test_sign_without_key_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPO_AUDIT_KEY", raising=False)
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main,
            [
                "audit-detector",
                str(spec),
                "--corpus-id",
                "demo-corpus",
                "--captured-at",
                "2026-07-07T15:00:00+02:00",
                "--sign",
            ],
        )
        assert result.exit_code != 0
        assert "SPO_AUDIT_KEY" in result.output

    def test_sign_without_sealing_is_reported(self, tmp_path):
        spec = _write_spec(tmp_path / "scores.json", _skilful_spec())
        result = CliRunner().invoke(
            main,
            ["audit-detector", str(spec), "--sign"],
        )
        assert result.exit_code != 0
        assert "--sign requires" in result.output
