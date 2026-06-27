# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — assurance-case CLI command tests

"""CliRunner tests for the ``spo assurance-case`` command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.assurance as cli_assurance
from scpn_phase_orchestrator.runtime.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_evidence(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "evidence_id": "twin",
                    "category": "twin_confidence",
                    "summary": "twin healthy",
                    "record": {"coverage": 0.9},
                },
                {
                    "evidence_id": "cptc",
                    "category": "conformal_gate",
                    "summary": "gate calibrated",
                    "record": {"alpha": 0.1},
                },
            ]
        ),
        encoding="utf-8",
    )


def test_cli_module_exposes_command() -> None:
    assert cli_assurance.assurance_case.name == "assurance-case"
    assert cli_assurance.certification_evidence.name == "certification-evidence"


def test_build_from_evidence_file_to_stdout(runner: CliRunner, tmp_path: Path) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(evidence)]
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.output)
    assert bundle["schema"] == "scpn_assurance_case_bundle_v1"
    assert bundle["actuation_permitted"] is False
    assert bundle["system_name"] == "Sys"


def test_build_to_output_file(runner: CliRunner, tmp_path: Path) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        main,
        [
            "assurance-case",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    bundle = json.loads(out.read_text(encoding="utf-8"))
    assert bundle["bundle_hash"] in result.output


def test_build_certification_evidence_package(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "review_package"
    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads((out / "assurance_bundle.json").read_text(encoding="utf-8"))
    vectors = json.loads((out / "test_vectors.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == "scpn_certification_evidence_package_v1"
    assert manifest["system_name"] == "Sys"
    assert manifest["assurance_bundle_hash"] == bundle["bundle_hash"]
    assert {row["path"] for row in manifest["files"]} == {
        "assurance_bundle.json",
        "test_vectors.json",
    }
    assert [row["evidence_id"] for row in vectors["evidence_hash_vectors"]] == [
        "cptc",
        "twin",
    ]
    assert "technical evidence-mapping package" in manifest["disclaimer"]


def test_certification_evidence_refuses_non_empty_output_dir(
    runner: CliRunner, tmp_path: Path
) -> None:
    evidence = tmp_path / "ev.json"
    _write_evidence(evidence)
    out = tmp_path / "review_package"
    out.mkdir()
    (out / "stale.json").write_text("{}", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "certification-evidence",
            "--system",
            "Sys",
            "--evidence-file",
            str(evidence),
            "--output-dir",
            str(out),
        ],
    )

    assert result.exit_code != 0
    assert "output directory is not empty" in result.output


def test_build_from_audit_log(runner: CliRunner, tmp_path: Path) -> None:
    log = tmp_path / "audit.jsonl"
    log.write_text(
        json.dumps({"header": True, "n_oscillators": 2, "dt": 0.01}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--audit-log", str(log)]
    )
    assert result.exit_code == 0, result.output
    bundle = json.loads(result.output)
    categories = {item["category"] for item in bundle["evidence"]}
    assert "audit_logging" in categories


def test_no_evidence_is_an_error(runner: CliRunner) -> None:
    result = runner.invoke(main, ["assurance-case", "--system", "Sys"])
    assert result.exit_code != 0
    assert "no evidence supplied" in result.output


def test_missing_field_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps([{"evidence_id": "x", "category": "twin_confidence"}]),
        encoding="utf-8",
    )
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(bad)]
    )
    assert result.exit_code != 0
    assert "missing required field" in result.output


def test_non_object_row_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(["not-an-object"]), encoding="utf-8")
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(bad)]
    )
    assert result.exit_code != 0
    assert "must be a JSON object" in result.output


def test_invalid_category_is_reported(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "evidence_id": "x",
                "category": "not_a_category",
                "summary": "s",
                "record": {"v": 1},
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(
        main, ["assurance-case", "--system", "Sys", "--evidence-file", str(bad)]
    )
    assert result.exit_code != 0
    assert "is invalid" in result.output
