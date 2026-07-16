# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — `spo quickstart` command tests

"""Tests for the `spo quickstart` golden-path demo command.

The command chains validation, simulation, replay and reporting on the bundled
research-tier power binding; the suite proves the end-to-end path, the file
output, and every failure branch (missing asset, validation failure, audit
integrity failure, and the empty-report path).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli import quickstart as quickstart_mod


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.mark.parametrize("domain", ["power", "eeg"])
def test_quickstart_runs_the_golden_path(runner: CliRunner, domain: str) -> None:
    result = runner.invoke(main, ["quickstart", domain, "--steps", "30"])
    assert result.exit_code == 0, result.output
    assert "[1/4] validate: OK" in result.output
    assert "[2/4] run: 30 steps" in result.output
    assert "[3/4] replay: audit hash chain verified" in result.output
    assert "[4/4] report:" in result.output
    assert "Explainability Report" in result.output
    assert "safety_tier=research" in result.output


@pytest.mark.parametrize("domain", ["power", "eeg"])
def test_quickstart_bundles_a_research_tier_asset(domain: str) -> None:
    binding = quickstart_mod._ASSET_ROOT / domain / "binding_spec.yaml"
    policy = quickstart_mod._ASSET_ROOT / domain / "policy.yaml"
    assert binding.exists()
    assert policy.exists()
    assert "safety_tier: research" in binding.read_text(encoding="utf-8")


def test_quickstart_writes_the_report_to_a_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    report_path = tmp_path / "power_report.md"
    result = runner.invoke(
        main,
        ["quickstart", "power", "--steps", "30", "--output", str(report_path)],
    )
    assert result.exit_code == 0, result.output
    assert f"Markdown report written to {report_path}" in result.output
    assert "Explainability Report" in report_path.read_text(encoding="utf-8")


def test_quickstart_rejects_an_unknown_domain(runner: CliRunner) -> None:
    result = runner.invoke(main, ["quickstart", "fusion"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output


def test_quickstart_errors_when_the_asset_is_missing(
    runner: CliRunner, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(quickstart_mod, "_ASSET_ROOT", tmp_path / "absent")
    result = runner.invoke(main, ["quickstart", "power"])
    assert result.exit_code != 0
    assert "quickstart asset not found" in result.output


def test_quickstart_fails_on_validation_errors(runner: CliRunner, monkeypatch) -> None:
    monkeypatch.setattr(
        quickstart_mod, "validate_binding_spec", lambda _spec: ["bad layer"]
    )
    result = runner.invoke(main, ["quickstart", "power"])
    assert result.exit_code == 1
    assert "ERROR: bad layer" in result.output


def test_quickstart_fails_on_audit_integrity_failure(
    runner: CliRunner, monkeypatch
) -> None:
    monkeypatch.setattr(
        quickstart_mod.ReplayEngine,
        "verify_integrity",
        staticmethod(lambda _entries: (False, 0)),
    )
    result = runner.invoke(main, ["quickstart", "power", "--steps", "20"])
    assert result.exit_code == 1
    assert "audit hash chain failed verification" in result.output


def test_quickstart_handles_an_empty_report_summary(
    runner: CliRunner, monkeypatch
) -> None:
    monkeypatch.setattr(
        quickstart_mod, "build_audit_report_summary", lambda *a, **k: {}
    )
    result = runner.invoke(main, ["quickstart", "power", "--steps", "20"])
    assert result.exit_code == 0, result.output
    assert "[4/4] report: generated" in result.output


def _committed_evidence() -> dict[str, Any]:
    return json.loads(quickstart_mod._EVIDENCE_RECORD.read_text(encoding="utf-8"))


def test_quickstart_evidence_reverifies_and_prints_the_verdict(
    runner: CliRunner,
) -> None:
    result = runner.invoke(main, ["quickstart", "evidence"])
    assert result.exit_code == 0, result.output
    assert "top-level seal: VERIFIED" in result.output
    assert "nested PRC seal: VERIFIED" in result.output
    assert "verdict: flagged_for_review" in result.output
    assert "real, non-synthetic" in result.output
    # The honest boundary must be stated on the onboarding path.
    assert "review_only=True" in result.output
    assert "not a live-actuation claim" in result.output


def test_quickstart_evidence_writes_the_verdict_to_a_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "verdict.txt"
    result = runner.invoke(main, ["quickstart", "evidence", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert f"Evidence verdict written to {out}" in result.output
    assert "top-level seal: VERIFIED" in out.read_text(encoding="utf-8")


def test_verify_evidence_seals_accepts_the_committed_record() -> None:
    assert quickstart_mod._verify_evidence_seals(_committed_evidence()) == (True, True)


def test_verify_evidence_seals_rejects_a_tampered_top_level() -> None:
    record = _committed_evidence()
    record["sample_count"] = int(record["sample_count"]) + 1
    top_ok, nested_ok = quickstart_mod._verify_evidence_seals(record)
    assert top_ok is False
    assert nested_ok is True


def test_verify_evidence_seals_rejects_a_tampered_nested_prc() -> None:
    record = _committed_evidence()
    record["prc_evidence"]["flagged_count"] = 999
    _top_ok, nested_ok = quickstart_mod._verify_evidence_seals(record)
    assert nested_ok is False


def test_verify_evidence_seals_handles_a_missing_nested_prc() -> None:
    record = _committed_evidence()
    record["prc_evidence"] = None
    _top_ok, nested_ok = quickstart_mod._verify_evidence_seals(record)
    assert nested_ok is False


def test_quickstart_evidence_fails_closed_on_a_broken_seal(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record = _committed_evidence()
    record["sample_count"] = int(record["sample_count"]) + 1
    tampered = tmp_path / "tampered_evidence.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(quickstart_mod, "_EVIDENCE_RECORD", tampered)

    result = runner.invoke(main, ["quickstart", "evidence"])

    assert result.exit_code == 1
    assert "top-level seal: FAILED" in result.output
    assert "failed to recompute" in result.output


def test_quickstart_evidence_reports_a_missing_record(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(quickstart_mod, "_EVIDENCE_RECORD", tmp_path / "absent.json")

    result = runner.invoke(main, ["quickstart", "evidence"])

    assert result.exit_code != 0
    assert "sealed evidence record not found" in result.output


def test_quickstart_auditor_bundles_a_flagged_synthetic_fixture() -> None:
    spec = json.loads(quickstart_mod._AUDITOR_SCORES.read_text(encoding="utf-8"))
    assert spec["event_scores"] and spec["null_scores"]
    # The onboarding fixture must not be mistaken for real detector output.
    assert "synthetic" in spec["description"].lower()
    assert "not real" in spec["description"].lower()


def test_quickstart_auditor_runs_the_detector_audit(runner: CliRunner) -> None:
    result = runner.invoke(main, ["quickstart", "auditor"])
    assert result.exit_code == 0, result.output
    assert "detection_rate=" in result.output
    assert "achieved_false_alarm=" in result.output
    assert "permutation p=" in result.output
    assert "beats_chance=True" in result.output
    assert "synthetic onboarding fixture, not real detector output" in result.output


def test_quickstart_auditor_writes_the_verdict_to_a_file(
    runner: CliRunner, tmp_path: Path
) -> None:
    out = tmp_path / "auditor.txt"
    result = runner.invoke(main, ["quickstart", "auditor", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert f"Auditor verdict written to {out}" in result.output
    assert "detection_rate=" in out.read_text(encoding="utf-8")


def test_quickstart_auditor_reports_missing_scores(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(quickstart_mod, "_AUDITOR_SCORES", tmp_path / "absent.json")
    result = runner.invoke(main, ["quickstart", "auditor"])
    assert result.exit_code != 0
    assert "bundled auditor scores not found" in result.output


def test_quickstart_auditor_reports_malformed_scores(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"detector_name": "d"}), encoding="utf-8")
    monkeypatch.setattr(quickstart_mod, "_AUDITOR_SCORES", malformed)
    result = runner.invoke(main, ["quickstart", "auditor"])
    assert result.exit_code != 0
    assert "malformed" in result.output
