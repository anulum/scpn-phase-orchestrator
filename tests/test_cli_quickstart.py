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

from pathlib import Path

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main
from scpn_phase_orchestrator.runtime.cli import quickstart as quickstart_mod


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_quickstart_power_runs_the_golden_path(runner: CliRunner) -> None:
    result = runner.invoke(main, ["quickstart", "power", "--steps", "30"])
    assert result.exit_code == 0, result.output
    assert "[1/4] validate: OK" in result.output
    assert "[2/4] run: 30 steps" in result.output
    assert "[3/4] replay: audit hash chain verified" in result.output
    assert "[4/4] report:" in result.output
    assert "Explainability Report" in result.output
    assert "safety_tier=research" in result.output


def test_quickstart_bundles_a_research_tier_asset() -> None:
    binding = quickstart_mod._ASSET_ROOT / "power" / "binding_spec.yaml"
    policy = quickstart_mod._ASSET_ROOT / "power" / "policy.yaml"
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
