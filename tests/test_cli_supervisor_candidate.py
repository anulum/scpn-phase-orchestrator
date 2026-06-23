# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — supervisor-candidate CLI tests

"""Tests for the ``spo supervisor-candidate`` command.

The command is checked over a valid scenario (summary output and the sealed
bundle JSON), array-valued knobs, and the malformed-input, missing-field,
empty-observation, and invalid-value error paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli._app import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _scenario() -> dict[str, Any]:
    return {
        "candidate": {"alpha": 0.2, "zeta": 0.05, "channel_weights": [1.0]},
        "baseline": {"alpha": 0.0, "zeta": 0.0, "channel_weights": [0.0]},
        "incumbent": {"alpha": 0.1, "zeta": 0.02, "channel_weights": [0.5]},
        "observations": [
            {"coherence": 0.82, "previous_coherence": 0.74, "lyapunov_exponent": -0.02}
        ],
        "constraints": {"max_lyapunov_exponent": 0.0},
        "safety_tier": "research",
        "numeric_provenance": {"active_backend": "python", "parity_tolerance": 1e-9},
    }


def _write(tmp_path: Path, scenario: dict[str, Any]) -> str:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    return str(path)


def test_valid_scenario_prints_summary(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        main, ["supervisor-candidate", _write(tmp_path, _scenario())]
    )
    assert result.exit_code == 0
    assert "=== supervisor candidate ===" in result.output
    assert "safe_and_improved=" in result.output
    assert "top knob:" in result.output


def test_output_writes_sealed_bundle(runner: CliRunner, tmp_path: Path) -> None:
    out = tmp_path / "bundle.json"
    result = runner.invoke(
        main,
        [
            "supervisor-candidate",
            _write(tmp_path, _scenario()),
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["schema"] == "studio.supervisor_candidate.v1"
    assert len(record["digest"]) == 64
    assert record["safety_tier"] == "research"


def test_array_valued_knobs_scenario(runner: CliRunner, tmp_path: Path) -> None:
    scenario = _scenario()
    scenario["candidate"]["alpha"] = [0.2, 0.1]
    scenario["baseline"]["alpha"] = [0.0, 0.0]
    scenario["incumbent"]["alpha"] = [0.1, 0.05]
    result = runner.invoke(main, ["supervisor-candidate", _write(tmp_path, scenario)])
    assert result.exit_code == 0


def test_candidate_equal_to_baseline_omits_top_knob(
    runner: CliRunner, tmp_path: Path
) -> None:
    scenario = _scenario()
    scenario["candidate"] = dict(scenario["baseline"])
    result = runner.invoke(main, ["supervisor-candidate", _write(tmp_path, scenario)])
    assert result.exit_code == 0
    assert "top knob:" not in result.output


def test_malformed_json_errors(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    result = runner.invoke(main, ["supervisor-candidate", str(path)])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_missing_field_errors(runner: CliRunner, tmp_path: Path) -> None:
    scenario = _scenario()
    del scenario["incumbent"]
    result = runner.invoke(main, ["supervisor-candidate", _write(tmp_path, scenario)])
    assert result.exit_code != 0
    assert "incumbent" in result.output


def test_empty_observations_errors(runner: CliRunner, tmp_path: Path) -> None:
    scenario = _scenario()
    scenario["observations"] = []
    result = runner.invoke(main, ["supervisor-candidate", _write(tmp_path, scenario)])
    assert result.exit_code != 0
    assert "at least one observation" in result.output


def test_invalid_knob_value_errors(runner: CliRunner, tmp_path: Path) -> None:
    scenario = _scenario()
    scenario["candidate"]["alpha"] = "not-a-number"
    result = runner.invoke(main, ["supervisor-candidate", _write(tmp_path, scenario)])
    assert result.exit_code != 0
    assert "number" in result.output


def test_non_list_channel_weights_errors(runner: CliRunner, tmp_path: Path) -> None:
    scenario = _scenario()
    scenario["candidate"]["channel_weights"] = 1.0
    result = runner.invoke(main, ["supervisor-candidate", _write(tmp_path, scenario)])
    assert result.exit_code != 0
    assert "channel_weights must be a list" in result.output
