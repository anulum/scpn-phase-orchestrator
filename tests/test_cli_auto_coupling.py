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
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_auto_coupling_estimation_emits_deterministic_text_matrix_output(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "phases.csv"
    source_path.write_text("0.0,0.2,0.4,0.6\n0.1,0.3,0.5,0.7\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "auto-coupling-estimation",
            str(source_path),
        ],
    )

    assert result.exit_code == 0
    lines = result.output.splitlines()
    assert len(lines) >= 3
    assert lines[0].startswith("auto-coupling-estimation method=")
    matrix_lines = lines[1:]
    assert len(matrix_lines) == 2
    for matrix_line in matrix_lines:
        values = [float(value) for value in matrix_line.split(",")]
        assert len(values) == 2


def test_auto_coupling_estimation_json_output_is_structured(
    runner: CliRunner, tmp_path: Path
) -> None:
    source_path = tmp_path / "phases.csv"
    source_path.write_text("0.0,0.2,0.4,0.6\n0.1,0.3,0.5,0.7\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "auto-coupling-estimation",
            str(source_path),
            "--json-out",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    assert payload.get("orientation") == "source_to_target"
    assert isinstance(payload.get("shape"), list)
    assert len(payload["shape"]) == 2


def test_auto_coupling_estimation_rejects_flat_phase_series(
    runner: CliRunner, tmp_path: Path
) -> None:
    source_path = tmp_path / "phases.csv"
    source_path.write_text("0.0,0.1,0.2\n", encoding="utf-8")

    result = runner.invoke(main, ["auto-coupling-estimation", str(source_path)])

    assert result.exit_code == 1
    assert "phase-series source must be a 2-D table" in result.output


def test_auto_coupling_estimation_rejects_non_numeric_phase_series(
    runner: CliRunner, tmp_path: Path
) -> None:
    source_path = tmp_path / "phases.csv"
    source_path.write_text("0.0,abc\n1.0,2.0\n", encoding="utf-8")

    result = runner.invoke(main, ["auto-coupling-estimation", str(source_path)])

    assert result.exit_code == 1
    assert "could not read numeric phase-series data" in result.output
