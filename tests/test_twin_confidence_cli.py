# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — twin-confidence CLI command tests

"""Tests for the ``spo twin-confidence`` command and its JSONL loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from scpn_phase_orchestrator.runtime.cli import main


def _tick(
    rng: np.random.Generator,
    *,
    n: int = 64,
    w: int = 16,
    phase_drift: float = 0.05,
) -> dict[str, list[float]]:
    model_phases = rng.uniform(0.0, 2.0 * np.pi, n)
    observed_phases = model_phases + rng.normal(0.0, phase_drift, n)
    model_order = rng.uniform(0.45, 0.55, w)
    observed_order = np.clip(model_order + rng.normal(0.0, 0.01, w), 0.0, 1.0)
    return {
        "model_phases": model_phases.tolist(),
        "observed_phases": observed_phases.tolist(),
        "model_order": model_order.tolist(),
        "observed_order": observed_order.tolist(),
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def streams(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(2024)
    calibration = _write_jsonl(tmp_path / "cal.jsonl", [_tick(rng) for _ in range(24)])
    observations = _write_jsonl(tmp_path / "obs.jsonl", [_tick(rng) for _ in range(6)])
    return calibration, observations


def test_twin_confidence_human_lines(streams: tuple[Path, Path]) -> None:
    calibration, observations = streams
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(observations),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ticks scored:      6" in result.output
    assert "worst status:" in result.output
    assert "status counts:" in result.output


def test_twin_confidence_json_out(streams: tuple[Path, Path]) -> None:
    calibration, observations = streams
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(observations),
            "--json-out",
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.output)
    assert record["tick_count"] == 6
    assert set(record) >= {"worst_status", "mean_confidence", "summary_hash"}


def test_twin_confidence_prometheus(streams: tuple[Path, Path]) -> None:
    calibration, observations = streams
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(observations),
            "--prometheus",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "spo_twin_confidence_mean " in result.output
    assert 'spo_twin_confidence_status_total{status="critical"}' in result.output


def test_twin_confidence_fail_on_critical(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    # Tight baseline: near-identical nominal ticks.
    calibration = _write_jsonl(
        tmp_path / "cal.jsonl", [_tick(rng, phase_drift=0.01) for _ in range(24)]
    )
    # Divergent observation: uniform model vs concentrated observed, R collapse.
    divergent = {
        "model_phases": rng.uniform(0.0, 2.0 * np.pi, 64).tolist(),
        "observed_phases": [0.1] * 64,
        "model_order": [0.5] * 16,
        "observed_order": [0.02] * 16,
    }
    observations = _write_jsonl(tmp_path / "obs.jsonl", [divergent])
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(observations),
            "--fail-on-critical",
        ],
    )
    assert result.exit_code == 2
    assert "worst status:      critical" in result.output


def test_twin_confidence_invalid_n_bins(streams: tuple[Path, Path]) -> None:
    calibration, observations = streams
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(observations),
            "--n-bins",
            "0",
        ],
    )
    assert result.exit_code != 0
    assert "n_bins" in result.output


def test_twin_confidence_malformed_json(
    tmp_path: Path, streams: tuple[Path, Path]
) -> None:
    calibration, _ = streams
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(bad),
        ],
    )
    assert result.exit_code != 0
    assert "malformed JSON" in result.output


def test_twin_confidence_missing_field(
    tmp_path: Path, streams: tuple[Path, Path]
) -> None:
    calibration, _ = streams
    incomplete = _write_jsonl(tmp_path / "inc.jsonl", [{"model_phases": [0.1, 0.2]}])
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(incomplete),
        ],
    )
    assert result.exit_code != 0
    assert "missing field" in result.output


def test_twin_confidence_non_object_tick(
    tmp_path: Path, streams: tuple[Path, Path]
) -> None:
    calibration, _ = streams
    not_object = tmp_path / "arr.jsonl"
    not_object.write_text("[1, 2, 3]\n", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(not_object),
        ],
    )
    assert result.exit_code != 0
    assert "must be a JSON object" in result.output


def test_twin_confidence_non_numeric_field(
    tmp_path: Path, streams: tuple[Path, Path]
) -> None:
    calibration, _ = streams
    nonnumeric = _write_jsonl(
        tmp_path / "nn.jsonl",
        [
            {
                "model_phases": ["a", "b"],
                "observed_phases": [0.1, 0.2],
                "model_order": [0.5],
                "observed_order": [0.5],
            }
        ],
    )
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(nonnumeric),
        ],
    )
    assert result.exit_code != 0
    assert "non-numeric" in result.output


def test_twin_confidence_empty_stream(
    tmp_path: Path, streams: tuple[Path, Path]
) -> None:
    calibration, _ = streams
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n  \n", encoding="utf-8")
    result = CliRunner().invoke(
        main,
        [
            "twin-confidence",
            "--calibration",
            str(calibration),
            "--observations",
            str(empty),
        ],
    )
    assert result.exit_code != 0
    assert "no ticks found" in result.output
