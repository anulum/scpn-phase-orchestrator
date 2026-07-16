# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — auto-bind / auto-coupling CLI branch tests

"""Branch tests for the binding CLI auto-bind and auto-coupling commands.

Covers the event-log and graph source-kind dispatch branches, the OSError
read-failure path, the ``.npy`` phase-series loader branch, and the
auto-coupling estimation error handler.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.binding as _binding
from scpn_phase_orchestrator.runtime.cli._app import main

assert _binding is not None


@pytest.fixture
def runner() -> CliRunner:
    """Return a Click runner."""
    return CliRunner()


def _phase_series() -> np.ndarray:
    """Return a small valid oscillator-by-time phase series."""
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 2.0 * np.pi, (3, 64))


def test_auto_bind_event_log_emits_yaml(runner: CliRunner, tmp_path: Path) -> None:
    events = [
        {"time": 0.0, "source": "sensor_a", "event": "open"},
        {"time": 0.5, "source": "sensor_b", "event": "close"},
        {"time": 1.0, "source": "sensor_c", "event": "trip"},
    ]
    source = tmp_path / "events.json"
    source.write_text(json.dumps(events), encoding="utf-8")

    result = runner.invoke(
        main,
        ["auto-bind", "event-log-json", str(source), "--project-name", "evt"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() != ""


def test_auto_bind_graph_emits_yaml(runner: CliRunner, tmp_path: Path) -> None:
    graph = {"nodes": [{"id": "a"}, {"id": "b"}], "edges": []}
    source = tmp_path / "graph.json"
    source.write_text(json.dumps(graph), encoding="utf-8")

    result = runner.invoke(
        main,
        ["auto-bind", "graph-json", str(source), "--project-name", "grf"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() != ""


def test_auto_bind_reports_an_os_read_error(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "events.json"
    source.write_text(json.dumps([{"time": 0.0, "source": "a", "event": "x"}]), "utf-8")

    def _raise(self: Path, *args: object, **kwargs: object) -> str:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(Path, "read_text", _raise)

    result = runner.invoke(
        main,
        ["auto-bind", "event-log-json", str(source), "--project-name", "evt"],
    )

    assert result.exit_code == 1
    assert "could not read source file" in result.output


def test_auto_coupling_loads_a_npy_table(runner: CliRunner, tmp_path: Path) -> None:
    source = tmp_path / "series.npy"
    np.save(source, _phase_series())

    result = runner.invoke(main, ["auto-coupling-estimation", str(source)])

    assert result.exit_code == 0, result.output
    assert "auto-coupling-estimation method=" in result.output


def test_auto_coupling_rejects_a_degenerate_series(
    runner: CliRunner, tmp_path: Path
) -> None:
    # A single-oscillator table is a valid 2-D file, so it clears the loader, but
    # the estimator needs at least two oscillators and raises, and the CLI then
    # fails closed through its ClickException handler.
    source = tmp_path / "series.npy"
    np.save(source, np.zeros((1, 8), dtype=np.float64))

    result = runner.invoke(main, ["auto-coupling-estimation", str(source)])

    assert result.exit_code != 0
    assert "at least 2 oscillators" in result.output


def _planted_kuramoto_csv(tmp_path: Path) -> Path:
    """Write a planted two-node Kuramoto trajectory CSV and return its path."""
    omega = (1.0, 1.3)
    coupling = 0.8
    dt = 0.02
    state = np.asarray([0.0, 0.4], dtype=np.float64)
    lines = ["theta_0,theta_1"]
    for _ in range(260):
        lines.append(f"{state[0]:.6f},{state[1]:.6f}")
        drift = np.asarray(
            [
                omega[0] + coupling * np.sin(state[1] - state[0]),
                omega[1] + coupling * np.sin(state[0] - state[1]),
            ]
        )
        state = state + dt * drift
    source = tmp_path / "kuramoto.csv"
    source.write_text("\n".join(lines), encoding="utf-8")
    return source


def test_auto_bind_emit_equations_prints_discovered_dynamics(
    runner: CliRunner, tmp_path: Path
) -> None:
    source = _planted_kuramoto_csv(tmp_path)

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(source),
            "--project-name",
            "kuramoto",
            "--sample-rate-hz",
            "50",
            "--emit-equations",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Discovered dynamics (kuramoto_sine_phase_differences)" in result.output
    assert "posture: discovered" in result.output
    assert "tier: partial" in result.output
    # Never advertise a self-fit as externally validated.
    assert "externally_validated" not in result.output
    assert "sin(theta_1 - theta_0)" in result.output
    assert (
        "theta_1 -> theta_0" in result.output or "theta_0 -> theta_1" in result.output
    )


def test_auto_bind_emit_equations_reports_absence_for_non_csv_kinds(
    runner: CliRunner, tmp_path: Path
) -> None:
    events = [{"time": 0.0, "source": "a", "event": "x"}]
    source = tmp_path / "events.json"
    source.write_text(json.dumps(events), encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "event-log-json",
            str(source),
            "--project-name",
            "evt",
            "--emit-equations",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "No discovered phase dynamics" in result.output


def test_auto_bind_strict_confidence_flags_downgrade_the_posture(
    runner: CliRunner, tmp_path: Path
) -> None:
    source = _planted_kuramoto_csv(tmp_path)

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(source),
            "--project-name",
            "kuramoto",
            "--sample-rate-hz",
            "50",
            "--sindy-min-samples-per-parameter",
            "10000",
            "--emit-equations",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "posture: discovered" not in result.output
    assert "tier: scaffold" in result.output


def test_auto_bind_json_out_pins_the_discovered_dynamics_schema(
    runner: CliRunner, tmp_path: Path
) -> None:
    # Golden drift guard: pin the discovered-dynamics record SHAPE (the key set
    # and stable categorical values), not brittle float strings, so a schema
    # change is caught while a numeric change is not a false alarm.
    source = _planted_kuramoto_csv(tmp_path)

    result = runner.invoke(
        main,
        [
            "auto-bind",
            "time-series-csv",
            str(source),
            "--project-name",
            "kuramoto",
            "--sample-rate-hz",
            "50",
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.output
    audit = json.loads(result.output)
    record = audit["binding"]["provenance"]["discovered_dynamics"]
    assert set(record) == {
        "library",
        "status",
        "equations",
        "coupling_edges",
        "confidence",
        "content_hash",
    }
    assert set(record["confidence"]) == {
        "tier",
        "posture",
        "r_squared",
        "samples_per_parameter",
        "reasons",
    }
    assert record["library"] == "kuramoto_sine_phase_differences"
    assert record["status"] == "fitted"
    assert record["confidence"]["tier"] == "partial"
    assert record["confidence"]["posture"] == "discovered"
    assert len(record["content_hash"]) == 64
    assert set(record["coupling_edges"][0]) == {
        "source",
        "target",
        "coefficient",
        "abs_coefficient",
    }
    options = audit["binding"]["provenance"]["sindy_options"]
    assert options["phase_sindy_threshold"] == 0.05
    assert set(options["confidence_policy"]) == {
        "min_r_squared",
        "min_samples_per_parameter",
    }
