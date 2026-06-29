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
