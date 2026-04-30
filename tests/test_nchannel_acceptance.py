# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — N-channel audit/report acceptance tests

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

from scpn_phase_orchestrator.audit.logger import AuditLogger
from scpn_phase_orchestrator.audit.replay import ReplayEngine
from scpn_phase_orchestrator.cli import main
from scpn_phase_orchestrator.reporting.explainability import (
    build_explainability_report,
)
from scpn_phase_orchestrator.reporting.plots import CoherencePlot
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.visualization.network import (
    coupling_heatmap_json,
    network_graph_json,
)
from scpn_phase_orchestrator.visualization.torus import (
    phase_wheel_json,
    torus_points_json,
)

CHANNEL_NAMES = ["P", "I", "S", "H", "Risk"]


def _state(values: list[float], regime: str = "NOMINAL") -> UPDEState:
    return UPDEState(
        layers=[
            LayerState(R=value, psi=0.1 * (idx + 1)) for idx, value in enumerate(values)
        ],
        cross_layer_alignment=np.eye(len(values)),
        stability_proxy=sum(values) / len(values),
        regime_id=regime,
    )


def _step(step: int, values: list[float]) -> dict[str, Any]:
    return {
        "step": step,
        "regime": "NOMINAL" if step == 0 else "DEGRADED",
        "stability": sum(values) / len(values),
        "layers": [
            {"R": value, "psi": 0.1 * (idx + 1)} for idx, value in enumerate(values)
        ],
        "actions": [],
    }


def test_audit_and_replay_preserve_five_channel_layer_records(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    binding_config = {
        "channels": {
            name: {"driver_keys": [f"{name.lower()}_signal"]} for name in CHANNEL_NAMES
        },
        "channel_groups": {
            "all_control": {"channels": CHANNEL_NAMES, "required": True},
        },
        "cross_channel_couplings": [
            {"source": "H", "target": "I", "strength": 0.35},
            {"source": "Risk", "target": "S", "strength": 0.42},
        ],
    }
    with AuditLogger(log_path) as logger:
        logger.log_header(
            n_oscillators=8,
            dt=0.01,
            binding_config=binding_config,
        )
        logger.log_step(0, _state([0.91, 0.82, 0.73, 0.64, 0.55]), [])

    replay = ReplayEngine(log_path)
    entries = replay.load()
    header = replay.load_header(entries)
    assert header is not None
    assert set(header["binding_config"]["channels"]) == set(CHANNEL_NAMES)

    reconstructed = replay.replay_step(entries[1])
    assert len(reconstructed.layers) == 5
    assert reconstructed.cross_layer_alignment.shape == (5, 5)
    assert [round(layer.R, 2) for layer in reconstructed.layers] == [
        0.91,
        0.82,
        0.73,
        0.64,
        0.55,
    ]


def test_cli_report_counts_late_appearing_fifth_channel(tmp_path: Path) -> None:
    log_path = tmp_path / "nchannel.jsonl"
    entries = [
        _step(0, [0.8, 0.7, 0.6, 0.5]),
        _step(1, [0.81, 0.71, 0.61, 0.51, 0.41]),
    ]
    log_path.write_text(
        "\n".join(json.dumps(entry) for entry in entries) + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["report", str(log_path), "--json-out"])

    assert result.exit_code == 0
    summary = json.loads(result.output)
    assert summary["layers"] == 5
    assert summary["layer_r_final"] == [0.81, 0.71, 0.61, 0.51, 0.41]


def test_reporting_extractors_keep_all_five_channels() -> None:
    steps = [
        _step(0, [0.8, 0.7, 0.6, 0.5]),
        _step(1, [0.81, 0.71, 0.61, 0.51, 0.41]),
    ]

    plot = CoherencePlot(steps)
    _, n_layers, r_series = plot._extract_r_series()
    report = build_explainability_report(steps)

    assert n_layers == 5
    assert r_series[4] == [0.0, 0.41]
    assert report.layers == 5
    assert any("Layer 4:" in line for line in report.metric_summary)


def test_visualisation_helpers_accept_five_channel_payloads() -> None:
    phases = np.linspace(0.0, 2.0 * np.pi, len(CHANNEL_NAMES), endpoint=False)
    couplings = np.array(
        [
            [0.0, 0.2, 0.1, 0.0, 0.4],
            [0.2, 0.0, 0.3, 0.5, 0.0],
            [0.1, 0.3, 0.0, 0.2, 0.6],
            [0.0, 0.5, 0.2, 0.0, 0.7],
            [0.4, 0.0, 0.6, 0.7, 0.0],
        ]
    )
    r_values = [0.91, 0.82, 0.73, 0.64, 0.55]

    graph = json.loads(network_graph_json(couplings, CHANNEL_NAMES, r_values))
    heatmap = json.loads(coupling_heatmap_json(couplings, CHANNEL_NAMES))
    wheel = json.loads(phase_wheel_json(phases, CHANNEL_NAMES))
    torus = json.loads(torus_points_json(phases, r_values))

    assert [node["name"] for node in graph["nodes"]] == CHANNEL_NAMES
    assert heatmap["labels"] == CHANNEL_NAMES
    assert [entry["name"] for entry in wheel["oscillators"]] == CHANNEL_NAMES
    assert len(torus["points"]) == len(CHANNEL_NAMES)
