# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — auto-binding proposal tests

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
)
from scpn_phase_orchestrator.binding import load_binding_spec


def test_time_series_csv_proposal_is_reviewable_and_validated() -> None:
    csv_text = "time,grid,load\n0.00,0.0,1.0\n0.01,0.2,0.9\n0.02,0.4,0.7\n"

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=100.0,
        project_name="grid_replay",
    )

    assert proposal.source.source_kind == "time_series_csv"
    assert proposal.source.channel_count == 2
    assert proposal.source.sample_count == 3
    assert "grid_replay" in proposal.binding.yaml_text
    assert proposal.binding.inferred_channels == ("P", "I")
    assert "phase_quality" in proposal.binding.confidence_factors
    assert proposal.binding.validation_errors == ()


def test_time_series_csv_yaml_binds_layer_families_in_channel_order(
    tmp_path: Path,
) -> None:
    csv_text = (
        "time,grid,load,breaker\n"
        "0.00,0.0,1.0,0.0\n"
        "0.01,0.2,0.9,1.0\n"
        "0.02,0.4,0.7,0.0\n"
    )

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=100.0,
        project_name="grid_replay",
    )

    spec = _load_proposal_yaml(tmp_path, proposal.binding.yaml_text)
    layer_channels = tuple(
        spec.oscillator_families[layer.family].channel
        for layer in sorted(spec.layers, key=lambda item: item.index)
    )

    assert layer_channels == proposal.binding.inferred_channels
    assert spec.objectives.good_layers == [0, 1, 2]


def test_event_log_proposal_records_low_confidence_for_sparse_sources() -> None:
    events = [
        {"time": 0.0, "source": "breaker", "event": "open"},
        {"time": 2.0, "source": "breaker", "event": "close"},
    ]

    proposal = propose_binding_from_event_log(
        json.dumps(events),
        project_name="event_replay",
    )

    assert proposal.source.source_kind == "event_log_json"
    assert proposal.source.event_count == 2
    assert proposal.binding.confidence_factors["event_density"] < 0.5
    assert proposal.binding.provenance["input_family"] == "event_log"


def test_event_log_yaml_binds_event_family_to_layer(tmp_path: Path) -> None:
    events = [
        {"time": 0.0, "source": "breaker", "event": "open"},
        {"time": 2.0, "source": "breaker", "event": "close"},
    ]

    proposal = propose_binding_from_event_log(
        json.dumps(events),
        project_name="event_replay",
    )

    spec = _load_proposal_yaml(tmp_path, proposal.binding.yaml_text)
    layer = spec.layers[0]

    assert layer.family is not None
    assert spec.oscillator_families[layer.family].channel == "I"


def test_graph_proposal_rejects_edges_with_unknown_nodes() -> None:
    graph = {
        "nodes": [{"id": "a"}],
        "edges": [{"source": "a", "target": "missing"}],
    }

    with pytest.raises(ValueError, match="unknown graph node"):
        propose_binding_from_graph(json.dumps(graph), project_name="bad_graph")


def test_graph_yaml_binds_graph_family_to_layer(tmp_path: Path) -> None:
    graph = {
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [{"source": "a", "target": "b"}],
    }

    proposal = propose_binding_from_graph(
        json.dumps(graph),
        project_name="graph_replay",
    )

    spec = _load_proposal_yaml(tmp_path, proposal.binding.yaml_text)
    layer = spec.layers[0]

    assert layer.family is not None
    assert spec.oscillator_families[layer.family].channel == "S"


def _load_proposal_yaml(tmp_path: Path, yaml_text: str):
    path = tmp_path / "binding_spec.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return load_binding_spec(path)
