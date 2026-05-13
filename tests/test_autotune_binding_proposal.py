# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — auto-binding proposal tests

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

from scpn_phase_orchestrator.binding import load_binding_spec
from scpn_phase_orchestrator.exceptions import BindingError


def _load_binding_proposal_module() -> ModuleType:
    return importlib.import_module("scpn_phase_orchestrator.autotune.binding_proposal")


binding_proposal = _load_binding_proposal_module()
propose_binding_from_event_log = binding_proposal.propose_binding_from_event_log
propose_binding_from_graph = binding_proposal.propose_binding_from_graph
propose_binding_from_time_series_csv = (
    binding_proposal.propose_binding_from_time_series_csv
)


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


def test_time_series_csv_proposal_records_discovery_evidence() -> None:
    csv_text = "\n".join(
        [
            "time,source,driven,independent",
            "0.00,0.00,0.00,1.00",
            "0.10,0.20,0.10,0.98",
            "0.20,0.40,0.20,0.92",
            "0.30,0.60,0.31,0.83",
            "0.40,0.78,0.42,0.70",
            "0.50,0.94,0.54,0.54",
            "0.60,1.07,0.66,0.36",
            "0.70,1.17,0.77,0.17",
        ]
    )

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=10.0,
        project_name="discovered_replay",
    )

    discovery = proposal.binding.provenance["discovery_evidence"]
    assert discovery["sample_period_s"] == pytest.approx(0.1)
    assert discovery["columns"] == ("source", "driven", "independent")
    assert discovery["sindy"]["active_terms"] > 0
    assert discovery["sindy"]["library"] == "affine_state_derivative"
    assert discovery["correlation_graph"]["edge_count"] >= 1
    assert discovery["clustering"]["cluster_count"] >= 1
    assert "sindy_sparsity" in proposal.binding.confidence_factors
    assert "correlation_graph_density" in proposal.binding.confidence_factors


def test_time_series_csv_infers_sample_rate_from_time_column() -> None:
    csv_text = "time,grid,load\n0.0,0.0,1.0\n0.2,0.2,0.8\n0.4,0.4,0.6\n"

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=None,
        project_name="inferred_rate_replay",
    )

    assert proposal.binding.provenance["sample_rate_hz"] == pytest.approx(5.0)
    assert proposal.binding.provenance["sample_rate_inference"] == "time_column"
    assert "sample_period_s: 0.2" in proposal.binding.yaml_text


def test_time_series_csv_yaml_binds_layer_families_in_channel_order(
    tmp_path: Path,
) -> None:
    csv_text = (
        "time,grid,load,breaker\n0.00,0.0,1.0,0.0\n0.01,0.2,0.9,1.0\n0.02,0.4,0.7,0.0\n"
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


@pytest.mark.parametrize(
    ("csv_text", "sample_rate_hz", "expected_error"),
    [
        ("time,grid\n0.0,1.0\n", 0.0, "sample_rate_hz"),
        ("", 100.0, "CSV header"),
        ("time\n0.0\n", 100.0, "signal channel"),
        ("time,grid\n", 100.0, "at least one sample"),
        ("time,grid\n0.0,not-a-number\n", 100.0, "non-numeric sample"),
    ],
)
def test_time_series_csv_rejects_invalid_replay_shapes(
    csv_text: str,
    sample_rate_hz: float | None,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        propose_binding_from_time_series_csv(
            csv_text,
            sample_rate_hz=sample_rate_hz,
            project_name="invalid_replay",
        )


def test_time_series_csv_rejects_missing_rate_without_time_column() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        propose_binding_from_time_series_csv(
            "grid,load\n0.0,1.0\n0.2,0.8\n",
            sample_rate_hz=None,
            project_name="missing_rate_replay",
        )


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


@pytest.mark.parametrize(
    ("events", "expected_error"),
    [
        ([], "at least one event"),
        ([{"time": 0.0, "source": "breaker"}], "event field"),
        ([1], "event log"),
    ],
)
def test_event_log_rejects_non_reviewable_event_streams(
    events: object,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        propose_binding_from_event_log(
            json.dumps(events),
            project_name="invalid_event_replay",
        )


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


@pytest.mark.parametrize(
    ("graph", "expected_error"),
    [
        ([], "graph must be a mapping"),
        ({"nodes": []}, "at least one node"),
        ({"nodes": [{"label": "missing-id"}]}, "non-empty string id"),
        ({"nodes": [{"id": ""}]}, "non-empty string id"),
        ({"nodes": [{"id": "a"}], "edges": "a->b"}, "graph.edges"),
    ],
)
def test_graph_proposal_rejects_invalid_topology_payloads(
    graph: object,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):
        propose_binding_from_graph(json.dumps(graph), project_name="invalid_graph")


def test_proposal_requires_non_empty_project_name() -> None:
    csv_text = "time,grid\n0.00,0.0\n0.01,0.2\n"

    with pytest.raises(ValueError, match="project_name"):
        propose_binding_from_time_series_csv(
            csv_text,
            sample_rate_hz=100.0,
            project_name="",
        )


def test_validation_errors_are_preserved_when_generated_yaml_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_generated_yaml(path: Path) -> object:
        raise BindingError(f"invalid generated binding: {path.name}")

    monkeypatch.setattr(binding_proposal, "load_binding_spec", reject_generated_yaml)

    proposal = propose_binding_from_graph(
        json.dumps({"nodes": [{"id": "a"}], "edges": []}),
        project_name="graph_replay",
    )

    assert proposal.binding.validation_errors == (
        "invalid generated binding: binding_spec.yaml",
    )
    assert proposal.binding.confidence_factors["validator_acceptance"] == 0.0


def _load_proposal_yaml(tmp_path: Path, yaml_text: str):
    path = tmp_path / "binding_spec.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return load_binding_spec(path)
