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
from typing import Any

import numpy as np
import pytest
import yaml

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


def test_time_series_csv_records_phase_sindy_evidence() -> None:
    csv_text = "\n".join(
        [
            "time,theta_source,theta_driven",
            "0.0,0.00,0.20",
            "0.1,0.10,0.26",
            "0.2,0.20,0.34",
            "0.3,0.30,0.43",
            "0.4,0.40,0.53",
            "0.5,0.50,0.64",
            "0.6,0.60,0.76",
            "0.7,0.70,0.88",
        ]
    )

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=None,
        project_name="phase_replay",
    )
    discovery = proposal.binding.provenance["discovery_evidence"]

    assert discovery["phase_sindy"]["status"] == "fitted"
    assert discovery["phase_sindy"]["library"] == "kuramoto_sine_phase_differences"
    assert discovery["phase_sindy"]["coupling_edge_count"] >= 1
    assert discovery["sindy_model_selection"]["candidate_count"] == 2
    assert discovery["sindy_model_selection"]["selected_library"] in {
        "affine_state_derivative",
        "kuramoto_sine_phase_differences",
    }
    assert discovery["learned_graph"]["status"] == "fitted"
    assert "learned_graph_density" in proposal.binding.confidence_factors
    assert "phase_sindy_sparsity" in proposal.binding.confidence_factors


def _planted_kuramoto_csv() -> str:
    """Return a CSV of a planted two-node Kuramoto trajectory (a simulation)."""
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
    return "\n".join(lines)


def test_time_series_csv_surfaces_discovered_dynamics_in_provenance() -> None:
    from scpn_phase_orchestrator.autotune.sindy_options import SindyOptions

    proposal = propose_binding_from_time_series_csv(
        _planted_kuramoto_csv(),
        sample_rate_hz=50.0,
        project_name="kuramoto_replay",
        sindy_options=SindyOptions(phase_sindy_threshold=0.05),
    )
    provenance = proposal.binding.provenance

    record = provenance["discovered_dynamics"]
    assert record["confidence"]["posture"] == "discovered"
    assert record["confidence"]["tier"] == "partial"
    # A self-fit can never be surfaced as externally validated.
    assert record["confidence"]["tier"] != "externally_validated"
    assert len(record["content_hash"]) == 64
    assert record["equations"]
    assert record["coupling_edges"]

    options_record = provenance["sindy_options"]
    assert options_record["phase_sindy_threshold"] == 0.05
    assert options_record["confidence_policy"]["min_r_squared"] == 0.9
    assert options_record["confidence_policy"]["min_samples_per_parameter"] == 5.0


def test_time_series_csv_confidence_policy_downgrades_discovery_posture() -> None:
    from scpn_phase_orchestrator.autotune.sindy_confidence import SindyConfidencePolicy
    from scpn_phase_orchestrator.autotune.sindy_options import SindyOptions

    csv_text = _planted_kuramoto_csv()
    strict = SindyOptions(
        confidence_policy=SindyConfidencePolicy(min_samples_per_parameter=10_000.0)
    )

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=50.0,
        project_name="strict_replay",
        sindy_options=strict,
    )
    record = proposal.binding.provenance["discovered_dynamics"]

    # An unreachable determination threshold must downgrade the very same fit.
    assert record["confidence"]["posture"] != "discovered"
    assert record["confidence"]["tier"] == "scaffold"
    assert record["confidence"]["reasons"]


def test_time_series_csv_defaults_to_shared_options_when_unspecified() -> None:
    proposal = propose_binding_from_time_series_csv(
        _planted_kuramoto_csv(),
        sample_rate_hz=50.0,
        project_name="default_options_replay",
    )
    options_record = proposal.binding.provenance["sindy_options"]

    assert options_record["phase_sindy_threshold"] == 0.05
    assert "discovered_dynamics" in proposal.binding.provenance


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


def test_time_series_csv_yaml_binds_extractor_parameter_proposals(
    tmp_path: Path,
) -> None:
    csv_text = "time,grid,load\n0.00,0.0,1.0\n0.01,0.2,0.9\n0.02,0.4,0.7\n"

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=100.0,
        project_name="grid_replay",
    )

    spec = _load_proposal_yaml(tmp_path, proposal.binding.yaml_text)
    family = spec.oscillator_families["auto_p_0"]
    extractor_proposals = proposal.binding.provenance["extractor_parameter_proposals"]

    assert family.config["source_column"] == "grid"
    assert family.config["source_column_index"] == 0
    assert family.config["sample_rate_hz"] == pytest.approx(100.0)
    assert family.config["sample_period_s"] == pytest.approx(0.01)
    assert family.config["normalisation"] == "zscore"
    assert family.config["proposal_status"] == "review_only"
    assert extractor_proposals[0]["family"] == "auto_p_0"
    assert extractor_proposals[0]["parameters"]["source_column"] == "grid"


def test_time_series_csv_yaml_quotes_injection_prone_column_names(
    tmp_path: Path,
) -> None:
    csv_text = (
        'time,"grid: evil","load # comment","break\n'
        'line"\n0.00,0.0,1.0,0.0\n0.01,0.2,0.9,1.0\n'
    )

    proposal = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=100.0,
        project_name="yaml_injection_review",
    )

    parsed = yaml.safe_load(proposal.binding.yaml_text)
    spec = _load_proposal_yaml(tmp_path, proposal.binding.yaml_text)
    config_values = [
        parsed["oscillator_families"][layer.family]["config"]["source_column"]
        for layer in sorted(spec.layers, key=lambda item: item.index)
    ]

    assert config_values == ["grid: evil", "load # comment", "break\nline"]
    assert proposal.binding.validation_errors == ()


def test_time_series_csv_yaml_binds_initial_k_template(
    tmp_path: Path,
) -> None:
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
        project_name="k_replay",
    )

    spec = _load_proposal_yaml(tmp_path, proposal.binding.yaml_text)
    initial_k = proposal.binding.provenance["initial_coupling_proposal"]

    assert "auto_initial_k" in spec.coupling.templates
    initial_k_matrix = [list(row) for row in initial_k["matrix"]]
    assert spec.coupling.templates["auto_initial_k"]["matrix"] == initial_k_matrix
    assert spec.coupling.templates["auto_initial_k"]["orientation"] == (
        "target_by_source"
    )
    assert spec.cross_channel_couplings
    assert {edge.template for edge in spec.cross_channel_couplings} == {
        "auto_initial_k"
    }
    assert list(initial_k["columns"]) == ["source", "driven", "independent"]
    matrix = initial_k_matrix
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
    assert all(row[index] == 0.0 for index, row in enumerate(matrix))
    assert max(value for row in matrix for value in row) > 0.0


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


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_time_series_csv_rejects_non_finite_samples(value: str) -> None:
    csv_text = f"time,grid,load\n0.0,{value},1.0\n0.1,2.0,3.0\n"

    with pytest.raises(ValueError, match="non-finite sample"):
        propose_binding_from_time_series_csv(
            csv_text,
            sample_rate_hz=10.0,
            project_name="invalid_non_finite_replay",
        )


def test_time_series_csv_rejects_missing_rate_without_time_column() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        propose_binding_from_time_series_csv(
            "grid,load\n0.0,1.0\n0.2,0.8\n",
            sample_rate_hz=None,
            project_name="missing_rate_replay",
        )


def test_time_series_csv_deterministic_candidate_ordering_and_output(
    tmp_path: Path,
) -> None:
    csv_text = "\n".join(
        [
            "time,grid,load,event,ring,sensor",
            "0.0,0.0,1.0,0.0,1.0,2.0",
            "0.1,0.2,0.9,0.1,0.9,1.8",
            "0.2,0.4,0.8,0.2,0.8,1.6",
        ]
    )

    first = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=10.0,
        project_name="candidate_order_replay",
    )
    second = propose_binding_from_time_series_csv(
        csv_text,
        sample_rate_hz=10.0,
        project_name="candidate_order_replay",
    )

    assert first.binding.yaml_sha256 == second.binding.yaml_sha256
    assert first.binding.inferred_channels == ("P", "I", "S", "P", "I")
    assert tuple(
        proposal_channel["channel"]
        for proposal_channel in first.binding.provenance[
            "extractor_parameter_proposals"
        ]
    ) == ("P", "I", "S", "P", "I")

    spec = _load_proposal_yaml(tmp_path, first.binding.yaml_text)
    layer_family_order = tuple(
        spec.oscillator_families[layer.family].channel
        for layer in sorted(spec.layers, key=lambda item: item.index)
    )
    assert layer_family_order == first.binding.inferred_channels


def test_event_log_scoring_reflects_diversity_and_span() -> None:
    events = [
        {"time": 0.0, "source": "sensor_b", "event": "open"},
        {"time": 0.5, "source": "sensor_a", "event": "close"},
        {"time": 1.0, "source": "sensor_c", "event": "trip"},
        {"time": 1.25, "source": "sensor_b", "event": "reset"},
    ]

    proposal = propose_binding_from_event_log(
        json.dumps(events),
        project_name="event_scoring_replay",
    )

    assert proposal.binding.confidence_factors["event_density"] == pytest.approx(0.4)
    assert proposal.binding.confidence_factors["source_diversity"] == pytest.approx(1.0)
    assert proposal.binding.provenance["source_names"] == (
        "sensor_a",
        "sensor_b",
        "sensor_c",
    )
    assert proposal.binding.provenance["time_span_s"] == pytest.approx(1.25)


def test_event_log_proposal_is_deterministic_for_duplicate_sources() -> None:
    events = [
        {"time": 2.0, "source": "sensor_z", "event": "trip"},
        {"time": 0.0, "source": "sensor_a", "event": "open"},
        {"time": 1.0, "source": "sensor_z", "event": "reset"},
        {"time": 1.5, "source": "sensor_b", "event": "close"},
    ]

    first = propose_binding_from_event_log(
        json.dumps(events),
        project_name="event_determinism_replay",
    )
    second = propose_binding_from_event_log(
        json.dumps(events),
        project_name="event_determinism_replay",
    )

    assert first.binding.yaml_sha256 == second.binding.yaml_sha256
    assert first.binding.provenance["source_names"] == (
        "sensor_a",
        "sensor_b",
        "sensor_z",
    )
    assert (
        first.binding.confidence_factors["event_density"]
        == (second.binding.confidence_factors["event_density"])
    )


def test_event_log_rejects_malformed_json_payload() -> None:
    with pytest.raises(json.JSONDecodeError):
        propose_binding_from_event_log(
            "{not-valid-json",
            project_name="bad_event_json",
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


@pytest.mark.parametrize("mode", ["time_series", "event_log", "graph"])
def test_proposal_output_is_review_only_and_non_actuating(mode: str) -> None:
    if mode == "time_series":
        proposal = propose_binding_from_time_series_csv(
            "time,grid,load\n0.0,0.0,1.0\n0.01,0.2,0.9\n",
            sample_rate_hz=100.0,
            project_name="review_only_ts",
        )
    elif mode == "event_log":
        proposal = propose_binding_from_event_log(
            json.dumps([{"time": 0.0, "source": "breaker", "event": "open"}]),
            project_name="review_only_events",
        )
    else:
        proposal = propose_binding_from_graph(
            json.dumps({"nodes": [{"id": "a"}], "edges": []}),
            project_name="review_only_graph",
        )

    assert proposal.runtime.regime == "proposal_only"
    assert proposal.runtime.replay_status == "proposal_only"
    assert proposal.runtime.K == 0.45
    assert proposal.metadata["proposal_mode"] == "review_only"


def test_validate_binding_spec_rejection_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        binding_proposal,
        "validate_binding_spec",
        lambda _spec: ("synthetic validator finding",),
    )

    proposal = propose_binding_from_graph(
        json.dumps({"nodes": [{"id": "a"}], "edges": []}),
        project_name="synthetic_validation_replay",
    )

    assert proposal.binding.validation_errors == ("synthetic validator finding",)
    assert proposal.binding.confidence_factors["validator_acceptance"] == 0.0


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


def test_graph_proposal_edge_density_is_clamped_to_one() -> None:
    graph = {
        "nodes": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
        "edges": [
            {"source": "a", "target": "b"},
            {"source": "b", "target": "c"},
            {"source": "c", "target": "a"},
            {"source": "a", "target": "c"},
            {"source": "b", "target": "a"},
        ],
    }

    proposal = propose_binding_from_graph(
        json.dumps(graph),
        project_name="dense_graph_replay",
    )

    assert proposal.binding.confidence_factors["edge_density"] == 1.0
    assert proposal.binding.confidence_factors["topology_integrity"] == 1.0
    assert proposal.binding.provenance["node_count"] == 3
    assert proposal.binding.provenance["edge_count"] == 5


def test_graph_proposal_is_deterministic_for_repeated_submissions() -> None:
    graph = {
        "nodes": [{"id": "z"}, {"id": "a"}, {"id": "m"}],
        "edges": [
            {"source": "z", "target": "a"},
            {"source": "a", "target": "m"},
            {"source": "m", "target": "z"},
            {"source": "z", "target": "m"},
        ],
    }

    first = propose_binding_from_graph(
        json.dumps(graph),
        project_name="graph_determinism_replay",
    )
    second = propose_binding_from_graph(
        json.dumps(graph),
        project_name="graph_determinism_replay",
    )

    assert first.binding.yaml_sha256 == second.binding.yaml_sha256
    assert first.binding.yaml_text == second.binding.yaml_text
    assert first.binding.confidence_factors["topology_integrity"] == 1.0


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


def test_inferred_channels_rotation_is_deterministic():
    assert binding_proposal._inferred_channels(5, prefer_event=True) == (
        "I",
        "P",
        "S",
        "I",
        "P",
    )
    assert binding_proposal._inferred_channels(5, prefer_event=False) == (
        "P",
        "I",
        "S",
        "P",
        "I",
    )


def test_families_for_channels_maps_unknown_channels_to_physical_extractor():
    families = binding_proposal._families_for_channels(("P", "noise", "S", ""))
    assert families == (
        ("auto_p_0", "P", "physical"),
        ("auto_noise_1", "noise", "physical"),
        ("auto_s_2", "S", "ring"),
        ("auto__3", "", "physical"),
    )


def test_resolve_sample_rate_hz_validates_explicit_rate_inputs():
    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        binding_proposal._resolve_sample_rate_hz(
            0.0,
            rows=[],
            fieldnames=["time", "value"],
        )

    with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
        binding_proposal._resolve_sample_rate_hz(
            float("nan"),
            rows=[],
            fieldnames=["time", "value"],
        )


def test_cross_channel_couplings_filters_invalid_entries():
    coupling = {
        "edges": (
            {"source_channel": "P", "target_channel": "I", "strength": 0.2},
            {"source_channel": "I", "target_channel": "", "strength": 0.5},
            {"source_channel": "P", "target_channel": "S", "strength": "0.9"},
            {"source_channel": "I", "target_channel": "P", "strength": -0.3},
            {"source_channel": "P", "target_channel": "I", "strength": 0.2},
        )
    }

    couplings = binding_proposal._cross_channel_couplings(coupling)

    assert couplings == [
        {
            "source": "P",
            "target": "I",
            "strength": 0.2,
            "mode": "directed",
            "template": "auto_initial_k",
        }
    ]


def test_edge_sequence_returns_empty_for_malformed_payload() -> None:
    assert binding_proposal._edge_sequence({"sec": "not-a-mapping"}, "sec") == ()
    assert binding_proposal._edge_sequence({"sec": {"edges": "abc"}}, "sec") == ()


def test_accumulate_directed_edges_skips_malformed_edges() -> None:
    matrix = np.zeros((2, 2), dtype=np.float64)
    column_index = {"a": 0, "b": 1}
    edges: list[dict[str, Any]] = [
        {"source": 1, "target": "b", "w": 0.5},
        {"source": "x", "target": "b", "w": 0.5},
        {"source": "a", "target": "b", "w": True},
        {"source": "a", "target": "b", "w": float("inf")},
    ]

    binding_proposal._accumulate_directed_edges(
        matrix, column_index=column_index, edges=edges, weight_field="w"
    )

    assert float(matrix.sum()) == 0.0


def test_accumulate_correlation_edges_skips_malformed_edges() -> None:
    matrix = np.zeros((2, 2), dtype=np.float64)
    column_index = {"a": 0, "b": 1}
    edges: list[dict[str, Any]] = [
        {"source": 1, "target": "b", "abs_correlation": 0.5},
        {"source": "x", "target": "b", "abs_correlation": 0.5},
        {"source": "a", "target": "b", "abs_correlation": True},
        {"source": "a", "target": "b", "abs_correlation": float("nan")},
    ]

    binding_proposal._accumulate_correlation_edges(
        matrix, column_index=column_index, edges=edges
    )

    assert float(matrix.sum()) == 0.0


def test_cross_channel_couplings_skips_malformed_edges() -> None:
    proposal: dict[str, Any] = {
        "edges": [
            "not-a-mapping",
            {"source_channel": 1, "target_channel": "b", "strength": 0.5},
        ]
    }

    assert binding_proposal._cross_channel_couplings(proposal) == []
