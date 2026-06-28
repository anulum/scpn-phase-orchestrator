# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Evolutionary grammar CLI contract tests

"""CLI contract tests for the evolutionary supervisor grammar commands."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import scpn_phase_orchestrator.runtime.cli.evolutionary_grammar as _grammar_cli
from scpn_phase_orchestrator.runtime.cli._app import main

assert _grammar_cli is not None

_SAMPLE_DSL = (
    "rule throttle_guard: if R < 0.95 and K > 0.10 then set K += 0.04\n"
    "rule recovery_guard: if R >= 0.20 then set K -= 0.03\n"
    "rule safety_guard: if R < 0.40 then set K = 0.12\n"
)


def _net_mapping() -> dict[str, object]:
    """Return a valid net-like mapping payload."""
    return {
        "places": [
            {"name": "idle", "token_bound": 2},
            {"name": "nominal", "token_bound": 4},
        ],
        "transitions": [
            {"name": "to_nominal", "guard_weights": {"R": 0.8}},
            {"name": "to_degraded", "guard_weights": {"R": 0.2}},
        ],
        "arcs": [
            {
                "place": "idle",
                "transition": "to_nominal",
                "direction": "input",
                "weight": 1,
            }
        ],
    }


def _net_records() -> list[dict[str, object]]:
    """Return a valid net-like record-array payload."""
    return [
        {"kind": "place", "name": "idle", "token_bound": 2},
        {"kind": "place", "name": "nominal", "token_bound": 3},
        {"kind": "transition", "name": "to_nominal", "guard_weights": {"R": 0.5}},
        {
            "kind": "arc",
            "place": "idle",
            "transition": "to_nominal",
            "direction": "input",
            "weight": 1,
        },
    ]


def _topology() -> dict[str, object]:
    """Return a valid topology payload with nodes and edges."""
    return {
        "nodes": [
            {"node_id": 0, "community": "alpha"},
            {"node_id": 1, "community": "alpha"},
            {"node_id": 2, "community": "beta"},
        ],
        "edges": [{"nodes": [0, 1], "weight": 0.28}],
    }


def _write_text(path: Path, text: str) -> Path:
    """Write text content to ``path``."""
    path.write_text(text, encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    """Write a JSON payload to ``path``."""
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _run(args: Sequence[str]) -> object:
    """Invoke the CLI with ``args``."""
    return CliRunner().invoke(main, list(args))


def test_policy_dsl_search_emits_review_bundle(tmp_path: Path) -> None:
    dsl_path = _write_text(tmp_path / "policy.dsl", _SAMPLE_DSL)
    output_path = tmp_path / "bundle.json"

    result = _run(
        [
            "evolutionary-policy-dsl-search",
            str(dsl_path),
            "--generations",
            "2",
            "--population",
            "4",
            "--output",
            str(output_path),
        ]
    )

    assert result.exit_code == 0, result.output
    stdout_payload = json.loads(result.output)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_payload == written
    assert stdout_payload["schema"] == "scpn_evolutionary_grammar_review_bundle_v1"
    assert stdout_payload["version"] == "1.0.0"
    assert stdout_payload["grammar"] == "policy-dsl"
    assert stdout_payload["non_actuating"] is True
    assert stdout_payload["actuation_permitted"] is False
    assert stdout_payload["operator_review_required"] is True
    assert len(stdout_payload["bundle_hash"]) == 64
    assert stdout_payload["report"]["candidate_count"] >= 1


def test_policy_dsl_search_is_deterministic(tmp_path: Path) -> None:
    dsl_path = _write_text(tmp_path / "policy.dsl", _SAMPLE_DSL)

    first = _run(["evolutionary-policy-dsl-search", str(dsl_path)])
    second = _run(["evolutionary-policy-dsl-search", str(dsl_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert json.loads(first.output) == json.loads(second.output)


def test_policy_dsl_search_rejects_malformed_dsl(tmp_path: Path) -> None:
    dsl_path = _write_text(tmp_path / "policy.dsl", "this is not a rule line\n")

    result = _run(["evolutionary-policy-dsl-search", str(dsl_path)])

    assert result.exit_code == 1
    assert "evolutionary policy-DSL search failed" in result.output


def test_petri_mutation_emits_review_bundle(tmp_path: Path) -> None:
    net_path = _write_json(tmp_path / "net.json", _net_mapping())

    result = _run(
        [
            "evolutionary-petri-mutation",
            str(net_path),
            "--candidates-per-generation",
            "3",
        ]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["grammar"] == "petri"
    assert payload["non_actuating"] is True
    assert len(payload["bundle_hash"]) == 64


def test_petri_mutation_accepts_record_array_net(tmp_path: Path) -> None:
    net_path = _write_json(tmp_path / "net.json", _net_records())

    result = _run(["evolutionary-petri-mutation", str(net_path)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["grammar"] == "petri"


def test_petri_mutation_rejects_empty_net(tmp_path: Path) -> None:
    net_path = _write_json(tmp_path / "net.json", {})

    result = _run(["evolutionary-petri-mutation", str(net_path)])

    assert result.exit_code == 1
    assert "evolutionary Petri mutation failed" in result.output


def test_petri_mutation_rejects_scalar_net(tmp_path: Path) -> None:
    net_path = _write_json(tmp_path / "net.json", 5)

    result = _run(["evolutionary-petri-mutation", str(net_path)])

    assert result.exit_code == 1
    assert "net payload must be a JSON object or array" in result.output


def test_petri_mutation_rejects_malformed_json(tmp_path: Path) -> None:
    net_path = _write_text(tmp_path / "net.json", "{not json")

    result = _run(["evolutionary-petri-mutation", str(net_path)])

    assert result.exit_code == 1
    assert "malformed net JSON" in result.output


def test_topology_mutation_emits_review_bundle(tmp_path: Path) -> None:
    topology_path = _write_json(tmp_path / "topology.json", _topology())
    output_path = tmp_path / "bundle.json"

    result = _run(
        [
            "evolutionary-topology-mutation",
            str(topology_path),
            "--population",
            "4",
            "--output",
            str(output_path),
        ]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["grammar"] == "topology"
    assert payload["report"]["candidate_count"] >= 1


def test_topology_mutation_rejects_missing_nodes(tmp_path: Path) -> None:
    topology_path = _write_json(tmp_path / "topology.json", {"edges": []})

    result = _run(["evolutionary-topology-mutation", str(topology_path)])

    assert result.exit_code == 1
    assert "nodes must be a JSON array" in result.output


def test_topology_mutation_rejects_empty_records(tmp_path: Path) -> None:
    topology_path = _write_json(tmp_path / "topology.json", {"nodes": [], "edges": []})

    result = _run(["evolutionary-topology-mutation", str(topology_path)])

    assert result.exit_code == 1
    assert "evolutionary topology mutation failed" in result.output


def test_topology_mutation_rejects_non_object_node(tmp_path: Path) -> None:
    topology_path = _write_json(tmp_path / "topology.json", {"nodes": [1], "edges": []})

    result = _run(["evolutionary-topology-mutation", str(topology_path)])

    assert result.exit_code == 1
    assert "nodes[0] must be a JSON object" in result.output


def test_output_write_failure_is_reported(tmp_path: Path) -> None:
    dsl_path = _write_text(tmp_path / "policy.dsl", _SAMPLE_DSL)
    output_path = tmp_path / "missing-parent" / "bundle.json"

    result = _run(
        [
            "evolutionary-policy-dsl-search",
            str(dsl_path),
            "--output",
            str(output_path),
        ]
    )

    assert result.exit_code == 1
    assert "cannot write evolutionary grammar review bundle" in result.output


def test_read_text_reports_unreadable_file() -> None:
    with pytest.raises(click.ClickException, match="cannot read policy DSL file"):
        _grammar_cli._read_text(
            Path("/definitely/missing/policy.dsl"), artifact="policy DSL"
        )


def test_load_net_like_reports_unreadable_file() -> None:
    with pytest.raises(click.ClickException, match="cannot read net file"):
        _grammar_cli._load_net_like(Path("/definitely/missing/net.json"))


def test_json_array_of_objects_accepts_valid() -> None:
    payload: Mapping[str, object] = {"nodes": [{"node_id": 0}]}
    result = _grammar_cli._json_array_of_objects(payload, "nodes")
    assert result == ({"node_id": 0},)
