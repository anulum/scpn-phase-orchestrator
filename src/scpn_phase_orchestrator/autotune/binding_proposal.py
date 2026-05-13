# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — reviewable auto-binding proposals

from __future__ import annotations

import csv
import io
import json
import tempfile
from collections.abc import Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from scpn_phase_orchestrator.autotune.discovery import (
    discover_time_series_structure,
    infer_sample_rate_from_time_column,
)
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.exceptions import BindingError
from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ImportedSourceSummary,
    JsonValue,
    RuntimeSnapshot,
    StudioProjectState,
)

__all__ = [
    "propose_binding_from_event_log",
    "propose_binding_from_graph",
    "propose_binding_from_time_series_csv",
]


_CHANNEL_ORDER = ("P", "I", "S")


class _EventLogSourceSummary(ImportedSourceSummary):
    @property
    def event_count(self) -> int:
        return self.sample_count


def propose_binding_from_time_series_csv(
    csv_text: str,
    *,
    sample_rate_hz: float | None,
    project_name: str,
) -> StudioProjectState:
    """Propose a review-only binding for a tabular time-series replay."""
    payload = csv_text.encode("utf-8")
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        raise ValueError("CSV header is required")

    rows = list(reader)
    time_columns = {"time", "timestamp", "t"}
    channels = tuple(
        field
        for field in reader.fieldnames
        if field.strip().lower() not in time_columns
    )
    if not channels:
        raise ValueError("CSV must contain at least one signal channel")
    if not rows:
        raise ValueError("CSV must contain at least one sample")
    signal_table = _numeric_signal_table(rows, channels)

    inferred_channels = _inferred_channels(len(channels), prefer_event=False)
    resolved_sample_rate_hz, sample_rate_inference = _resolve_sample_rate_hz(
        sample_rate_hz,
        rows=rows,
        fieldnames=reader.fieldnames,
    )
    sample_period_s = 1.0 / resolved_sample_rate_hz
    discovery = discover_time_series_structure(
        signal_table,
        columns=channels,
        sample_period_s=sample_period_s,
    )
    yaml_text = _binding_yaml(
        project_name=project_name,
        sample_period_s=sample_period_s,
        family_specs=_families_for_channels(inferred_channels),
    )
    validation_errors = _validation_errors(yaml_text)
    confidence: dict[str, float] = {
        "phase_quality": _bounded_confidence(min(1.0, len(rows) / 3.0)),
        "channel_coverage": _bounded_confidence(min(1.0, len(channels) / 2.0)),
        "validator_acceptance": 1.0 if not validation_errors else 0.0,
    }
    confidence.update(
        {
            name: _bounded_confidence(value)
            for name, value in discovery.confidence_evidence.items()
        }
    )
    source_columns: list[JsonValue] = []
    source_columns.extend(channels)
    provenance: dict[str, JsonValue] = {
        "input_family": "time_series",
        "sample_rate_hz": float(resolved_sample_rate_hz),
        "sample_rate_inference": sample_rate_inference,
        "source_columns": source_columns,
        "discovery_evidence": discovery.to_audit_record(),
        "validator": "load_binding_spec+validate_binding_spec",
    }
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=payload,
        channel_count=len(channels),
        sample_count=len(rows),
    )
    return _project_state(
        project_name=project_name,
        source=source,
        binding=BindingProposal(
            yaml_text=yaml_text,
            validation_errors=validation_errors,
            inferred_channels=inferred_channels,
            confidence_factors=confidence,
            provenance=provenance,
        ),
    )


def propose_binding_from_event_log(
    json_text: str,
    *,
    project_name: str,
) -> StudioProjectState:
    """Propose a review-only binding for a JSON event log."""
    payload = json_text.encode("utf-8")
    events = _event_sequence(json.loads(json_text))
    source_names = sorted(
        {
            str(event.get("source"))
            for event in events
            if isinstance(event.get("source"), str) and event.get("source")
        }
    )
    times = [
        float(event["time"])
        for event in events
        if isinstance(event.get("time"), int | float)
        and not isinstance(event.get("time"), bool)
    ]
    span_s = max(times) - min(times) if len(times) >= 2 else 0.0
    yaml_text = _binding_yaml(
        project_name=project_name,
        sample_period_s=1.0,
        family_specs=(("event_log", "I", "event"),),
    )
    validation_errors = _validation_errors(yaml_text)
    event_density = _bounded_confidence(min(1.0, len(events) / 10.0))
    source_name_values: list[JsonValue] = []
    source_name_values.extend(source_names)
    provenance: dict[str, JsonValue] = {
        "input_family": "event_log",
        "event_count": len(events),
        "source_names": source_name_values,
        "time_span_s": span_s,
        "validator": "load_binding_spec+validate_binding_spec",
    }
    source = _EventLogSourceSummary.from_payload(
        source_kind="event_log_json",
        payload=payload,
        channel_count=max(1, len(source_names)),
        sample_count=len(events),
    )
    return _project_state(
        project_name=project_name,
        source=source,
        binding=BindingProposal(
            yaml_text=yaml_text,
            validation_errors=validation_errors,
            inferred_channels=("I",),
            confidence_factors={
                "event_density": event_density,
                "source_diversity": _bounded_confidence(
                    min(1.0, len(source_names) / 3.0)
                ),
                "validator_acceptance": 1.0 if not validation_errors else 0.0,
            },
            provenance=provenance,
        ),
    )


def propose_binding_from_graph(
    json_text: str,
    *,
    project_name: str,
) -> StudioProjectState:
    """Propose a review-only binding for a graph JSON payload."""
    payload = json_text.encode("utf-8")
    graph = _mapping(json.loads(json_text), "graph")
    nodes = _sequence(graph.get("nodes"), "graph.nodes")
    edges = _sequence(graph.get("edges", ()), "graph.edges")
    node_ids = {_node_id(node) for node in nodes}
    if not node_ids:
        raise ValueError("graph must contain at least one node")
    for edge in edges:
        edge_map = _mapping(edge, "graph.edges[]")
        edge_source = str(edge_map.get("source", ""))
        target = str(edge_map.get("target", ""))
        missing = sorted({edge_source, target} - node_ids)
        if missing:
            raise ValueError(f"unknown graph node in edge: {missing[0]}")

    yaml_text = _binding_yaml(
        project_name=project_name,
        sample_period_s=1.0,
        family_specs=(("graph_topology", "S", "graph"),),
    )
    validation_errors = _validation_errors(yaml_text)
    source = ImportedSourceSummary.from_payload(
        source_kind="graph_json",
        payload=payload,
        channel_count=1,
        sample_count=len(nodes),
    )
    return _project_state(
        project_name=project_name,
        source=source,
        binding=BindingProposal(
            yaml_text=yaml_text,
            validation_errors=validation_errors,
            inferred_channels=("S",),
            confidence_factors={
                "topology_integrity": 1.0,
                "edge_density": _bounded_confidence(
                    min(1.0, len(edges) / len(node_ids))
                ),
                "validator_acceptance": 1.0 if not validation_errors else 0.0,
            },
            provenance={
                "input_family": "graph",
                "node_count": len(node_ids),
                "edge_count": len(edges),
                "validator": "load_binding_spec+validate_binding_spec",
            },
        ),
    )


def _numeric_signal_table(
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
) -> np.ndarray:
    values: list[list[float]] = []
    for row_index, row in enumerate(rows):
        row_values: list[float] = []
        for column in columns:
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"CSV channel {column!r} has non-numeric sample at row {row_index}"
                ) from exc
            if not isfinite(value):
                raise ValueError(
                    f"CSV channel {column!r} has non-finite sample at row {row_index}"
                )
            row_values.append(value)
        values.append(row_values)
    return np.asarray(values, dtype=np.float64)


def _resolve_sample_rate_hz(
    sample_rate_hz: float | None,
    *,
    rows: Sequence[Mapping[str, str]],
    fieldnames: Sequence[str],
) -> tuple[float, str]:
    if sample_rate_hz is None:
        return infer_sample_rate_from_time_column(rows, fieldnames)
    if sample_rate_hz <= 0.0 or not isfinite(sample_rate_hz):
        raise ValueError("sample_rate_hz must be positive")
    return float(sample_rate_hz), "explicit"


def _event_sequence(value: object) -> tuple[Mapping[str, Any], ...]:
    events = _sequence(value, "event log")
    if not events:
        raise ValueError("event log must contain at least one event")
    event_maps = tuple(_mapping(event, "event log[]") for event in events)
    for event in event_maps:
        if "event" not in event:
            raise ValueError("event log entries must include an event field")
    return event_maps


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> tuple[Any, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{label} must be a sequence")
    return tuple(value)


def _node_id(node: object) -> str:
    node_map = _mapping(node, "graph.nodes[]")
    raw_id = node_map.get("id")
    if not isinstance(raw_id, str) or not raw_id:
        raise ValueError("graph nodes must include a non-empty string id")
    return raw_id


def _inferred_channels(count: int, *, prefer_event: bool) -> tuple[str, ...]:
    order = ("I", "P", "S") if prefer_event else _CHANNEL_ORDER
    return tuple(order[index % len(order)] for index in range(count))


def _families_for_channels(channels: Sequence[str]) -> tuple[tuple[str, str, str], ...]:
    families = []
    for index, channel in enumerate(channels):
        extractor = {"P": "physical", "I": "event", "S": "ring"}.get(
            channel, "physical"
        )
        families.append((f"auto_{channel.lower()}_{index}", channel, extractor))
    return tuple(families)


def _binding_yaml(
    *,
    project_name: str,
    sample_period_s: float,
    family_specs: Sequence[tuple[str, str, str]],
) -> str:
    layer_lines = []
    family_lines = []
    good_layers = []
    for index, (family_name, channel, extractor_type) in enumerate(family_specs):
        oscillator_id = f"osc_{index}"
        good_layers.append(str(index))
        layer_lines.extend(
            [
                f"  - name: replay_{index}",
                f"    index: {index}",
                f"    oscillator_ids: [{oscillator_id}]",
                f"    family: {family_name}",
            ]
        )
        family_lines.extend(
            [
                f"  {family_name}:",
                f"    channel: {channel}",
                f"    extractor_type: {extractor_type}",
                "    config: {}",
            ]
        )

    return "\n".join(
        [
            "# Review-only binding proposal. Inspect before copying into a domainpack.",
            f"name: {_yaml_string(project_name)}",
            'version: "0.1.0"',
            "safety_tier: research",
            f"sample_period_s: {sample_period_s:.12g}",
            f"control_period_s: {max(sample_period_s, sample_period_s * 10.0):.12g}",
            "",
            "layers:",
            *layer_lines,
            "",
            "oscillator_families:",
            *family_lines,
            "",
            "coupling:",
            "  base_strength: 0.45",
            "  decay_alpha: 0.3",
            "  templates: {}",
            "",
            "drivers:",
            "  physical:",
            "    zeta: 0.0",
            "    psi: 0.0",
            "  informational:",
            "    zeta: 0.02",
            "  symbolic:",
            "    zeta: 0.02",
            "",
            "objectives:",
            f"  good_layers: [{', '.join(good_layers)}]",
            "  bad_layers: []",
            "  good_weight: 1.0",
            "  bad_weight: 1.0",
            "",
            "boundaries: []",
            "",
            "actuators:",
            "  - name: coupling_global",
            "    knob: K",
            "    scope: global",
            "    limits: [0.0, 3.0]",
            "",
            "amplitude:",
            "  mu: 1.0",
            "  epsilon: 0.3",
            "  amp_coupling_strength: 0.2",
            "  amp_coupling_decay: 0.3",
            "",
            "policy: policy.yaml",
            "",
        ]
    )


def _validation_errors(yaml_text: str) -> tuple[str, ...]:
    with tempfile.TemporaryDirectory(prefix="spo_binding_proposal_") as tmpdir:
        path = Path(tmpdir) / "binding_spec.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        try:
            spec = load_binding_spec(path)
        except (BindingError, ValueError) as exc:
            return (str(exc),)
        return tuple(validate_binding_spec(spec))


def _project_state(
    *,
    project_name: str,
    source: ImportedSourceSummary,
    binding: BindingProposal,
) -> StudioProjectState:
    return StudioProjectState(
        project_name=project_name,
        source=source,
        binding=binding,
        runtime=RuntimeSnapshot(
            R=0.0,
            Psi=0.0,
            K=0.45,
            alpha=0.3,
            zeta=0.0,
            regime="proposal_only",
            replay_status="proposal_only",
        ),
        metadata={"proposal_mode": "review_only"},
    )


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _yaml_string(value: str) -> str:
    if not value:
        raise ValueError("project_name must be non-empty")
    return json.dumps(value)
