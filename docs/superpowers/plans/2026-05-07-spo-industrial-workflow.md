# SPO Industrial Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an industrial-grade, reviewable SPO workflow that connects Studio, auto-binding, replay-only learners, and hierarchy adapter boundaries through tested package APIs.

**Architecture:** Move state and behaviour out of Streamlit tools into small package modules under `src/scpn_phase_orchestrator/studio/`, `autotune/`, and `supervisor/`. Streamlit remains a thin operator surface over deterministic workflow objects, and all learner and hierarchy paths stay non-actuating.

**Tech Stack:** Python 3.12, NumPy, existing binding/autotune/supervisor APIs, Streamlit for the first UI surface, pytest, Ruff, mypy, Bandit, MkDocs.

---

## File Structure

- Create `src/scpn_phase_orchestrator/studio/__init__.py`: public exports for Studio workflow helpers.
- Create `src/scpn_phase_orchestrator/studio/workflow.py`: dataclasses for source summaries, binding proposals, runtime snapshots, export manifests, and project state.
- Create `tests/test_studio_workflow.py`: workflow serialisation, hash stability, safety flags, and no-Streamlit dependency tests.
- Create `src/scpn_phase_orchestrator/autotune/binding_proposal.py`: deterministic auto-binding proposal builders for time-series, event-log, and graph inputs.
- Create `tests/test_autotune_binding_proposal.py`: fixtures for supported imports, malformed input rejection, confidence evidence, and binding validator integration.
- Create `src/scpn_phase_orchestrator/autotune/learners.py`: deterministic PPO-like, SAC-like, and hybrid physics learner proposal generators behind replay-only gates.
- Create `tests/test_autotune_learners.py`: deterministic candidates, unsafe replay rejection, audit serialisation, and non-actuation checks.
- Create `src/scpn_phase_orchestrator/supervisor/hierarchy_adapters.py`: decoded JSONL, REST payload, and WebSocket-frame boundary helpers that feed `HierarchyTransportRuntime`.
- Create `tests/test_supervisor_hierarchy_adapters.py`: stale sequence, protocol mismatch, frame validation, JSONL determinism, and parent-plan integration.
- Create `domainpacks/power_grid/hierarchy_transport_demo.py`: power-grid reduced-summary adapter demo.
- Create `domainpacks/cardiac_rhythm/hierarchy_transport_demo.py`: cardiac reduced-summary adapter demo.
- Create `domainpacks/edge_consensus_nchannel/hierarchy_transport_demo.py`: heterogeneous N-channel adapter demo.
- Modify `tools/spo_studio.py`: convert to a thin UI over workflow helpers, with import, binding review, live metrics, replay learner review, hierarchy monitor, and exports.
- Create `tests/test_spo_studio_helpers.py`: pure helper tests for chart payloads, knob updates, export payloads, and disabled unsafe exports.
- Create docs:
  - `docs/guide/spo_studio_operator.md`
  - `docs/guide/auto_binding.md`
  - `docs/guide/autotune_learners.md`
  - `docs/guide/hierarchy_live_adapters.md`
  - `docs/reference/api/studio.md`
- Modify `docs/roadmap.md` and `ROADMAP.md` only after verified implementation slices exist.
- Modify `mkdocs.yml` to include new public guide and API pages.

## Existing Dirty Worktree Rule

The current workspace has uncommitted hierarchy edits in:

- `ROADMAP.md`
- `docs/reference/api/supervisor.md`
- `docs/roadmap.md`
- `src/scpn_phase_orchestrator/supervisor/__init__.py`
- `src/scpn_phase_orchestrator/supervisor/hierarchy.py`
- `tests/test_supervisor_hierarchy.py`

Before Task 4, either finish and commit those hierarchy edits as their own
slice, or adapt Task 4 to build on them without reverting or overwriting them.

## Task 1: Workflow Core

**Files:**
- Create: `src/scpn_phase_orchestrator/studio/__init__.py`
- Create: `src/scpn_phase_orchestrator/studio/workflow.py`
- Test: `tests/test_studio_workflow.py`

- [ ] **Step 1: Write the failing workflow serialisation test**

Add this test file:

```python
from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ExportManifest,
    ImportedSourceSummary,
    RuntimeSnapshot,
    StudioProjectState,
)


def test_project_state_serialises_with_stable_hashes() -> None:
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=b"t,a,b\n0,0.0,1.0\n1,0.2,0.8\n",
        channel_count=2,
        sample_count=2,
    )
    proposal = BindingProposal(
        yaml_text="version: 1\nname: demo\n",
        validation_errors=(),
        inferred_channels=("P", "I"),
        confidence_factors={"phase_quality": 0.75},
        provenance={"source_kind": source.source_kind},
    )
    snapshot = RuntimeSnapshot(
        R=0.81,
        Psi=0.12,
        K=1.4,
        alpha=0.0,
        zeta=0.2,
        regime="nominal",
        layer_metrics=(("grid", 0.8),),
        hierarchy_watermarks={"edge-a": 3},
        replay_status="proposal_only",
    )
    manifest = ExportManifest.review_artifact(
        target_kind="binding_spec",
        file_name="binding_spec.yaml",
        payload="version: 1\nname: demo\n",
        command="spo run binding_spec.yaml --audit audit.jsonl",
    )
    state = StudioProjectState(
        project_name="demo",
        source=source,
        binding=proposal,
        runtime=snapshot,
        exports=(manifest,),
    )

    record = state.to_audit_record()
    restored = json.loads(json.dumps(record))

    assert restored["project_name"] == "demo"
    assert restored["source"]["sha256"] == source.sha256
    assert restored["exports"][0]["payload_sha256"] == manifest.payload_sha256
    assert restored["exports"][0]["safety_posture"] == "review_artifact"
    assert restored["runtime"]["hierarchy_watermarks"] == {"edge-a": 3}


def test_export_manifest_rejects_deployable_without_warning() -> None:
    with pytest.raises(ValueError, match="deployable exports require warnings"):
        ExportManifest(
            target_kind="docker",
            file_name="deploy.json",
            payload="{}",
            command="docker compose up",
            safety_posture="deployable",
            warnings=(),
        )


def test_workflow_module_does_not_import_streamlit() -> None:
    import sys

    assert "streamlit" not in sys.modules
```

- [ ] **Step 2: Run the workflow test to verify it fails**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_studio_workflow.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'scpn_phase_orchestrator.studio'`.

- [ ] **Step 3: Implement the workflow core**

Create `src/scpn_phase_orchestrator/studio/__init__.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Operator workflow helpers for SPO Studio."""

from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ExportManifest,
    ImportedSourceSummary,
    RuntimeSnapshot,
    StudioProjectState,
)

__all__ = [
    "BindingProposal",
    "ExportManifest",
    "ImportedSourceSummary",
    "RuntimeSnapshot",
    "StudioProjectState",
]
```

Create `src/scpn_phase_orchestrator/studio/workflow.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Serialisable project-state spine for SPO Studio workflows."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "BindingProposal",
    "ExportManifest",
    "ImportedSourceSummary",
    "RuntimeSnapshot",
    "StudioProjectState",
]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(payload: str) -> str:
    return _sha256_bytes(payload.encode("utf-8"))


def _require_non_empty(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True)
class ImportedSourceSummary:
    source_kind: str
    sha256: str
    channel_count: int = 0
    sample_count: int = 0
    event_count: int = 0
    graph_node_count: int = 0
    graph_edge_count: int = 0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.source_kind, "source_kind")
        _require_non_empty(self.sha256, "sha256")
        for name in (
            "channel_count",
            "sample_count",
            "event_count",
            "graph_node_count",
            "graph_edge_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @classmethod
    def from_payload(
        cls,
        *,
        source_kind: str,
        payload: bytes,
        channel_count: int = 0,
        sample_count: int = 0,
        event_count: int = 0,
        graph_node_count: int = 0,
        graph_edge_count: int = 0,
        warnings: tuple[str, ...] = (),
    ) -> "ImportedSourceSummary":
        return cls(
            source_kind=source_kind,
            sha256=_sha256_bytes(payload),
            channel_count=channel_count,
            sample_count=sample_count,
            event_count=event_count,
            graph_node_count=graph_node_count,
            graph_edge_count=graph_edge_count,
            warnings=warnings,
        )

    def to_audit_record(self) -> dict[str, object]:
        return {
            "source_kind": self.source_kind,
            "sha256": self.sha256,
            "channel_count": self.channel_count,
            "sample_count": self.sample_count,
            "event_count": self.event_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class BindingProposal:
    yaml_text: str
    validation_errors: tuple[str, ...] = ()
    inferred_channels: tuple[str, ...] = ()
    confidence_factors: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_non_empty(self.yaml_text, "yaml_text")
        for name, value in self.confidence_factors.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence factor {name!r} must be within [0, 1]")

    @property
    def yaml_sha256(self) -> str:
        return _sha256_text(self.yaml_text)

    @property
    def review_required(self) -> bool:
        return bool(self.validation_errors)

    def to_audit_record(self) -> dict[str, object]:
        return {
            "yaml_sha256": self.yaml_sha256,
            "validation_errors": list(self.validation_errors),
            "inferred_channels": list(self.inferred_channels),
            "confidence_factors": dict(self.confidence_factors),
            "provenance": dict(self.provenance),
            "review_required": self.review_required,
        }


@dataclass(frozen=True)
class RuntimeSnapshot:
    R: float
    Psi: float
    K: float
    alpha: float
    zeta: float
    regime: str
    layer_metrics: tuple[tuple[str, float], ...] = ()
    hierarchy_watermarks: Mapping[str, int] = field(default_factory=dict)
    replay_status: str = "not_run"

    def __post_init__(self) -> None:
        _require_non_empty(self.regime, "regime")
        _require_non_empty(self.replay_status, "replay_status")
        for node, sequence in self.hierarchy_watermarks.items():
            _require_non_empty(node, "hierarchy watermark node")
            if sequence < 0:
                raise ValueError("hierarchy watermark sequence must be non-negative")

    def to_audit_record(self) -> dict[str, object]:
        return {
            "R": float(self.R),
            "Psi": float(self.Psi),
            "K": float(self.K),
            "alpha": float(self.alpha),
            "zeta": float(self.zeta),
            "regime": self.regime,
            "layer_metrics": [
                {"layer": layer, "R": float(value)}
                for layer, value in self.layer_metrics
            ],
            "hierarchy_watermarks": dict(self.hierarchy_watermarks),
            "replay_status": self.replay_status,
        }


@dataclass(frozen=True)
class ExportManifest:
    target_kind: str
    file_name: str
    payload: str
    command: str
    safety_posture: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("target_kind", "file_name", "payload", "command"):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.safety_posture not in {"review_artifact", "deployable"}:
            raise ValueError("safety_posture must be review_artifact or deployable")
        if self.safety_posture == "deployable" and not self.warnings:
            raise ValueError("deployable exports require warnings")

    @classmethod
    def review_artifact(
        cls,
        *,
        target_kind: str,
        file_name: str,
        payload: str,
        command: str,
        warnings: tuple[str, ...] = (),
    ) -> "ExportManifest":
        return cls(
            target_kind=target_kind,
            file_name=file_name,
            payload=payload,
            command=command,
            safety_posture="review_artifact",
            warnings=warnings,
        )

    @property
    def payload_sha256(self) -> str:
        return _sha256_text(self.payload)

    def to_audit_record(self) -> dict[str, object]:
        return {
            "target_kind": self.target_kind,
            "file_name": self.file_name,
            "payload_sha256": self.payload_sha256,
            "command": self.command,
            "safety_posture": self.safety_posture,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class StudioProjectState:
    project_name: str
    source: ImportedSourceSummary
    binding: BindingProposal
    runtime: RuntimeSnapshot | None = None
    exports: tuple[ExportManifest, ...] = ()

    def __post_init__(self) -> None:
        _require_non_empty(self.project_name, "project_name")

    def to_audit_record(self) -> dict[str, object]:
        return {
            "project_name": self.project_name,
            "source": self.source.to_audit_record(),
            "binding": self.binding.to_audit_record(),
            "runtime": (
                self.runtime.to_audit_record() if self.runtime is not None else None
            ),
            "exports": [export.to_audit_record() for export in self.exports],
        }
```

- [ ] **Step 4: Run the workflow tests to verify green**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_studio_workflow.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/scpn_phase_orchestrator/studio/__init__.py src/scpn_phase_orchestrator/studio/workflow.py tests/test_studio_workflow.py
git commit -m "feat: add SPO Studio workflow core" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 2: Auto-Binding Proposal Package

**Files:**
- Create: `src/scpn_phase_orchestrator/autotune/binding_proposal.py`
- Modify: `src/scpn_phase_orchestrator/autotune/__init__.py`
- Test: `tests/test_autotune_binding_proposal.py`

- [ ] **Step 1: Write failing tests for CSV, event, and graph proposals**

Create `tests/test_autotune_binding_proposal.py`:

```python
from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.autotune.binding_proposal import (
    propose_binding_from_event_log,
    propose_binding_from_graph,
    propose_binding_from_time_series_csv,
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


def test_graph_proposal_rejects_edges_with_unknown_nodes() -> None:
    graph = {
        "nodes": [{"id": "a"}],
        "edges": [{"source": "a", "target": "missing"}],
    }

    with pytest.raises(ValueError, match="unknown graph node"):
        propose_binding_from_graph(json.dumps(graph), project_name="bad_graph")
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_autotune_binding_proposal.py -q
```

Expected: fail with `ModuleNotFoundError` for `autotune.binding_proposal`.

- [ ] **Step 3: Implement the proposal module**

Create `src/scpn_phase_orchestrator/autotune/binding_proposal.py` with deterministic parsers, summary builders, and YAML rendering:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Reviewable auto-binding proposal builders."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from io import StringIO
from tempfile import NamedTemporaryFile

from scpn_phase_orchestrator.binding.loader import BindingLoadError, load_binding_spec
from scpn_phase_orchestrator.binding.validator import validate_binding_spec
from scpn_phase_orchestrator.studio.workflow import (
    BindingProposal,
    ImportedSourceSummary,
    StudioProjectState,
)

__all__ = [
    "propose_binding_from_event_log",
    "propose_binding_from_graph",
    "propose_binding_from_time_series_csv",
]


@dataclass(frozen=True)
class _ChannelPlan:
    name: str
    extractor: str
    channel: str


def propose_binding_from_time_series_csv(
    csv_text: str,
    *,
    sample_rate_hz: float,
    project_name: str,
) -> StudioProjectState:
    rows = list(csv.DictReader(StringIO(csv_text)))
    if not rows:
        raise ValueError("time-series CSV must contain at least one row")
    fieldnames = tuple(rows[0].keys())
    if len(fieldnames) < 2:
        raise ValueError("time-series CSV must contain time plus at least one channel")
    channels = tuple(field for field in fieldnames if field.lower() != "time")
    if not channels:
        raise ValueError("time-series CSV must contain at least one non-time channel")
    plans = tuple(
        _ChannelPlan(name=name, extractor="hilbert", channel=_default_channel(index))
        for index, name in enumerate(channels)
    )
    yaml_text = _render_binding_yaml(project_name, sample_rate_hz, plans)
    source = ImportedSourceSummary.from_payload(
        source_kind="time_series_csv",
        payload=csv_text.encode("utf-8"),
        channel_count=len(channels),
        sample_count=len(rows),
    )
    return StudioProjectState(
        project_name=project_name,
        source=source,
        binding=_proposal(
            yaml_text,
            inferred_channels=tuple(plan.channel for plan in plans),
            confidence_factors={"phase_quality": min(1.0, len(rows) / 64.0)},
            provenance={"input_family": "time_series", "sample_rate_hz": sample_rate_hz},
        ),
    )


def propose_binding_from_event_log(
    json_text: str,
    *,
    project_name: str,
) -> StudioProjectState:
    events = json.loads(json_text)
    if not isinstance(events, list) or not events:
        raise ValueError("event log must be a non-empty JSON list")
    sources = Counter(str(event.get("source", "")) for event in events)
    if "" in sources:
        raise ValueError("event log entries must include source")
    plans = tuple(
        _ChannelPlan(name=source, extractor="event", channel=_default_channel(index))
        for index, source in enumerate(sorted(sources))
    )
    yaml_text = _render_binding_yaml(project_name, 1.0, plans)
    source = ImportedSourceSummary.from_payload(
        source_kind="event_log_json",
        payload=json_text.encode("utf-8"),
        event_count=len(events),
        channel_count=len(plans),
    )
    density = min(1.0, len(events) / max(8.0, 4.0 * len(plans)))
    return StudioProjectState(
        project_name=project_name,
        source=source,
        binding=_proposal(
            yaml_text,
            inferred_channels=tuple(plan.channel for plan in plans),
            confidence_factors={"event_density": density},
            provenance={"input_family": "event_log"},
        ),
    )


def propose_binding_from_graph(
    json_text: str,
    *,
    project_name: str,
) -> StudioProjectState:
    graph = json.loads(json_text)
    nodes = graph.get("nodes")
    edges = graph.get("edges", [])
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("graph JSON must include a non-empty nodes list")
    node_ids = tuple(str(node.get("id", "")) for node in nodes)
    if any(not node for node in node_ids):
        raise ValueError("graph nodes must include id")
    known = set(node_ids)
    for edge in edges:
        if edge.get("source") not in known or edge.get("target") not in known:
            raise ValueError("graph edge references unknown graph node")
    plans = tuple(
        _ChannelPlan(name=node_id, extractor="graph", channel=_default_channel(index))
        for index, node_id in enumerate(node_ids)
    )
    yaml_text = _render_binding_yaml(project_name, 1.0, plans)
    source = ImportedSourceSummary.from_payload(
        source_kind="graph_json",
        payload=json_text.encode("utf-8"),
        channel_count=len(plans),
        graph_node_count=len(node_ids),
        graph_edge_count=len(edges),
    )
    connectivity = min(1.0, len(edges) / max(1.0, len(node_ids)))
    return StudioProjectState(
        project_name=project_name,
        source=source,
        binding=_proposal(
            yaml_text,
            inferred_channels=tuple(plan.channel for plan in plans),
            confidence_factors={"graph_connectivity": connectivity},
            provenance={"input_family": "graph"},
        ),
    )


def _default_channel(index: int) -> str:
    defaults = ("P", "I", "S")
    if index < len(defaults):
        return defaults[index]
    return f"X{index + 1}"


def _render_binding_yaml(
    project_name: str,
    sample_rate_hz: float,
    plans: tuple[_ChannelPlan, ...],
) -> str:
    sample_period = 1.0 / sample_rate_hz if sample_rate_hz > 0 else 1.0
    families = "\n".join(
        f"  {plan.name}:\n"
        f"    channel: {plan.channel}\n"
        f"    extractor_type: {plan.extractor}\n"
        f"    config:\n"
        f"      channel_ids: [{plan.name!r}]"
        for plan in plans
    )
    layers = "\n".join(
        f"  - name: {plan.name}\n"
        f"    oscillator_ids: [{plan.name!r}]\n"
        f"    oscillator_family: {plan.name}"
        for plan in plans
    )
    return (
        "version: 1\n"
        f"name: {project_name}\n"
        f"sample_period_s: {sample_period:.9g}\n"
        "oscillator_families:\n"
        f"{families}\n"
        "layers:\n"
        f"{layers}\n"
        "drivers:\n"
        "  channels: {}\n"
    )


def _proposal(
    yaml_text: str,
    *,
    inferred_channels: tuple[str, ...],
    confidence_factors: dict[str, float],
    provenance: dict[str, object],
) -> BindingProposal:
    errors = _validate_yaml_text(yaml_text)
    return BindingProposal(
        yaml_text=yaml_text,
        validation_errors=tuple(errors),
        inferred_channels=inferred_channels,
        confidence_factors=confidence_factors,
        provenance=provenance,
    )


def _validate_yaml_text(yaml_text: str) -> list[str]:
    with NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        f.write(yaml_text)
        f.flush()
        try:
            spec = load_binding_spec(f.name)
        except BindingLoadError as exc:
            return [str(exc)]
    return validate_binding_spec(spec)
```

Update `src/scpn_phase_orchestrator/autotune/__init__.py` to export the three proposal builders.

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_autotune_binding_proposal.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/scpn_phase_orchestrator/autotune/__init__.py src/scpn_phase_orchestrator/autotune/binding_proposal.py tests/test_autotune_binding_proposal.py
git commit -m "feat: add reviewable auto-binding proposals" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 3: Replay-Only Learner Proposal Interfaces

**Files:**
- Create: `src/scpn_phase_orchestrator/autotune/learners.py`
- Modify: `src/scpn_phase_orchestrator/autotune/__init__.py`
- Test: `tests/test_autotune_learners.py`

- [ ] **Step 1: Write failing learner tests**

Create `tests/test_autotune_learners.py`:

```python
from __future__ import annotations

from scpn_phase_orchestrator.autotune.learners import (
    generate_hybrid_physics_proposal,
    generate_ppo_like_proposal,
    generate_sac_like_proposal,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    RewardObservation,
)


def _safe_observation(candidate: KnobPolicyCandidate) -> RewardObservation:
    return RewardObservation(
        R_good=0.82,
        R_bad=0.03,
        regime_churn=0.0,
        unsafe_actuation=False,
        metadata={"candidate_K": float(candidate.K)},
    )


def _unsafe_observation(candidate: KnobPolicyCandidate) -> RewardObservation:
    return RewardObservation(
        R_good=0.91,
        R_bad=0.8,
        regime_churn=3.0,
        unsafe_actuation=True,
        metadata={"candidate_K": float(candidate.K)},
    )


def test_ppo_like_proposal_is_deterministic_and_non_actuating() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)
    second = generate_ppo_like_proposal(seed, _safe_observation, seed_value=7)

    assert proposal.learner_kind == "ppo_like_replay"
    assert proposal.to_audit_record() == second.to_audit_record()
    assert proposal.actuation_permitted is False
    assert proposal.policy_search.proposal.accepted is True


def test_sac_like_proposal_rejects_unsafe_replay() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_sac_like_proposal(seed, _unsafe_observation, seed_value=11)

    assert proposal.learner_kind == "sac_like_replay"
    assert proposal.actuation_permitted is False
    assert proposal.policy_search.proposal.accepted is False


def test_hybrid_physics_proposal_records_prior() -> None:
    seed = KnobPolicyCandidate(K=1.0, alpha=0.0, zeta=0.1, Psi=0.0)

    proposal = generate_hybrid_physics_proposal(
        seed,
        _safe_observation,
        critical_coupling_estimate=1.4,
    )

    record = proposal.to_audit_record()
    assert record["learner_kind"] == "hybrid_physics_replay"
    assert record["physics_prior"]["critical_coupling_estimate"] == 1.4
    assert record["actuation_permitted"] is False
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_autotune_learners.py -q
```

Expected: fail with `ModuleNotFoundError` for `autotune.learners`.

- [ ] **Step 3: Implement learner proposal generators**

Create `src/scpn_phase_orchestrator/autotune/learners.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Replay-only learner-shaped autotune proposal interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from scpn_phase_orchestrator.autotune.policy_search import (
    AdaptiveReplayPolicySearchConfig,
    AdaptiveReplayPolicySearchResult,
    ReplayPolicyEvaluator,
    search_adaptive_replay_policy,
)
from scpn_phase_orchestrator.autotune.reward import (
    KnobPolicyCandidate,
    OfflinePolicySearchConfig,
    PolicyProposalConfig,
)

__all__ = [
    "LearnerPolicyProposal",
    "generate_hybrid_physics_proposal",
    "generate_ppo_like_proposal",
    "generate_sac_like_proposal",
]


@dataclass(frozen=True)
class LearnerPolicyProposal:
    learner_kind: str
    policy_search: AdaptiveReplayPolicySearchResult
    actuation_permitted: bool = False
    learner_metadata: dict[str, object] = field(default_factory=dict)
    physics_prior: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actuation_permitted:
            raise ValueError("learner proposals must remain non-actuating")

    def to_audit_record(self) -> dict[str, object]:
        return {
            "learner_kind": self.learner_kind,
            "actuation_permitted": self.actuation_permitted,
            "learner_metadata": dict(self.learner_metadata),
            "physics_prior": dict(self.physics_prior),
            "policy_search": self.policy_search.to_audit_record(),
        }


def generate_ppo_like_proposal(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    seed_value: int = 0,
) -> LearnerPolicyProposal:
    return _generate_seeded_proposal(
        "ppo_like_replay",
        seed,
        evaluator,
        seed_value=seed_value,
        entropy_scale=0.04,
        step_scale=1.0,
    )


def generate_sac_like_proposal(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    seed_value: int = 0,
) -> LearnerPolicyProposal:
    return _generate_seeded_proposal(
        "sac_like_replay",
        seed,
        evaluator,
        seed_value=seed_value,
        entropy_scale=0.08,
        step_scale=0.7,
    )


def generate_hybrid_physics_proposal(
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    critical_coupling_estimate: float,
) -> LearnerPolicyProposal:
    if not np.isfinite(critical_coupling_estimate) or critical_coupling_estimate <= 0:
        raise ValueError("critical_coupling_estimate must be finite and positive")
    adjusted = KnobPolicyCandidate(
        K=float(0.5 * (float(seed.K) + critical_coupling_estimate)),
        alpha=seed.alpha,
        zeta=seed.zeta,
        Psi=seed.Psi,
        channel_weights=seed.channel_weights,
        cross_channel_gains=seed.cross_channel_gains,
    )
    search = search_adaptive_replay_policy(
        adjusted,
        evaluator,
        adaptive_config=AdaptiveReplayPolicySearchConfig(
            base_search_config=OfflinePolicySearchConfig(
                K_step=0.1 * critical_coupling_estimate,
                alpha_step=0.05,
                zeta_step=0.05,
                Psi_step=0.05,
                include_baseline=True,
            ),
            iterations=2,
            step_decay=0.5,
        ),
        proposal_config=PolicyProposalConfig(require_safe=True),
    )
    return LearnerPolicyProposal(
        learner_kind="hybrid_physics_replay",
        policy_search=search,
        physics_prior={"critical_coupling_estimate": critical_coupling_estimate},
    )


def _generate_seeded_proposal(
    learner_kind: str,
    seed: KnobPolicyCandidate,
    evaluator: ReplayPolicyEvaluator,
    *,
    seed_value: int,
    entropy_scale: float,
    step_scale: float,
) -> LearnerPolicyProposal:
    rng = np.random.default_rng(seed_value)
    jitter = float(rng.normal(0.0, entropy_scale))
    adjusted = KnobPolicyCandidate(
        K=max(0.0, float(seed.K) + jitter),
        alpha=seed.alpha,
        zeta=seed.zeta,
        Psi=seed.Psi,
        channel_weights=seed.channel_weights,
        cross_channel_gains=seed.cross_channel_gains,
    )
    search = search_adaptive_replay_policy(
        adjusted,
        evaluator,
        adaptive_config=AdaptiveReplayPolicySearchConfig(
            base_search_config=OfflinePolicySearchConfig(
                K_step=0.1 * step_scale,
                alpha_step=0.05 * step_scale,
                zeta_step=0.05 * step_scale,
                Psi_step=0.05 * step_scale,
                include_baseline=True,
            ),
            iterations=2,
            step_decay=0.5,
        ),
        proposal_config=PolicyProposalConfig(require_safe=True),
    )
    return LearnerPolicyProposal(
        learner_kind=learner_kind,
        policy_search=search,
        learner_metadata={
            "seed_value": seed_value,
            "entropy_scale": entropy_scale,
            "step_scale": step_scale,
        },
    )
```

Update `src/scpn_phase_orchestrator/autotune/__init__.py` to export `LearnerPolicyProposal`, `generate_ppo_like_proposal`, `generate_sac_like_proposal`, and `generate_hybrid_physics_proposal`.

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_autotune_learners.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add src/scpn_phase_orchestrator/autotune/__init__.py src/scpn_phase_orchestrator/autotune/learners.py tests/test_autotune_learners.py
git commit -m "feat: add replay-only learner proposals" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 4: Hierarchy Live-Adapter Boundaries

**Files:**
- Create: `src/scpn_phase_orchestrator/supervisor/hierarchy_adapters.py`
- Modify: `src/scpn_phase_orchestrator/supervisor/__init__.py`
- Test: `tests/test_supervisor_hierarchy_adapters.py`

- [ ] **Step 1: Resolve existing hierarchy edits**

Run:

```bash
git status --short -- src/scpn_phase_orchestrator/supervisor/hierarchy.py tests/test_supervisor_hierarchy.py docs/reference/api/supervisor.md
```

Expected: either no uncommitted hierarchy changes, or the current uncommitted hierarchy runtime slice is intentionally kept and Task 4 imports `HierarchyTransportRuntime` from that local implementation.

- [ ] **Step 2: Write failing adapter tests**

Create `tests/test_supervisor_hierarchy_adapters.py`:

```python
from __future__ import annotations

import json

from scpn_phase_orchestrator.supervisor.hierarchy import ChildSupervisorSummary
from scpn_phase_orchestrator.supervisor.hierarchy_adapters import (
    HierarchyDecodedFrameAdapter,
    HierarchyJsonlReplayAdapter,
    HierarchyRestPayloadAdapter,
)


def _record(sequence: int = 1) -> dict[str, object]:
    return {
        "protocol_version": "spo-hierarchy-sync/v1",
        "source_node": "edge-a",
        "sequence": sequence,
        "summary": ChildSupervisorSummary(
            name="grid",
            channel="P",
            R=0.72,
            psi=0.1,
        ).to_audit_record(),
    }


def test_jsonl_adapter_ingests_valid_records_and_rejects_stale_sequence() -> None:
    adapter = HierarchyJsonlReplayAdapter()

    first = adapter.ingest_jsonl(json.dumps(_record(1)))
    second = adapter.ingest_jsonl(json.dumps(_record(1)))

    assert first["accepted_count"] == 1
    assert second["accepted_count"] == 0
    assert second["rejected"][0]["reason"] == "stale_or_duplicate_sequence"


def test_rest_payload_adapter_requires_content_type() -> None:
    adapter = HierarchyRestPayloadAdapter()

    rejected = adapter.submit_payload(_record(1), headers={"content-type": "text/plain"})

    assert rejected["accepted_count"] == 0
    assert rejected["rejected"][0]["reason"] == "unsupported_content_type"


def test_decoded_frame_adapter_preserves_parent_plan_summary() -> None:
    adapter = HierarchyDecodedFrameAdapter()

    result = adapter.submit_frame(_record(1))

    assert result["accepted_count"] == 1
    assert result["plan"]["parent"]["layer_count"] == 1
    assert result["watermarks"]["edge-a"] == 1
```

- [ ] **Step 3: Run tests to verify red**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_supervisor_hierarchy_adapters.py -q
```

Expected: fail with `ModuleNotFoundError` for `supervisor.hierarchy_adapters`.

- [ ] **Step 4: Implement adapter boundaries**

Create `src/scpn_phase_orchestrator/supervisor/hierarchy_adapters.py`:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Decoded live-adapter boundaries for hierarchy sync payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from scpn_phase_orchestrator.supervisor.hierarchy import HierarchyTransportRuntime

__all__ = [
    "HierarchyDecodedFrameAdapter",
    "HierarchyJsonlReplayAdapter",
    "HierarchyRestPayloadAdapter",
]


@dataclass
class _BaseHierarchyAdapter:
    runtime: HierarchyTransportRuntime = field(default_factory=HierarchyTransportRuntime)

    def _ledger_record(self, ledger) -> dict[str, object]:
        audit = ledger.to_audit_record()
        return {
            "accepted_count": len(ledger.accepted),
            "rejected": audit["rejected"],
            "plan": audit["plan"],
            "watermarks": self.runtime.to_audit_record()["previous_sequences"],
        }

    def _rejected(self, reason: str) -> dict[str, object]:
        return {
            "accepted_count": 0,
            "rejected": [{"reason": reason}],
            "plan": None,
            "watermarks": self.runtime.to_audit_record()["previous_sequences"],
        }


@dataclass
class HierarchyJsonlReplayAdapter(_BaseHierarchyAdapter):
    def ingest_jsonl(self, text: str) -> dict[str, object]:
        records = [line for line in text.splitlines() if line.strip()]
        if not records:
            return self._rejected("empty_jsonl")
        try:
            ledger = self.runtime.ingest_batch(records)
        except ValueError as exc:
            if "at least one hierarchy sync envelope" in str(exc):
                return self._rejected("no_accepted_records")
            raise
        return self._ledger_record(ledger)


@dataclass
class HierarchyRestPayloadAdapter(_BaseHierarchyAdapter):
    def submit_payload(
        self,
        payload: Mapping[str, object],
        *,
        headers: Mapping[str, str],
    ) -> dict[str, object]:
        content_type = headers.get("content-type", headers.get("Content-Type", ""))
        if "application/json" not in content_type:
            return self._rejected("unsupported_content_type")
        ledger = self.runtime.ingest_batch([payload])
        return self._ledger_record(ledger)


@dataclass
class HierarchyDecodedFrameAdapter(_BaseHierarchyAdapter):
    def submit_frame(self, frame: Mapping[str, object]) -> dict[str, object]:
        ledger = self.runtime.ingest_batch([frame])
        return self._ledger_record(ledger)
```

Update `src/scpn_phase_orchestrator/supervisor/__init__.py` to export the three adapter classes.

- [ ] **Step 5: Run tests to verify green**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_supervisor_hierarchy.py tests/test_supervisor_hierarchy_adapters.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

Run:

```bash
git add src/scpn_phase_orchestrator/supervisor/__init__.py src/scpn_phase_orchestrator/supervisor/hierarchy_adapters.py tests/test_supervisor_hierarchy_adapters.py
git commit -m "feat: add hierarchy adapter boundaries" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 5: Multi-Domain Hierarchy Demos

**Files:**
- Create: `domainpacks/power_grid/hierarchy_transport_demo.py`
- Create: `domainpacks/cardiac_rhythm/hierarchy_transport_demo.py`
- Create: `domainpacks/edge_consensus_nchannel/hierarchy_transport_demo.py`
- Test: `tests/test_hierarchy_transport_demos.py`

- [ ] **Step 1: Write failing demo tests**

Create `tests/test_hierarchy_transport_demos.py`:

```python
from __future__ import annotations

from domainpacks.cardiac_rhythm.hierarchy_transport_demo import run_demo as run_cardiac
from domainpacks.edge_consensus_nchannel.hierarchy_transport_demo import (
    run_demo as run_edge,
)
from domainpacks.power_grid.hierarchy_transport_demo import run_demo as run_grid


def test_hierarchy_transport_demos_emit_reproducible_audit_records() -> None:
    records = [run_grid(), run_cardiac(), run_edge()]

    assert [record["domain"] for record in records] == [
        "power_grid",
        "cardiac_rhythm",
        "edge_consensus_nchannel",
    ]
    assert all(record["accepted_count"] >= 1 for record in records)
    assert all(record["network_opened"] is False for record in records)
    assert records[2]["channel_count"] > 3
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_hierarchy_transport_demos.py -q
```

Expected: fail because the demo modules do not exist.

- [ ] **Step 3: Implement the demo modules**

Each module should create two or more `ChildSupervisorSummary` records, submit
them through `HierarchyDecodedFrameAdapter`, and return an audit record with
`domain`, `accepted_count`, `network_opened=False`, and `channel_count`.

Use this shape for `domainpacks/power_grid/hierarchy_transport_demo.py` and vary
names/channels per domain:

```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Power-grid hierarchy transport boundary demo."""

from __future__ import annotations

from scpn_phase_orchestrator.supervisor.hierarchy import (
    ChildSupervisorSummary,
    build_hierarchy_sync_envelope,
)
from scpn_phase_orchestrator.supervisor.hierarchy_adapters import (
    HierarchyDecodedFrameAdapter,
)


def run_demo() -> dict[str, object]:
    adapter = HierarchyDecodedFrameAdapter()
    summaries = (
        ChildSupervisorSummary(name="generator_area", channel="P", R=0.74, psi=0.1),
        ChildSupervisorSummary(name="load_area", channel="I", R=0.68, psi=0.3),
    )
    accepted_count = 0
    for sequence, summary in enumerate(summaries, start=1):
        envelope = build_hierarchy_sync_envelope(
            summary,
            source_node=f"grid-edge-{sequence}",
            sequence=sequence,
        )
        accepted_count += int(adapter.submit_frame(envelope.to_audit_record())["accepted_count"])
    return {
        "domain": "power_grid",
        "accepted_count": accepted_count,
        "network_opened": False,
        "channel_count": len({summary.channel for summary in summaries}),
    }
```

- [ ] **Step 4: Run tests to verify green**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_hierarchy_transport_demos.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add domainpacks/power_grid/hierarchy_transport_demo.py domainpacks/cardiac_rhythm/hierarchy_transport_demo.py domainpacks/edge_consensus_nchannel/hierarchy_transport_demo.py tests/test_hierarchy_transport_demos.py
git commit -m "feat: add hierarchy transport demos" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 6: Studio UI Over Workflow Core

**Files:**
- Modify: `tools/spo_studio.py`
- Test: `tests/test_spo_studio_helpers.py`

- [ ] **Step 1: Write failing helper tests**

Create `tests/test_spo_studio_helpers.py`:

```python
from __future__ import annotations

from tools.spo_studio import (
    build_export_options,
    build_metric_chart_payload,
    clamp_knob_state,
)


def test_clamp_knob_state_bounds_operator_inputs() -> None:
    state = clamp_knob_state(K=-1.0, alpha=12.0, zeta=99.0, Psi=-3.0)

    assert state == {"K": 0.0, "alpha": 10.0, "zeta": 10.0, "Psi": 0.0}


def test_metric_chart_payload_preserves_live_order() -> None:
    payload = build_metric_chart_payload(
        [
            {"R": 0.4, "Psi": 0.1, "K": 1.0},
            {"R": 0.8, "Psi": 0.2, "K": 1.4},
        ]
    )

    assert payload["R"] == [0.4, 0.8]
    assert payload["Psi"] == [0.1, 0.2]
    assert payload["K"] == [1.0, 1.4]


def test_export_options_disable_deployable_when_validation_fails() -> None:
    options = build_export_options(validation_errors=("missing layer",))

    assert options["binding_spec"]["enabled"] is False
    assert options["docker_manifest"]["enabled"] is False
    assert options["audit_summary"]["enabled"] is True
```

- [ ] **Step 2: Run tests to verify red**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_spo_studio_helpers.py -q
```

Expected: fail because helper functions are missing.

- [ ] **Step 3: Refactor `tools/spo_studio.py` helpers before UI code**

Add pure functions near the top of `tools/spo_studio.py`:

```python
def clamp_knob_state(*, K: float, alpha: float, zeta: float, Psi: float) -> dict[str, float]:
    return {
        "K": min(10.0, max(0.0, float(K))),
        "alpha": min(10.0, max(0.0, float(alpha))),
        "zeta": min(10.0, max(0.0, float(zeta))),
        "Psi": min(10.0, max(0.0, float(Psi))),
    }


def build_metric_chart_payload(history: list[dict[str, float]]) -> dict[str, list[float]]:
    return {
        "R": [float(row["R"]) for row in history],
        "Psi": [float(row["Psi"]) for row in history],
        "K": [float(row["K"]) for row in history],
    }


def build_export_options(*, validation_errors: tuple[str, ...]) -> dict[str, dict[str, object]]:
    valid = not validation_errors
    return {
        "binding_spec": {"enabled": valid, "reason": "valid binding required"},
        "docker_manifest": {"enabled": valid, "reason": "valid binding required"},
        "wasm_manifest": {"enabled": valid, "reason": "valid binding required"},
        "audit_summary": {"enabled": True, "reason": "review artefact"},
    }
```

Then update the Streamlit body to call these helpers for knob state, chart
payload, and export controls.

- [ ] **Step 4: Run helper tests to verify green**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest tests/test_spo_studio_helpers.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Manually launch Studio for smoke verification**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/streamlit run tools/spo_studio.py --server.headless true --server.port 8509
```

Expected: Streamlit starts locally. Stop it after confirming startup. If Streamlit is unavailable in the local venv, record that and rely on helper tests plus docs.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add tools/spo_studio.py tests/test_spo_studio_helpers.py
git commit -m "feat: wire SPO Studio workflow helpers" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 7: Documentation, API Reference, and Roadmap Reconciliation

**Files:**
- Create: `docs/guide/spo_studio_operator.md`
- Create: `docs/guide/auto_binding.md`
- Create: `docs/guide/autotune_learners.md`
- Create: `docs/guide/hierarchy_live_adapters.md`
- Create: `docs/reference/api/studio.md`
- Modify: `mkdocs.yml`
- Modify: `docs/roadmap.md`
- Modify: `ROADMAP.md`

- [ ] **Step 1: Add guide pages with explicit safety language**

Create each guide with SPDX header, exact command examples, input schemas, and a
section named `Safety boundary` that states replay/proposal-only behaviour.

- [ ] **Step 2: Add API reference page**

Create `docs/reference/api/studio.md`:

```markdown
<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->

# Studio Workflow API

::: scpn_phase_orchestrator.studio.workflow
```

- [ ] **Step 3: Add pages to MkDocs nav**

Modify `mkdocs.yml` to include:

```yaml
      - SPO Studio Operator: guide/spo_studio_operator.md
      - Auto-Binding: guide/auto_binding.md
      - Autotune Learners: guide/autotune_learners.md
      - Hierarchy Live Adapters: guide/hierarchy_live_adapters.md
```

Add the Studio API page under the reference/API section following the existing
API nav pattern.

- [ ] **Step 4: Reconcile roadmap status**

Mark only implemented verified slices as complete:

- Studio helper/core wiring: implemented foundation, operator UI still
  Streamlit-first.
- Auto-binding deterministic proposal package: implemented prototype.
- Replay-only learner-shaped proposal generators: implemented, no trained PPO
  or SAC performance claim.
- Hierarchy live-adapter decoded boundaries and multi-domain demos:
  implemented.

- [ ] **Step 5: Run docs and wording verification**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/python -m mkdocs build --strict --clean
rg -n "C[o]dex|O[p]enAI|C[h]atGPT|C[l]aude|G[e]mini|A[n]thropic|e[l]ite|S[U]PERIOR|S[T]RONG|E[T]ALON" docs/guide docs/reference/api/studio.md ROADMAP.md docs/roadmap.md
git diff --check
```

Expected: MkDocs passes; `rg` exits 1 with no matches; diff check passes.

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add docs/guide/spo_studio_operator.md docs/guide/auto_binding.md docs/guide/autotune_learners.md docs/guide/hierarchy_live_adapters.md docs/reference/api/studio.md mkdocs.yml docs/roadmap.md ROADMAP.md
git commit -m "docs: document industrial SPO workflow" -m "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>"
```

## Task 8: Final Verification and Session Log

**Files:**
- Create: `.coordination/sessions/SCPN_PHASE_ORCHESTRATOR/codex_2026-05-07_spo_industrial_workflow.md`

- [ ] **Step 1: Run focused verification**

Run:

```bash
env PYTHONPATH=src ./.venv/bin/pytest \
  tests/test_studio_workflow.py \
  tests/test_autotune_binding_proposal.py \
  tests/test_autotune_learners.py \
  tests/test_supervisor_hierarchy.py \
  tests/test_supervisor_hierarchy_adapters.py \
  tests/test_hierarchy_transport_demos.py \
  tests/test_spo_studio_helpers.py \
  -q
./.venv/bin/ruff check src/scpn_phase_orchestrator/studio src/scpn_phase_orchestrator/autotune src/scpn_phase_orchestrator/supervisor tools/spo_studio.py tests/test_studio_workflow.py tests/test_autotune_binding_proposal.py tests/test_autotune_learners.py tests/test_supervisor_hierarchy_adapters.py tests/test_hierarchy_transport_demos.py tests/test_spo_studio_helpers.py
./.venv/bin/mypy src/scpn_phase_orchestrator/studio src/scpn_phase_orchestrator/autotune src/scpn_phase_orchestrator/supervisor
./.venv/bin/bandit -q -r src/scpn_phase_orchestrator/studio src/scpn_phase_orchestrator/autotune src/scpn_phase_orchestrator/supervisor tools/spo_studio.py
env PYTHONPATH=src ./.venv/bin/python -m mkdocs build --strict --clean
git diff --check
```

Expected: all commands pass. If mypy or Bandit report pre-existing unrelated
issues, record exact output and run the narrower touched-file equivalent before
committing.

- [ ] **Step 2: Create the session log**

Create `.coordination/sessions/SCPN_PHASE_ORCHESTRATOR/codex_2026-05-07_spo_industrial_workflow.md` with factual summary, changed files, verification commands, and any blocked checks. Do not include credentials.

- [ ] **Step 3: Commit the session log only if project policy expects tracked internal logs**

Run:

```bash
git status --short .coordination/sessions/SCPN_PHASE_ORCHESTRATOR/codex_2026-05-07_spo_industrial_workflow.md
```

Expected: `.coordination/` is ignored. Do not force-add it.

- [ ] **Step 4: Final staged audit before any remaining commit**

Run:

```bash
git diff --cached --check
git diff --cached --name-only
rg -n "C[o]dex|O[p]enAI|C[h]atGPT|C[l]aude|G[e]mini|A[n]thropic|e[l]ite|S[U]PERIOR|S[T]RONG|E[T]ALON" $(git diff --cached --name-only)
test ! -e /media/anulum/724AA8E84AA8AA75/agentic-shared/FREEZE
```

Expected: diff check passes; staged files are explicit; wording scan returns no
matches in public files; freeze file is absent.
