# Subsystem: `studio` / `reporting` / `visualization` — operator surfaces

Human-facing review, explanation, and visualisation. `studio` 24 files,
`reporting` 5, `visualization` 4.

## `studio` — review surface

Builder functions (`build_canvas_graph`, `build_runtime_snapshot`,
`build_deployment_readiness`, `build_export_manifests`, `run_binding_spec_replay`,
…) and a registry of 13 review panels (information geometry, sheaf cohomology,
morphogenetic field, multiverse, strange-loop, evolutionary policy, lineage, …).
A `ui_helpers/` package (21 modules) provides canvas layout, deployment plans,
charts, and connector plans.

- **Inputs**: a project/binding state and runtime snapshots.
- **Outputs**: a Python-dataclass `ExportManifest` and canvas/deployment
  artefacts.
- **Scope boundary**: every panel is `execution_disabled=True` /
  `operator_review_required=True`. The runtime server emits a read-only
  `studio.control-feed.v1` envelope at `/api/studio-feed`, with SPO-specific
  live state under `runtime.schema=spo.studio-runtime-snapshot.v1`. The feed is
  additive to the existing local dashboard and WebSocket observer; it does not
  enable hardware writes, QPU execution, or policy promotion.
- **Federation manifest**: `studio/federation_manifest.py` builds the optional
  schema-A `CapabilityManifest` for STUDIO federation. The local fields are
  `transport_profile=local-first`,
  `evidence_types=["spo.runtime-state.v1", "spo.phase-coherence.v1",
  "spo.regime-state.v1"]`, `ui_module=None`, `contract_era=v1`, and
  `enumeration=language-agnostic`. The public architecture manifest mirrors
  those fields, and the focused manifest tests run the current STUDIO Platform
  schema-A federation gate against the emitted wire form.

## `reporting`

`CoherencePlot` (Matplotlib PNG/SVG), narrative explainability (policy-decision
trace), tabular summaries, and an operator-copilot advisor. Functional, not stub.

## `visualization`

JSON serialisers for a Three.js torus (`torus_points_json`, `phase_wheel_json`),
a D3 network graph (`network_graph_json`, `coupling_heatmap_json`), and a
WebSocket frame streamer. The client-side JavaScript that renders these payloads
is not in this repository.
