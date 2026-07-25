# Subsystem: `studio` / `reporting` / `visualization` — operator surfaces

Human-facing review, explanation, and visualisation. `studio` 25 files,
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
  `evidence_types=["studio.runtime-state.v1", "studio.phase-coherence.v1",
  "studio.regime-state.v1"]`, a pull-deployed `./SpoStudioPanel` `ui_module`,
  `contract_era=v1`, and `enumeration=language-agnostic`. Verb declarations
  include their hard functional `consumes`/`produces` edges; those edges are
  covered by the content digest and resolve without hidden upstreams. The public
  architecture manifest mirrors the complete wire form, and the focused
  manifest tests run the current STUDIO Platform schema-A federation gate
  against it.
- **Schema-B evidence**: `studio/evidence_bundles.py` reduces a validated live
  snapshot to immutable, content-addressed runtime-state, phase-coherence, and
  regime-state artifacts. Each artifact is bound into a Platform
  `EvidenceBundle`, admitted through the era-v2 federation gate only in
  `boundary` mode, and records the `numerical-model` substrate. Numeric seal
  fields are shortest-round-trip strings rather than JSON floats; the producer
  recursively rejects any float before canonicalization.

## `reporting`

`CoherencePlot` (Matplotlib PNG/SVG), narrative explainability (policy-decision
trace), tabular summaries, and an operator-copilot advisor. Functional, not stub.

## `visualization`

JSON serialisers for a Three.js torus (`torus_points_json`, `phase_wheel_json`),
a D3 network graph (`network_graph_json`, `coupling_heatmap_json`), and a
WebSocket frame streamer. The client-side JavaScript that renders these payloads
is not in this repository.
