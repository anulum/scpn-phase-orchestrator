# Subsystem: `studio` / `reporting` / `visualization` — operator surfaces

Human-facing review, explanation, and visualisation. `studio` 23 files
(~8.3k LOC), `reporting` 5, `visualization` 4.

## `studio` — review surface

Builder functions (`build_canvas_graph`, `build_runtime_snapshot`,
`build_deployment_readiness`, `build_export_manifests`, `run_binding_spec_replay`,
…) and a registry of 13 review panels (information geometry, sheaf cohomology,
morphogenetic field, multiverse, strange-loop, evolutionary policy, lineage, …).
A `ui_helpers/` package (20 modules) provides canvas layout, deployment plans,
charts, and connector plans.

- **Inputs**: a project/binding state and runtime snapshots.
- **Outputs**: a Python-dataclass `ExportManifest` and canvas/deployment
  artefacts.
- **Scope boundary**: every panel is `execution_disabled=True` /
  `operator_review_required=True`. The surface does **not** emit a live JSON
  studio feed (unlike the `studio.control-feed.v1` of the `scpn-control`
  vertical). It is wired only into the CLI review commands, not the core loop.
  Wiring it into a live STUDIO feed is an open item.

## `reporting`

`CoherencePlot` (Matplotlib PNG/SVG), narrative explainability (policy-decision
trace), tabular summaries, and an operator-copilot advisor. Functional, not stub.

## `visualization`

JSON serialisers for a Three.js torus (`torus_points_json`, `phase_wheel_json`),
a D3 network graph (`network_graph_json`, `coupling_heatmap_json`), and a
WebSocket frame streamer. The client-side JavaScript that renders these payloads
is not in this repository.
