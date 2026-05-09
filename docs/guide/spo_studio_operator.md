<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — SPO Studio operator guide -->

# SPO Studio Operator Guide

SPO Studio is the Streamlit operator surface for local replay, binding review,
auto-binding proposals, oscillator edits, live metrics, hierarchy visibility,
connector ownership review, and review artefact export. It is intentionally a
thin UI over
`scpn_phase_orchestrator.studio.ui_helpers`, so the behaviour can be tested
without Streamlit.

The current implementation is a validated operator prototype, not a finished
product-grade Studio. It is useful for auditable replay, binding proposal,
metric inspection, and export review workflows. It still needs true drag/drop
layout polish, richer guided onboarding, live connector execution, package
materialisation commands, and hardware-target packaging before it should be
described as a good standalone product.

Run it with:

```bash
streamlit run tools/spo_studio.py
```

## Workflow

1. Select a domainpack from the sidebar.
2. Set the replay-only knobs: `K`, `alpha`, `zeta`, and `Psi`.
3. Run replay. Studio builds a `StudioProjectState` with source, binding,
   runtime, and export records.
4. Review the tabs:
   - **Load**: current source summary and raw-source import for CSV, event-log
     JSON, or graph JSON binding proposals.
   - **Guide**: beginner-mode runtime summary, signal/coupling/objective/
     supervisor explanations in domain terms, next actions, and
     `beginner_guidance.json`.
   - **Binding**: generated or loaded YAML plus validation diagnostics.
   - **Oscillators**: editable oscillator table. Edits produce an
     `oscillator_edit_review.json` artefact rather than silently changing a
     live binding.
   - **Canvas**: editable layer/channel graph rows for the current binding.
     Nodes and cross-channel coupling edges produce a
     `canvas_edit_review.json` artefact and `canvas_layout_manifest.json`
     rather than silently changing a live binding.
   - **Live**: `R`, regime timeline, and per-layer metrics from local replay.
   - **Autotune**: replay-only status and knob record. No actuation is enabled.
   - **Hierarchy**: current hierarchy watermarks and reduced layer metrics.
   - **Connectors**: memory, JSONL, REST, gRPC, Kafka, and hardware connector
     ownership plan, contract hash, auth posture, and `connector_plan.json`.
   - **Exports**: deployment-readiness checklist, deployment package manifest,
     hardware target package, plus review artefacts for binding YAML, audit
     JSON, Docker manifest, WASM manifest, and project state.

## Guided Deployment Path

The **Exports** tab emits `deployment_readiness.json` and
`deployment_package.json` before the individual artefact downloads. It also
shows a beginner checklist in execution order: run local replay, validate the
binding, review Docker packaging, review WASM packaging, then attach hardware
evidence when available. A separate command table exposes only currently
reviewable commands, so blocked targets and hardware-without-evidence do not
emit command rows.

The readiness JSON gives each target a status and the next operator action:

- `docker`: ready when binding validation passes; review `binding_spec.yaml`,
  `spo_studio_audit.json`, and `docker_manifest.json` before packaging. The
  checklist includes the review commands for `docker compose config`, local
  image build, and local replay inside the image.
- `wasm`: ready when binding validation passes; review browser-safe replay
  constraints and the `wasm_manifest.json` artefact. The checklist includes the
  `wasm-pack` build command for the browser demo artefact.
- `hardware`: postponed until verified target evidence is attached. Studio does
  not mark hardware packaging ready from a local replay alone, and it emits no
  hardware command until that evidence exists.

If binding validation fails, all targets are blocked and the checklist carries
the validation messages as `blocked_reasons`. This keeps review artefacts
available while preventing deploy-like manifests from being treated as ready.

The package JSON gathers the same target readiness with export payload hashes,
required artefacts, review commands, blocked reasons, and safety gates. It is a
single handover manifest for packaging jobs; it does not build images, run
`wasm-pack`, open transports, or enable hardware output by itself.

`hardware_target_package.json` is stricter. It records FPGA Verilog and
neuromorphic schedule as target classes, but it remains `evidence_required`
until a generated artefact path, simulator parity report, target toolchain
version, and operator sign-off are attached. It keeps `hardware_write_permitted`
false and points operators back to the connector plan before any handoff.

## Canvas Review

The **Canvas** tab exposes the binding as a deterministic graph with layer
nodes, declared channel nodes, and cross-channel coupling edges. It is designed
for product-grade review workflows before direct drag/drop layout persistence:
operators can inspect the topology, edit node or edge rows, and download a
`canvas_edit_review.json` artefact that records before/after nodes, before/after
edges, and changed counts.

Canvas edits remain review-only. They do not rewrite `binding_spec.yaml`, open a
live connector, or enable actuation. This keeps topology edits auditable until
the binding update path has an explicit validation and review step.

`canvas_layout_manifest.json` persists only node positions, labels, and counts.
It is intended for handoff and future layout restore; it does not alter topology
or validation state.

## Beginner Mode

The **Guide** tab translates the current replay into operator-facing language:
which layers and channels are being reviewed, how `K`, `alpha`, `zeta`, and
`Psi` affect the replay, whether binding validation is blocking packaging, and
what regime the supervisor currently reports. The downloadable
`beginner_guidance.json` mirrors the on-screen cards for handover and review.

Beginner guidance is still non-actuating. It reads the replay result, canvas
graph, validation state, and runtime snapshot; it does not change the binding,
run live connectors, or enable hardware output.

## Connector Ownership

The **Connectors** tab turns the digital-twin binding contract into a review
plan for memory, JSONL, REST, gRPC, Kafka, and hardware transports. Offline
connectors are marked review-ready. Live transports are marked owner-required
with authentication required, and the hardware connector explicitly keeps
`hardware_write_permitted` false.

`connector_plan.json` includes the contract hash, sync capabilities, compatibility
result, ownership status, and safety flags. Studio does not open sockets, import
broker clients, start a gRPC server, or write to hardware.

## Error Recovery

Replay and source-import failures render `studio_error_report.json` instead of
exposing raw exception text. The report includes the operation, project name,
exception type, blocked status, and the next operator action. It intentionally
does not echo local paths, uploaded content, or raw exception messages.

## Safety Posture

Studio does not open hardware handles, run live transport, or actuate a target.
Connector plans are review records only. Knob changes alter only local replay.
Validation failures keep review artefacts available with warnings, while
deploy-like manifests are disabled and carry explicit disabled reasons.
