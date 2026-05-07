<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SCPN Phase Orchestrator — Industrial Workflow Design -->

# SPO Industrial Workflow Design

Date: 2026-05-07

This design covers the approved integrated vertical programme for four open
roadmap tracks:

- One-click SPO Studio web UI.
- Auto-binding from time-series, event logs, and graph signals to reviewable
  `binding_spec.yaml`.
- Replay-only PPO/SAC or hybrid physics learner surfaces behind existing
  non-actuating autotune gates.
- N-channel hierarchy live-adapter boundaries and broader multi-domain demos.

The implementation must be wired end to end. A feature is not complete if it
only adds a widget, an algorithm helper, or a standalone demo without audit
records, tests, documentation, and pipeline integration.

## Current Project State

The repository already contains foundations that should be reused:

- `tools/spo_studio.py` provides a Streamlit exploration UI for domainpack
  selection, `K`, `zeta`, and `Psi` controls, and simple `R` charts.
- `tools/binding_spec_studio.py` provides binding YAML load, validation,
  mapping display, synthetic extractor preview, and download.
- `src/scpn_phase_orchestrator/autotune/pipeline.py` identifies frequencies
  and coupling from multichannel time-series.
- `src/scpn_phase_orchestrator/autotune/policy_search.py` provides
  replay-only search and adaptive replay refinement around audit-ready policy
  proposals.
- `src/scpn_phase_orchestrator/supervisor/hierarchy.py` contains reduced
  child-summary planning, sync envelopes, offline gossip replay, and the
  stateful non-socket `HierarchyTransportRuntime` boundary.

The existing tools are valuable but are too UI-local for industrial use. The
new work should move behaviour into tested package modules and keep Streamlit
as a thin operator surface.

## Approach

Use one shared workflow spine:

1. Import data or load an existing domainpack.
2. Produce an explicit project model and audit-safe project summary.
3. Generate or edit a reviewable binding proposal.
4. Validate and simulate the proposal without applying live actuation.
5. Run replay-only tuning candidates behind existing proposal gates.
6. Visualise live `R`, `Psi`, `K`, regimes, and hierarchy health in Studio.
7. Export reproducible artefacts for review, replay, Docker/WASM packaging,
   and live-adapter handoff.

The shared spine keeps UI, auto-binding, learners, and hierarchy adapters from
forking their own state formats.

## Components

### Workflow Core

Add a package-level workflow module, for example
`scpn_phase_orchestrator.studio.workflow`, that owns the serialisable project
state. It should expose small dataclasses for:

- Imported source summaries: input kind, channel count, sample count, event
  count, graph node/edge counts, validation warnings, and deterministic hashes.
- Binding proposals: YAML text, validation diagnostics, inferred channels,
  oscillator families, coupling summary, confidence evidence, and provenance.
- Runtime snapshots: `R`, `Psi`, selected knob values, regime, layer metrics,
  hierarchy watermarks, and replay-only learner status.
- Export manifests: file names, content hashes, target kind, safety posture,
  required commands, and warnings.

The core should have no Streamlit dependency and no network ownership. It
should be usable from tests, CLI helpers, notebooks, and the web UI.

### One-Click SPO Studio

Keep Streamlit as the first implementation target because the repository
already uses it and it avoids introducing a JavaScript build stack. The Studio
upgrade should provide:

- Project tabs for load/import, binding editor, oscillator canvas, live run,
  autotune review, hierarchy monitor, and exports.
- Drag/drop-style oscillator editing using Streamlit data editors and graph
  tables first, with a real canvas component deferred until it is justified by
  user workflow friction.
- Live `R`, `Psi`, `K`, `alpha`, `zeta`, channel weights, cross-channel gains,
  regime timeline, and layer metrics.
- Tuning controls that only update local simulation or replay candidates unless
  an explicit reviewed export path is used.
- Export buttons for `binding_spec.yaml`, policy proposal JSON, audit summary
  JSON, Docker command manifest, WASM manifest, and hierarchy adapter manifest.

Studio must not hide validation failures. If a binding, learner proposal, or
transport record is unsafe, the UI should show the exact rejected reason and
disable export paths that would imply deployability.

### Auto-Binding Prototype

The auto-binding layer should accept three deterministic input families:

- Time-series CSV or matrix: channels by samples, sample rate, optional labels.
- Event-log JSON/JSONL: named events with timestamps, source IDs, and optional
  weights.
- Graph JSON: nodes, edges, optional time-indexed node signals, and channel
  labels.

The output is a binding proposal package, not an automatically trusted binding.
It must include:

- Proposed oscillator families and extractor types.
- Proposed channels and layer assignments.
- Estimated frequencies, coupling matrix summary, and N-channel algebra
  evidence where available.
- Confidence factors and reasons for low confidence.
- Validation diagnostics from the existing binding validator.
- Reproducible YAML text suitable for human review.

The initial algorithms should stay deterministic and transparent: Hilbert or
existing phase extraction for time-series, event cadence phase estimates for
event logs, and graph/ring symbolic extractors for graph inputs. More advanced
graph-learning can be added later behind the same proposal interface.

### Replay-Only Learners

The PPO/SAC requirement should be represented as learner-shaped proposal
interfaces before introducing heavyweight training dependencies. The first
industrial slice should provide:

- A `LearnerPolicyProposal` contract that records learner kind, seed policy,
  replay observations, reward configuration, safety gate result, and accepted
  proposal.
- Deterministic PPO-like and SAC-like candidate generators that exercise the
  same action-space and audit surfaces as future learners, without claiming
  trained policy performance.
- A hybrid physics learner that combines existing coordinate refinement with a
  physics-informed candidate prior and the existing replay evaluator.
- Explicit non-actuation guarantees: learners can only emit proposals, never
  control actions or hardware writes.

If real PPO/SAC dependencies are added later, they must plug into this contract
and pass the same replay, safety, and audit tests before any production claim.

### N-Channel Hierarchy Adapters

Keep supervisor hierarchy code transport-neutral. Add adapter-boundary modules
that validate decoded records and feed `HierarchyTransportRuntime`:

- JSONL replay adapter for reproducible live-adapter smoke tests.
- REST boundary helpers that accept decoded request payloads and headers but do
  not own an HTTP server.
- WebSocket-style frame boundary that accepts decoded frames and emits accepted
  or rejected ledger records without owning sockets or event loops.

Each adapter should report protocol version, source node, sequence watermark,
rejected reason, and parent-plan summary. Broader multi-domain demos should
cover at least power grid, cardiac rhythm, and one N-channel domainpack with
heterogeneous channels.

## Safety And Non-Goals

- No live actuation is introduced by this programme.
- No socket, broker, thread, or hardware handle should be owned by the
  supervisor hierarchy module.
- Studio is allowed to run local simulation and replay only.
- Auto-binding produces proposals; it does not silently overwrite domainpacks.
- PPO/SAC-shaped interfaces must not claim trained learner performance unless
  benchmarks are actually measured and documented.
- Export manifests must distinguish review artefacts from deployable artefacts.

## Test Requirements

Implementation must use test-first development for production behaviour. The
minimum coverage by slice is:

- Workflow core: serialisation round trips, hash stability, rejected invalid
  input summaries, export-manifest safety flags, and no Streamlit dependency.
- Auto-binding: CSV/event/graph fixtures, malformed input rejection, confidence
  evidence, binding validator integration, deterministic YAML output, and
  pipeline compatibility with an existing domainpack run path.
- Studio helper layer: pure helper functions for project state, knob updates,
  chart payloads, export payloads, and disabled unsafe export states.
- Learner proposals: PPO-like, SAC-like, and hybrid generators produce
  deterministic candidates; unsafe replay observations are rejected; accepted
  proposals are audit serialisable; no actuation object is emitted.
- Hierarchy adapters: stale sequence rejection, protocol mismatch rejection,
  JSONL replay determinism, decoded REST/frame validation, watermark
  persistence, and parent-plan integration.
- Demos: multi-domain examples produce reproducible audit records and do not
  require network or hardware access.

Focused verification should include Ruff, mypy, Bandit, relevant pytest files,
strict MkDocs, public wording scan, staged diff audit, and freeze check before
any commit.

## Documentation Requirements

Documentation must be updated with:

- A Studio operator guide showing import, binding review, replay tuning, live
  metrics, and export paths.
- An auto-binding guide with supported input schemas, confidence interpretation,
  and review workflow.
- An autotune learner guide that clearly states the non-actuating replay gate
  and distinguishes deterministic proposal generators from trained PPO/SAC.
- A hierarchy live-adapter guide that explains decoded boundary adapters,
  sequence watermarks, rejection reasons, and multi-domain replay demos.
- API reference pages for new public modules.
- Roadmap updates marking only implemented, verified slices as done.

## Delivery Sequence

Use small commits with explicit staging:

1. Workflow core and tests.
2. Auto-binding proposal package and tests.
3. Replay learner proposal interfaces and tests.
4. Hierarchy adapter boundaries and multi-domain demos.
5. Studio UI upgrade wired through the tested core.
6. Documentation, benchmarks, and roadmap reconciliation.

Each commit must be independently verifiable and must not include unrelated
workspace changes. Existing uncommitted hierarchy work should either be
completed first as its own slice or left untouched by this programme until the
owner decides how to integrate it.
