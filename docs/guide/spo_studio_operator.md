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
and review artefact export. It is intentionally a thin UI over
`scpn_phase_orchestrator.studio.ui_helpers`, so the behaviour can be tested
without Streamlit.

The current implementation is a validated operator prototype, not a finished
product-grade Studio. It is useful for auditable replay, binding proposal,
metric inspection, and export review workflows. It still needs true drag/drop
graph editing, guided beginner explanations, live connector ownership,
deployment packaging, polished error recovery, and hardware-target packaging
before it should be described as a good standalone product.

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
   - **Binding**: generated or loaded YAML plus validation diagnostics.
   - **Oscillators**: editable oscillator table. Edits produce an
     `oscillator_edit_review.json` artefact rather than silently changing a
     live binding.
   - **Live**: `R`, regime timeline, and per-layer metrics from local replay.
   - **Autotune**: replay-only status and knob record. No actuation is enabled.
   - **Hierarchy**: current hierarchy watermarks and reduced layer metrics.
   - **Exports**: deployment-readiness checklist plus review artefacts for
     binding YAML, audit JSON, Docker manifest, WASM manifest, and project
     state.

## Guided Deployment Path

The **Exports** tab emits `deployment_readiness.json` before the individual
artefact downloads. It also shows a beginner checklist in execution order:
run local replay, validate the binding, review Docker packaging, review WASM
packaging, then attach hardware evidence when available. The readiness JSON
gives each target a status and the next operator action:

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

## Safety Posture

Studio does not open hardware handles, run live transport, or actuate a target.
Knob changes alter only local replay. Validation failures keep review artefacts
available with warnings, while deploy-like manifests are disabled and carry
explicit disabled reasons.
