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
   - **Exports**: review artefacts for binding YAML, audit JSON, Docker manifest,
     WASM manifest, and project state.

## Safety Posture

Studio does not open hardware handles, run live transport, or actuate a target.
Knob changes alter only local replay. Validation failures keep review artefacts
available with warnings, while deploy-like manifests are disabled and carry
explicit disabled reasons.
