# `scpn_phase_orchestrator.nn.chimera` contract

Purpose:
- Provide differentiable chimera diagnostics (`local_order_parameter`,
  `chimera_index`, `detect_chimera`) for coupled oscillator states.

Public contract:
1. Local order parameter output is shape-stable and bounded in `[0, 1]`.
2. Chimera index returns finite scalar variance metric.
3. Coherent and incoherent masks from `detect_chimera` remain disjoint for
   ordered thresholds (`coherent_threshold > incoherent_threshold`).

Verification:
- `tests/test_nn_chimera.py::TestLocalOrderParameter::test_range`
- `tests/test_nn_chimera.py::TestChimeraIndex::test_scalar_output`
- `tests/test_nn_chimera.py::TestDetectChimera::test_threshold_masks_disjoint`

## Production interpretation

This contract supports state diagnostics in mixed-coherence regimes. In practice,
chimera indicators are used as an early warning when the system is moving from a
globally synchronized phase toward fragmented structure.

By keeping output semantics bounded and threshold masks disjoint, supervisory
logic can consume the diagnostic signals without introducing contradictory
interpretation paths.

## Why this contract matters in operations

- In high-asset environments, partial synchrony is often operationally important.
  You need to know where coherence is localised before applying global
  control.
- The disjoint-coherence invariant prevents a single oscillator from being
  simultaneously tagged as both coherent and incoherent at one threshold pair,
  which protects downstream action logic from ambiguous telemetry.
- Bounded outputs reduce the chance that monitor fusion layers accidentally
  interpret numerical drift as structural drift.

## Operator interpretation

Partial synchrony diagnostics are often used in escalation decisions where global
coherence is not yet a sufficient signal. Teams may still need a coherent subgroup
to stabilise while preserving other modes that carry important structure.

This contract therefore serves as a semantic guard around ambiguous transitions:
it gives strict rules so alerting and supervisor logic can separate mixed regimes
without forcing a binary interpretation too early.

The bounded and disjoint outputs are what make it safe to use in chained monitoring
pipelines, because each alarm signal maps cleanly to a single interpretation.

## Operational narrative

The contract is useful whenever a control stack needs a stable indicator for
mixed synchrony rather than a binary "locked vs. unlocked" answer.

In these conditions, `chimera_index` and `detect_chimera` provide interpretable
telemetry for partial-order states. This is frequently where physical systems
enter practical instability first: not as a global meltdown, but as a shift into
patchy coherence.

### Why this matters in review and handoff

- `local_order_parameter` provides a bounded local coherence signal that can be
  compared across layers and domains.
- `chimera_index` collapses regional coherence asymmetry into a single scalar for
  alert dashboards.
- Threshold masks with explicit disjoint constraints reduce ambiguity at escalation
  points where operator policy decisions are made.
