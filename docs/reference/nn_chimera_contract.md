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
