# `scpn_phase_orchestrator.nn.spectral` contract

Purpose:
- Provide differentiable spectral synchronisability metrics from coupling
  matrices.

Public contract:
1. `laplacian_spectrum(K)` returns sorted `(N,)` eigenvalue vector.
2. `algebraic_connectivity(K)` returns scalar λ₂ and supports gradients.
3. `sync_threshold(K, omegas)` remains finite for disconnected/degenerate
   graphs via denominator floor.

Verification:
- `tests/test_nn_spectral.py::TestLaplacianSpectrum::test_shape`
- `tests/test_nn_spectral.py::TestAlgebraicConnectivity::test_differentiable`
- `tests/test_nn_spectral.py::TestSyncThreshold::test_disconnected_graph_finite_threshold`

## Why this contract is important

Spectral metrics are frequently used for stability reasoning and graph health checks
before actuation policies are applied. This contract ensures both topology and
dynamics inputs stay in a numerically safe domain for those analyses.

Finite fallback behaviour for disconnected or degenerate graphs is a practical
safety clause: it avoids silent numerical failures in large, irregular coupling
topologies.

## Operational interpretation

- This contract is the first checkpoint before control proposals are promoted from
  simulation to action planning.
- A bounded `sync_threshold` in disconnected cases allows the same policy code to
  run on sparse production topologies without special-case handling for empty
  components.
- Deterministic ordering of the spectrum is required for reproducible stability
  reports across audit snapshots.

## Why this contract sits in the front of safety-critical loops

Spectral checks are usually the earliest diagnostic in this stack because they
signal whether a coupling graph can sustain coherent control action before policy
logic is allowed to act.

In a production lane, this contract gives an invariant baseline: topology-related
failure modes are bounded through explicit eigenvalue ordering and disconnected-graph
fallbacks, so downstream policy code does not consume unstable intermediate values.

By keeping this contract narrowly defined and machine-testable, the same module
can serve both offline science experiments and regulated deployment evidence trails.
