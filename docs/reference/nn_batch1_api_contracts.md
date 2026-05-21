# NN batch-1 API contracts (low-coverage closure tranche)

This document records per-file public behavioural contracts for the first
batch of low-coverage `nn/` modules targeted in coverage debt reduction.

## `scpn_phase_orchestrator.nn.supervisor`

Contract:
1. Public helper validators reject malformed scalar and shape metadata early.
2. Replay/audit helper JSON serialisation paths convert non-finite floats into
   explicit safe sentinels.
3. Internal metric prefixing helper accepts finite numeric payloads and rejects
   non-finite values.

Verification:
- `tests/test_coverage_batch1_low20.py::test_nn_supervisor_validation_helpers_and_json_paths`

## `scpn_phase_orchestrator.nn.functional`

Contract:
1. Masked and unmasked Kuramoto/Winfree stepping keep phase vectors bounded on
   `[0, 2π)` and preserve expected tensor shapes.
2. SAF helper paths operate on finite coupling/frequency arrays without
   producing non-finite outputs.

Verification:
- `tests/test_coverage_batch1_low20.py::test_nn_functional_masked_and_winfree_paths`
- `tests/test_coverage_batch1_low20.py::test_nn_functional_plv_and_laplacian_shapes`

## `scpn_phase_orchestrator.nn.inverse`

Contract:
1. Window extraction helper must generate deterministic `(starts, targets)`
   tensor shapes for fixed `window_size`.
2. Coupling symmetrisation helper must enforce zero diagonal and symmetric
   off-diagonal entries.

Verification:
- `tests/test_coverage_batch1_low20.py::test_nn_inverse_window_and_symmetry_helpers`
- `tests/test_coverage_batch1_low20.py::test_nn_inverse_window_helper_rejects_empty_window`

## `scpn_phase_orchestrator.nn.oim`

Contract:
1. OIM stepping keeps phase vectors shape-stable.
2. Soft colouring extraction returns one label per oscillator.
3. Continuous energy/violation helpers return finite non-negative outputs.

Verification:
- `tests/test_coverage_batch1_low20.py::test_nn_oim_energy_and_violation_paths`
- `tests/test_coverage_batch1_low20.py::test_nn_oim_coloring_violations_zero_for_disconnected_graph`

## `scpn_phase_orchestrator.nn.ude`

Contract:
1. Residual coupling module must emit finite scalar corrections.
2. UDE layer forward path preserves oscillator vector shape.

Verification:
- `tests/test_coverage_batch1_low20.py::test_nn_ude_residual_and_layer_shape_contract`

## `scpn_phase_orchestrator.nn.theta_neuron`

Contract:
1. Theta-neuron stepping returns bounded phase vectors with stable shape.
2. Theta layer forward path preserves oscillator count.

Verification:
- `tests/test_coverage_batch1_low20.py::test_nn_theta_neuron_step_and_layer_contract`
