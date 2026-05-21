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
