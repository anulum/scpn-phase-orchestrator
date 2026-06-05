# `scpn_phase_orchestrator.nn.reservoir` contract

Purpose:
- Provide Kuramoto-reservoir feature extraction, linear readout fitting, and
  prediction surfaces for differentiable reservoir workflows.

Public contract:
1. `reservoir_features(phases)` returns `(2*N + 1,)` with final scalar equal to
   order parameter `R`.
2. `reservoir_drive(...)` returns `(T, 2*N + 1)` and responds to input changes.
3. `ridge_readout(...)` + `reservoir_predict(...)` preserve linear regression
   shape contracts and support exact-fit low-regularisation cases.

Verification:
- `tests/test_reservoir.py::TestReservoirFeatures::test_output_shape`
- `tests/test_reservoir.py::TestReservoirDrive::test_output_shape`
- `tests/test_reservoir.py::TestRidgeReadout::test_perfect_linear_fit`
- `tests/test_reservoir.py::TestReservoirPredict::test_output_shape`

## Why this contract is production-relevant

Reservoir methods are used when SPO needs compact, model-derived feature streams
for lightweight predictors. Keeping shape and deterministic mapping guarantees
means these outputs can drive downstream controllers without custom glue per
experiment.

The linear readout contract is specifically relevant for explainability and
deployment reproducibility because it defines exactly which model dimensions are
trusted for prediction in the differentiable path.

## Deployment value

- Reservoir extraction is the lightweight bridge from nonlinear phase dynamics to
  downstream regression or controller policies.
- Deterministic feature shape is the minimum requirement for reproducible model
  cards and replay-based diagnostics across environments.
- The ridge predict path is used where explainability and compactness are as
  important as raw predictive accuracy.
