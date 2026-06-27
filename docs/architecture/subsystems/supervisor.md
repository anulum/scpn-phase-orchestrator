# Subsystem: `supervisor` — regime FSM, policy, MPC, formal export

Turns observer metrics into bounded, audit-ready control proposals. 47 Python
files, ~22.6k LOC (the largest subsystem). The live control path remains the Python
supervisor by default; the optional Rust `spo-supervisor` PyO3 surface is
validated through `scpn_phase_orchestrator.supervisor.rust_backend` and reported
by `spo doctor` when the `spo_kernel` wheel is installed.

## Inputs

- `UPDEState` (from `upde`) and `BoundaryState` (from `monitor/boundaries`).
- Metric mappings `Mapping[str, float]` (finiteness-guarded).
- Signatures: `RegimeManager.evaluate(upde_state, boundary_state) → Regime`;
  `SupervisorPolicy.decide(upde_state, boundary_state, …) → list[ControlAction]`.

## Outputs

- `ControlAction` (knob, scope, value, ttl_s, justification) — proposals only.
- `Regime` ∈ {NOMINAL, DEGRADED, CRITICAL, RECOVERY}.
- `Prediction` / free-energy assessment; PRISM, TLA+, and SMT-LIB export text;
  runtime certificates. Every output type carries `to_audit_record() → dict`.

## Processing model

- **Regime FSM** (`regimes.py`): hysteresis + cooldown classifier on the order
  parameter, bounded transition history.
- **Petri net** (`petri_net.py`, `petri_adapter.py`): guarded, first-match,
  ≤1 transition per step.
- **Policy DSL** (`policy_rules.py`): regime-conditioned condition/action rules
  with cooldowns and fire limits; an STL automaton path.
- **Predictive MPC** (`predictive.py`): Ott–Antonsen reduced-model horizon
  forecast; a variational free-energy variant.
- **Formal export** (`formal_export/`): Petri net / policy / STL → PRISM, TLA+,
  and generated SMT-LIB text, a verification package, and a runtime certificate.
- **Rust supervisor backend probe** (`rust_backend.py`): optional, fail-closed
  validation for `spo_kernel` symbols (`PyRegimeManager`,
  `PyBoundaryObserver`, `PyCoherenceMonitor`, `PyActiveInferenceAgent`, policy,
  Petri-net, projector, and rule-engine bindings) plus deterministic
  non-actuating smoke checks for regime classification, boundary observation,
  and coherence monitors.
- **Additional lanes**: Byzantine proposal signing (HMAC-SHA256) and offline BFT
  manifest; reduced-evidence hierarchy boundary (only R/ψ/regime/confidence cross
  a hierarchy edge — raw phases are forbidden); causal counterfactual rollouts;
  sheaf-Laplacian coherence; value-alignment guard; information-geometry control;
  morphogenetic and higher-order topology mutation; offline evolutionary search;
  federated aggregation manifest; multiverse branch evaluation.

## Wiring

The server constructs `RegimeManager` + `SupervisorPolicy`; `policy.decide()`
runs each step and returns `ControlAction`s for review (not auto-applied).
Formal export is reachable through the `formal-export` / `policy-dry-run` CLI.
`spo doctor` reports `rust-supervisor` readiness separately from the generic
Rust backend so operators can distinguish "FFI wheel importable" from "the
supervisor PyO3 contract is actually usable".

## Scope boundaries

- Several lanes are **offline / review-only**: federated transport (socket
  stubs, no live listener), causal counterfactual (batch, not reactive),
  evolutionary/topos grammar, multiverse branches, `*_examples` modules.
- Rust supervisor validation is an optional readiness probe, not a live-control
  dispatch switch; missing Rust supervisor bindings are warnings while the
  Python supervisor remains operational.
- No god-file (largest module ~1.1k LOC).
