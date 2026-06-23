# Subsystem: `supervisor` — regime FSM, policy, MPC, formal export

Turns observer metrics into bounded, audit-ready control proposals. 45 files,
~21.8k LOC (the largest subsystem). Pure NumPy — no Rust/JAX dispatch (the
`spo-supervisor` Rust crate exists but is not imported here).

## Inputs

- `UPDEState` (from `upde`) and `BoundaryState` (from `monitor/boundaries`).
- Metric mappings `Mapping[str, float]` (finiteness-guarded).
- Signatures: `RegimeManager.evaluate(upde_state, boundary_state) → Regime`;
  `SupervisorPolicy.decide(upde_state, boundary_state, …) → list[ControlAction]`.

## Outputs

- `ControlAction` (knob, scope, value, ttl_s, justification) — proposals only.
- `Regime` ∈ {NOMINAL, DEGRADED, CRITICAL, RECOVERY}.
- `Prediction` / free-energy assessment; PRISM and TLA+ export text; runtime
  certificates. Every output type carries `to_audit_record() → dict`.

## Processing model

- **Regime FSM** (`regimes.py`): hysteresis + cooldown classifier on the order
  parameter, bounded transition history.
- **Petri net** (`petri_net.py`, `petri_adapter.py`): guarded, first-match,
  ≤1 transition per step.
- **Policy DSL** (`policy_rules.py`): regime-conditioned condition/action rules
  with cooldowns and fire limits; an STL automaton path.
- **Predictive MPC** (`predictive.py`): Ott–Antonsen reduced-model horizon
  forecast; a variational free-energy variant.
- **Formal export** (`formal_export/`): Petri net / policy / STL → PRISM and
  TLA+ text, a verification package, and a runtime certificate.
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

## Scope boundaries

- Several lanes are **offline / review-only**: federated transport (socket
  stubs, no live listener), causal counterfactual (batch, not reactive),
  evolutionary/topos grammar, multiverse branches, `*_examples` modules.
- Formal export covers PRISM and TLA+; there is **no SMT-LIB exporter** despite a
  docstring reference to one.
- No god-file (largest module ~1.1k LOC).
