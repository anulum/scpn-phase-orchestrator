# Subsystem: `monitor` — dynamical observer array + STL

Observes the integrated dynamics and produces metrics the supervisor acts on.
89 files, ~18.5k LOC; of these, 30 are metric-producing observers and one is an
STL system, the remainder being per-language backend forwarders (46) and example
fixtures (8).

## Inputs

Per observer: `phases` `(N, T)` or `phases_trials` `(N_trials, T)` wrapped to
`[0, 2π)`; the coupling matrix `knm` `(N, N)`; sliding windows for time-resolved
measures; for twin measures, model-versus-observed phases and order parameter;
for STL, a `dict[str, list[float]]` trace keyed by signal name.

## Outputs

Frozen per-observer result dataclasses, e.g. `ChimeraState` (chimera index),
`LyapunovState` (`is_stable`), `CorrelationDimensionResult` (D2, Kaplan–Yorke),
`RQAResult` (determinism, laminarity), transfer entropy in nats,
`TwinConfidenceScore` (confidence ∈ [0,1] + status), sleep stage string, and an
STL `STLTraceResult` (robustness, satisfied).

## Processing model (selected)

- **Order/coherence**: Kuramoto R, ψ; chimera via local order parameter.
- **Chaos/stability**: Lyapunov spectrum (Benettin QR re-orthogonalisation).
- **Information**: transfer entropy (1-step Markov), ITPC, partial information
  decomposition (redundancy/synergy), an *approximate* integrated-information
  proxy (explicitly not exact IIT).
- **Topology/recurrence**: winding number, RQA (Eckmann/Marwan), correlation
  dimension, normalised persistent entropy, delay embedding.
- **Twin / assurance**: `twin_confidence` (Jensen–Shannon + Wasserstein-1)
  feeding `twin_conformal_gate` — a distribution-free conformal admission gate
  (Adaptive Conformal Inference, Gibbs & Candès 2021). The generic
  `runtime.simulation.simulate()` loop can now consume a deployment-provided
  `TwinConfidenceScore` source and calibrated `TwinConformalGate`; rejected
  conformal ticks suppress all proposed policy actions for that control interval
  and are surfaced in result/audit records.
- **Early warning**: ordinal-pattern transition entropy (`opt_entropy`).
- **Runtime verification**: `stl/` — an `rtamt`-backed STL monitor plus
  automaton/controller synthesis and action projection (optional dependency).

Papers are cited per module in code and are not vouched for here.

## Backends

Five-language fallback per kernel: **Rust → Mojo → Julia → Go → Python**, with
each observer's `_load_*_fns()` forwarding to
`experimental/accelerators/monitor/`.

## Wiring

Fed by `upde` output; consumed by `supervisor` (regime conditions, policy DSL,
STL into `policy_rules`). Observers are invoked ad hoc per call site rather than
through a single registry fan-out. The conformal twin-confidence gate is opt-in
at the live policy admission point: callers supply the observed-twin confidence
stream, and the core fails closed on rejected ticks by dropping the current
action set.

## Scope boundaries

- The default CLI run has no physical observed-twin feed, so conformal admission
  remains opt-in rather than automatically enabled for every binding spec.
- `oscillation_modes`, `modal_participation`, `koopman_edmd`, and
  `phase_koopman` are not removed: they are reviewed APIs with focused tests and
  are consumed by the dVOC oscillation-damping/Koopman review path where a plant
  boundary is explicit.
- `psychedelic` is a forward simulator, not a live observer; the integrated-
  information module is a labelled proxy.
