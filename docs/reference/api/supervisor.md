# Supervisor

The supervisor subsystem adds a regime-classification and control-*proposal*
layer over oscillator dynamics, which simulate-and-observe libraries (TVB,
neurolib, Brian2, NEST) do not provide. It predicts and detects synchronisation
problems and *proposes* bounded corrections for review — it does not close a
control loop on hardware.

## Pipeline position

```
UPDEEngine.step() ──→ phases ──→ compute_order_parameter()
                                          │
                                          ↓
                                   UPDEState (R, ψ, locks)
                                          │
                        ┌─────────────────┼─────────────────┐
                        ↓                 ↓                 ↓
                 RegimeManager     PetriNetAdapter   PredictiveSupervisor
                        │                 │                 │
                        └─────────┬───────┘                 │
                                  ↓                         ↓
                          SupervisorPolicy.decide()  ←──────┘
                                  │
                                  ├──→ CausalInterventionEngine
                                  │        (baseline vs intervention rollout)
                                  │
                                  ↓
                        list[ControlAction]
                                  │
                                  ↓
                         ActionProjector.project()
                                  │
                                  ↓
                         ActuationMapper.map_actions()
```

The supervisor sits between the engine output and the next engine step.
It consumes `UPDEState` and `BoundaryState`, produces `ControlAction`
instructions that modify `K_nm`, `ζ`, `Ψ`, or `ω` for the next step.

---

## Regime Manager

Finite state machine for synchronisation regimes with hysteresis,
cooldown, and event logging.

### Regime enum

| Value | Meaning | R range (default thresholds) |
|-------|---------|-----|
| `NOMINAL` | Healthy synchronisation | R ≥ 0.6 |
| `DEGRADED` | Partial desynchronisation | 0.3 ≤ R < 0.6 |
| `CRITICAL` | Synchronisation failure | R < 0.3 or hard violation |
| `RECOVERY` | Transitioning from CRITICAL | CRITICAL → R improving |

### Safety requirement SR-3

CRITICAL **must** pass through RECOVERY before reaching NOMINAL.
Direct CRITICAL → NOMINAL is forbidden. This prevents premature
resumption of normal operation after a synchronisation failure.

### Constructor

```python
RegimeManager(
    hysteresis: float = 0.05,      # band around thresholds
    cooldown_steps: int = 10,      # steps before next transition
    event_bus: EventBus | None = None,
    hysteresis_hold_steps: int = 0,  # consecutive proposals needed
)
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `evaluate` | `(UPDEState, BoundaryState) → Regime` | Proposes regime from metrics |
| `transition` | `(Regime) → Regime` | Applies FSM rules, returns actual |
| `force_transition` | `(Regime) → Regime` | Bypasses cooldown |

### Hysteresis

To prevent oscillation between regimes when R is near a threshold,
the manager applies a hysteresis band:

```
NOMINAL → DEGRADED: requires R < threshold - hysteresis
DEGRADED → NOMINAL: requires R > threshold + hysteresis
```

`hysteresis_hold_steps` adds an additional guard: the proposed regime
must be proposed for N consecutive steps before the transition fires.
CRITICAL always bypasses this hold (safety override).

### Cooldown

After a transition, subsequent non-CRITICAL transitions are blocked
for `cooldown_steps` evaluations. CRITICAL always bypasses cooldown.

### Transition history

`transition_history: deque[tuple[int, Regime, Regime]]` stores the
last 100 transitions as (step_number, old_regime, new_regime).

**Performance:** `evaluate()` < 10 μs.

::: scpn_phase_orchestrator.supervisor.regimes

---

## Higher-Order Topology Adaptation

`HigherOrderTopologySupervisor` is the first supervisor-side topology editor.
It consumes live phases plus the current pairwise `K_nm` matrix and returns a
next-step topology:

- bounded pairwise coupling updates from local phase alignment
- optional triadic `Hyperedge` proposals when global coherence is below target
- pruning of stale or incoherent higher-order edges
- serialisable audit metadata for added/pruned simplices and pairwise delta norm

The core control knob is `TopologyMutationPolicy.mutation_rate`. A value of
`0.0` freezes topology; larger values increase the maximum per-step pairwise
and triadic changes while preserving non-negative couplings and a zero
diagonal. `TopologyMutationPolicy.simplex_pairwise_support_floor` is the
policy-hardening gate for deployment reviews: a candidate 2-simplex is only
created when every pairwise edge inside that triad is already at or above the
configured support floor.

```python
import numpy as np

from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
)
from scpn_phase_orchestrator.upde.hypergraph import HypergraphEngine

policy = TopologyMutationPolicy(mutation_rate=0.2, coherence_floor=0.8)
topology = HigherOrderTopologySupervisor(policy)
result = topology.mutate(phases, knm)

engine = HypergraphEngine(len(phases), dt=0.01, hyperedges=list(result.hyperedges))
next_phases = engine.step(phases, omegas, pairwise_knm=result.knm)
audit_payload = result.to_audit_record()
```

This slice does not claim autonomous online structural control. It provides the
auditable mutation primitive that existing policy, causal, STL, simplicial, and
hypergraph paths can gate before applying a topology change.

Domainpack demos:

- `domainpacks/plasma_control/topology_adaptation_demo.py` runs one guarded
  mutation against the plasma-control binding and prints the audit payload as
  JSON.
- `domainpacks/traffic_flow/topology_adaptation_demo.py` builds pairwise
  support from transfer-entropy evidence before proposing traffic-corridor
  simplices, then records Lyapunov before/after energy and basin evidence for
  the proposed mutation.
- `domainpacks/network_security/topology_adaptation_demo.py` builds pairwise
  support from transfer-entropy evidence before proposing traffic/attack/defence
  simplices, then records Lyapunov before/after energy evidence for the
  proposed mutation.

::: scpn_phase_orchestrator.supervisor.topology

---

## Hierarchical Orchestration Summaries

`build_hierarchical_orchestration_plan()` is the generic nested-supervisor
foundation. Child supervisors exchange bounded summaries only: child name,
channel, `R`, `psi`, regime, confidence, and optional metadata. The parent
planner converts those summaries into a reduced `UPDEState`, computes
cross-child phase alignment, and emits escalation records for low confidence,
degraded coherence, critical coherence, or explicit child-regime escalation.

```python
from scpn_phase_orchestrator.supervisor import (
    ChildSupervisorSummary,
    build_hierarchical_orchestration_plan,
)

plan = build_hierarchical_orchestration_plan(
    [
        ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0),
        ChildSupervisorSummary("edge-b", "thermal", R=0.5, psi=1.2),
    ],
    degraded_threshold=0.65,
    critical_threshold=0.35,
)

parent_state = plan.parent_state
audit_payload = plan.to_audit_record()
```

The same reduced summaries can be wrapped in deterministic sync envelopes for
JSONL replay, message-bus transport, or parent-side cloud ingestion. The parent
ingestion helper rejects stale or duplicate sequence numbers per source node
and protocol-version mismatches before building the parent orchestration plan.
Direct envelope JSON parsing uses canonical finite JSON semantics: non-finite
constants and duplicate object keys are rejected before the reduced summary is
validated or admitted to the parent watermark ledger.

```python
from scpn_phase_orchestrator.supervisor import (
    build_hierarchy_sync_envelope,
    ingest_hierarchy_sync_envelopes,
)

envelope = build_hierarchy_sync_envelope(
    ChildSupervisorSummary("edge-a", "power", R=0.9, psi=0.0),
    source_node="edge-node-a",
    sequence=42,
)

ledger = ingest_hierarchy_sync_envelopes(
    [envelope],
    previous_sequences={"edge-node-a": 41},
)
sync_audit = ledger.to_audit_record()
```

`HierarchyTransportRuntime` is the next live-transport boundary. Caller-owned
REST, gRPC, Kafka, file, or hardware adapters can pass decoded mappings or JSON
strings into the runtime; the runtime parses reduced sync records, maintains
per-source sequence watermarks across batches, and emits the same parent
ledger. It still owns no socket, thread, broker client, or actuator handle.

```python
from scpn_phase_orchestrator.supervisor import HierarchyTransportRuntime

runtime = HierarchyTransportRuntime()
batch_ledger = runtime.ingest_batch([envelope.to_json()])
runtime_audit = runtime.to_audit_record()
```

For offline distributed-edge testing, `simulate_hierarchy_gossip_consensus()`
replays local consensus over accepted sync envelopes and a caller-supplied
neighbour map. Each node updates only its reduced coherence, phase, confidence,
and audit metadata; no sockets are opened and no raw observations enter the
consensus state.

```python
from scpn_phase_orchestrator.supervisor import simulate_hierarchy_gossip_consensus

rounds = simulate_hierarchy_gossip_consensus(
    [envelope],
    neighbour_map={"edge-node-a": ()},
    rounds=1,
)
consensus_audit = [round_record.to_audit_record() for round_record in rounds]
```

This slice does not open sockets, run a gossip protocol, or perform direct
actuation. It gives existing regime, policy, FEP, causal, STL, and audit paths a
common parent-level state built from reduced child evidence without moving raw
time series, local coupling matrices, or actuator targets across hierarchy
boundaries.

Domainpack demos:

- `domainpacks/power_grid/hierarchy_sync_demo.py` replays generation and
  demand/renewable edge summaries through the sync-envelope ingestion path.
- `domainpacks/cardiac_rhythm/hierarchy_sync_demo.py` replays pacemaker/atrial
  and ventricular/recovery summaries through the same parent planner.

::: scpn_phase_orchestrator.supervisor.hierarchy

## Hierarchy Adapter Boundaries

`hierarchy_adapters` adds decoded JSONL, REST-payload, and WebSocket-frame
helpers over `HierarchyTransportRuntime`. These helpers are transport
boundaries only: they do not open sockets, own HTTP servers, start event loops,
or apply actuation. They return `HierarchyAdapterResult` records containing
accepted/rejected counts, sequence watermarks, parent-plan summaries, and the
underlying sync ledger.

::: scpn_phase_orchestrator.supervisor.hierarchy_adapters

---

## Byzantine Meta-Orchestrator Manifest

`build_bft_meta_orchestrator_manifest()` turns signed child-supervisor policy
proposals into an offline quorum-review manifest. The manifest records the
winning payload hash, accepted and rejected node IDs, hash-linked audit parent,
blocked reasons when quorum is absent, and a canonical manifest hash.

The helper verifies HMAC-SHA256 proposal signatures against a supplied keyring,
but it does not open network transport or permit direct actuation. Accepted
manifests still have to pass the normal supervisor review gate before use.

::: scpn_phase_orchestrator.supervisor.byzantine

---

## Strange-Loop Supervisor Monitor

`StrangeLoopSupervisor` is the first self-referential supervisor slice. It
treats the supervisor's own action stream as a four-dimensional control
channel over `K`, `alpha`, `zeta`, and `Psi`. The monitor records recent
action bundles, computes a control phase, control coherence, drift score,
oscillation score, and over-control score, then returns conservative damping
recommendations for a normal policy or safety gate to approve.

```python
from scpn_phase_orchestrator.supervisor import StrangeLoopSupervisor

loop = StrangeLoopSupervisor(overcontrol_threshold=0.2)
assessment = loop.observe(actions_from_supervisor_policy)

if assessment.recommended_actions:
    audit_payload = assessment.to_audit_record()
```

This slice does not hot-patch the supervisor or claim autonomous
self-awareness. It provides an auditable meta-control signal that can detect
policy drift, control-loop oscillation, and excessive actuation before those
dynamics are fed back into the plant.

Long-run drift scenario helpers exercise that monitor across deterministic
40-step review traces for stable power-grid trims, cardiac policy drift,
traffic-control oscillation, and plasma over-control. The fixture corpus stays
non-actuating and execution-disabled, publishes stable scenario/result hashes,
and is gated in the reference suite so drift, oscillation, and over-control
threshold behavior remains reproducible across releases.
Studio renders the resulting audit records through the public
`scpn_phase_orchestrator.studio.build_strange_loop_studio_panel()` facade,
which preserves the
`strange_loop_drift_review_not_live_actuation` boundary, validates SHA-256
evidence hashes and finite metric ranges, and keeps all recommendations behind
the normal review and safety gate.

::: scpn_phase_orchestrator.supervisor.strange_loop

---

## Morphogenetic Topology Field

`MorphogeneticTopologySupervisor` evolves a persistent field over the pairwise
coupling topology. Each step combines:

- pairwise phase-alignment reaction terms
- incident-edge diffusion over the current topology field
- bounded growth and shrink rates
- a hard maximum per-step coupling delta

The result is a next-step `K_nm`, a carried `MorphogeneticFieldState`, grown and
shrunk edge lists, and compact field statistics for audit logs.

```python
from scpn_phase_orchestrator.supervisor import (
    MorphogeneticTopologySupervisor,
    build_morphogenetic_field_snapshot,
    render_morphogenetic_field_svg,
)

supervisor = MorphogeneticTopologySupervisor()
result = supervisor.step(phases, knm)

next_knm = result.knm
field_state = result.field_state
audit_payload = result.to_audit_record()
snapshot = build_morphogenetic_field_snapshot(result, top_k=5)
heatmap_rows = snapshot.heatmap_rows
svg_artifact = render_morphogenetic_field_svg(result, top_k=5)
```

This slice provides a reviewable grow/shrink primitive for topology shaping. It
does not bypass the existing policy, causal, STL, or action-projection gates.
The field snapshot helper is dependency-free and emits JSON-safe statistics,
ASCII heatmap rows, and strongest-edge records for reports or later UI
rendering. Coupling and carried topology-field matrices are strict off-diagonal
graph objects: boolean and complex aliases are rejected before float coercion,
and non-zero self-edge diagonals are rejected before any field evolution,
snapshot, or SVG rendering.

`render_morphogenetic_field_svg()` is the first richer UI rendering surface for
the same field state. It produces a deterministic, dependency-free SVG heatmap
plus top-edge labels and snapshot metadata. The renderer is passive: it turns an
already computed field into a review artefact and does not mutate policy,
coupling, or actuation state.
Studio packages those SVG artefacts through the public
`scpn_phase_orchestrator.studio.build_morphogenetic_field_studio_panel()`
facade, which validates complete SVG
documents, fixed-width heatmap rows, field-energy statistics, and sorted
off-diagonal topology edges before exposing the panel as passive operator
evidence.

`domainpacks/swarm_robotics/morphogenetic_field_demo.py` provides a deterministic
domainpack proof: it evaluates a split-flock phase state and emits the
morphogenetic field audit payload plus snapshot rows without live actuation.

`domainpacks/power_grid/morphogenetic_field_demo.py` provides the same
non-actuating proof for a stressed grid replay: generator rotor and area
frequency layers remain near-synchronised while tie-line, load-demand, and
renewable layers drift, producing reviewable grown/shrunk field-edge records.

`domainpacks/traffic_flow/morphogenetic_field_demo.py` extends the demo set with
a corridor spillback replay: corridor, network, and equity-pressure layers
remain locally aligned while intersection, demand, and weather phases stress
the field, again without live actuation.

`domainpacks/plasma_control/morphogenetic_field_demo.py` adds a research plasma
replay: transport-barrier, current-profile, and global-equilibrium layers remain
locally aligned while turbulence, tearing, ELM, and wall-interaction phases
stress the field, again without live actuation.

`domainpacks/network_security/morphogenetic_field_demo.py` adds a
lateral-movement replay: normal-traffic and defence-response layers remain
locally aligned while the attack-vector layer stresses the field, again without
live actuation.

::: scpn_phase_orchestrator.supervisor.morphogenetic

---

## Sheaf Coherence Supervisor

`SheafCoherenceSupervisor` evaluates N-channel node states against directed
restriction maps. It builds the block sheaf Laplacian, computes edge residuals,
and reports obstruction metrics for audit logs.

Inputs are fail-closed real-valued tensors: `node_states` must have shape
`(n_nodes, n_channels)` and `restriction_maps` must have shape
`(n_nodes, n_nodes, n_channels, n_channels)`. Boolean aliases, complex values,
non-finite values, and malformed object payloads are rejected before Laplacian
assembly so the obstruction score cannot depend on implicit dtype coercion.

This supervisor-facing sheaf-cohomology slice exposes obstruction score,
consistency energy, approximate kernel dimension, obstruction dimension, and a
review-only obstruction-aware control primitive. It does not claim a complete
formal proof system or autonomous sheaf-control loop.

```python
from scpn_phase_orchestrator.supervisor import (
    SheafCoherenceSupervisor,
    build_sheaf_obstruction_summary,
)

supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
result = supervisor.assess(node_states, restriction_maps)
summary = build_sheaf_obstruction_summary(result)

if result.obstruction_score > 0.1:
    audit_payload = summary.to_audit_record()
```

`propose_sheaf_obstruction_control()` projects an obstructed section one
bounded step down the sheaf-Laplacian consistency-energy gradient. The use case
is operator review: identify a mathematically justified state correction that
reduces obstruction while recording before/after cohomology dimensions. The
proposal is always non-actuating, execution-disabled, and review-required.

```python
from scpn_phase_orchestrator.supervisor import (
    propose_sheaf_obstruction_control,
)

proposal = propose_sheaf_obstruction_control(
    node_states,
    restriction_maps,
    step_size=0.25,
    max_update_norm=0.4,
)
assert proposal.projected_consistency_energy <= proposal.baseline_consistency_energy
assert proposal.execution_disabled
```

`domainpacks/edge_consensus_nchannel/sheaf_obstruction_demo.py` provides a
heterogeneous-domain replay: `P`, `I`, `S`, `Load`, `Trust`, and
`ConsensusHealth` node states are evaluated across edge, gateway, and parent
restriction maps, producing nominal and stressed obstruction audit records
without live actuation.

`domainpacks/power_grid/sheaf_obstruction_demo.py` adds a second
heterogeneous-domain replay. It evaluates generation, tie-line, load, and
renewable regions over rotor-angle, frequency-deviation, tie-flow, demand, and
renewable-ramp channels, then reports nominal versus line-fault obstruction
summaries.

`domainpacks/network_security/sheaf_obstruction_demo.py` adds a security replay.
It evaluates normal-traffic, attack-vector, and defence-response cohorts over
traffic-rate, threat-level, defence-phase, and trust-score channels, then
reports nominal versus lateral-movement obstruction summaries.

`build_sheaf_obstruction_summary()` hardens the raw obstruction metric into a
reviewable triage record. It classifies `nominal`, `warning`, and `critical`
states from explicit thresholds and reports the strongest residual edges so
operators can see which directed restrictions are failing.

Studio exposes this evidence through
`build_sheaf_cohomology_studio_panel(records, summaries, control_proposals)`.
That panel keeps obstruction records, residual-edge summaries, and bounded
review-only control proposals together while preserving disabled execution and
actuation gates.

::: scpn_phase_orchestrator.supervisor.sheaf

---

## Value-Alignment Guard

`ValueAlignmentGuard` is a hard safety wrapper around proposed
`ControlAction` lists. It evaluates explicit objective constraints, blocks
violating actions, and returns a forced fallback action set when the proposal
does not satisfy the configured score threshold.

The guard is intentionally simple and auditable: no hidden reward model is
loaded at runtime. Domainpacks can translate their safety or objective priors
into `ValueConstraint` entries and attach the resulting decision record to the
normal audit trace.

Policies may also include `ValueParetoObjective` entries. When present,
`ValueAlignmentGuard.evaluate(..., objective_deltas={...})` requires finite
objective deltas, blocks regressions beyond each objective's allowed tolerance,
and requires at least one positive configured objective to improve. Missing
objective evidence fails closed and forces the same safe fallback path. Audit
records include `pareto_violations` with the observed delta, required delta,
allowed regression, and counterfactual reason.

Binding specs may carry the same policy as a reviewable `value_alignment`
template:

```yaml
value_alignment:
  minimum_score: 0.8
  constraints:
    - name: limit-coupling
      knob: K
      scope: global
      max_abs_value: 0.1
      weight: 2.0
  fallback_actions:
    - knob: zeta
      scope: global
      value: 0.0
      ttl_s: 1.0
      justification: value guard safe hold
  pareto_objectives:
    - name: safety_margin
      min_delta: 0.01
      max_regression: 0.0
```

Use `value_alignment_policy_from_binding_spec(spec)` to convert that template
into a `ValueAlignmentPolicy`. Audit records include hard bound violations,
Pareto objective violations, and score-threshold counterfactuals so reviewers
can distinguish a blocked unsafe action, a candidate that regresses a protected
objective, and a fallback forced by the policy's minimum alignment score.

Domainpack templates now include review-time examples for cardiac rhythm,
power grid, network security, fusion equilibrium, neuroscience EEG, brain
connectome, sleep architecture, circadian biology, epidemic SIR, and other
simulation/replay domainpacks. These templates are guard priors for reviewable
candidate actions; they are not live medical, grid, vehicle, financial,
public-health, or security operating policies.

```python
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import (
    ValueAlignmentGuard,
    ValueAlignmentPolicy,
    ValueParetoObjective,
    ValueConstraint,
    value_alignment_policy_from_binding_spec,
)

policy = ValueAlignmentPolicy(
    constraints=(ValueConstraint("limit-coupling", knob="K", max_abs_value=0.1),),
    fallback_actions=(
        ControlAction("zeta", "global", 0.0, 1.0, "alignment fallback: hold"),
    ),
    pareto_objectives=(
        ValueParetoObjective("safety_margin", min_delta=0.01, max_regression=0.0),
    ),
)
decision = ValueAlignmentGuard(policy).evaluate(
    proposed_actions,
    objective_deltas={"safety_margin": 0.02},
)
actions_to_apply = decision.actions_to_apply
audit_payload = decision.to_audit_record()

templated_policy = value_alignment_policy_from_binding_spec(binding_spec)
```

::: scpn_phase_orchestrator.supervisor.alignment

---

## Policy Engine

Rule-based evaluation of supervisor actions.

### SupervisorPolicy

```python
SupervisorPolicy(
    regime_manager: RegimeManager,
    petri_adapter: PetriNetAdapter | None = None,
    gains: SupervisorPolicyGains | None = None,
    admission_gate: PolicyCBFAdmissionGate | None = None,
)
```

### decide()

```python
def decide(
    upde_state: UPDEState,
    boundary_state: BoundaryState,
    petri_ctx: dict[str, float] | None = None,
) -> list[ControlAction]
```

Returns a list of `ControlAction` instructions. Each action specifies:

| Field | Type | Example |
|-------|------|---------|
| `knob` | `str` | `"K"`, `"zeta"`, `"psi"` |
| `scope` | `str` | `"global"`, `"layer_0"` |
| `value` | `float` | `0.05` (K boost), `0.1` (zeta damp) |
| `ttl_s` | `float` | `5.0` (action expires after 5s) |
| `justification` | `str` | `"degraded: K boost"` |

When an optional `PolicyCBFAdmissionGate` is supplied, matching supervisor
actions are admitted through verified neural CBF filters before `decide()`
returns. `last_admission_records` exposes deterministic audit records for the
latest call, including the CBF filter digest, certificate digest, admission
status, admitted value, and SMT-LIB artefact hash.

### Regime-action mapping

| Regime | Actions |
|--------|---------|
| NOMINAL | None (no intervention) |
| DEGRADED | K boost +0.05 (global) |
| CRITICAL | ζ damping +0.1 + K reduce -0.03 (worst layer) |
| RECOVERY | K restore +0.025 (half boost, global) |

### Hard violation override

Hard boundary violations (`BoundaryState.hard_violations`) force
CRITICAL regardless of R values.

## Policy CBF Admission

`PolicyCBFAdmissionGate` is the opt-in bridge between heuristic supervisor
proposals and certificate-bound neural CBF admission. Each `PolicyCBFChannel`
selects one action knob/scope, validates a matching `BarrierCertificate` for the
provided `ControlBarrierFilter`, extracts named runtime metrics from `UPDEState`
and `BoundaryState`, and emits a deterministic SMT-LIB admission artefact for
the scalar CBF half-space checked at that decision. The gate does not execute
Z3 locally and does not actuate; it constrains, admits, or rejects proposal
values before downstream projection.

::: scpn_phase_orchestrator.supervisor.cbf_admission

---

**Performance:** `decide()` < 50 μs.

::: scpn_phase_orchestrator.supervisor.policy

---

## Causal Counterfactual Rollouts

`CausalInterventionEngine` evaluates proposed supervisor actions by running
paired UPDE trajectories from the same state:

- baseline: no action
- intervention: action-adjusted `K`, `alpha`, `zeta`, or `Psi`

The result is a `CounterfactualRollout` with `R` and `Psi` trajectories,
final and mean `R` deltas, signed final phase delta, and a serialisable audit
payload.
Counterfactual phases, frequency vectors, coupling matrices, phase-lag
matrices, and lagged causal traces are validated as finite real-valued numeric
arrays before simulation or causal scoring. Boolean aliases and
complex/object-complex payloads are rejected before float coercion so rollouts
and lagged-linear influence estimates stay on the real Kuramoto state space.

```python
from scpn_phase_orchestrator.supervisor import CausalInterventionEngine

engine = CausalInterventionEngine(n_oscillators=8, dt=0.01, horizon=20)
rollout = engine.evaluate_actions(phases, omegas, knm, alpha, 0.0, 0.0, actions)
record = rollout.to_audit_record()
attribution = rollout.attribute(threshold=1e-3).to_audit_record()
```

This is the first causal-supervision slice: it does not claim formal
do-calculus yet, but it makes every proposed actuation comparable against a
no-action counterfactual under the same UPDE dynamics.

`CounterfactualRollout.attribute()` compresses the final and mean `R` deltas
into an audit-ready effect label: `stabilising`, `neutral`, or `destabilising`.

`learn_causal_graph()` adds a lightweight live causal-model learner. It
estimates signed directed edges from lagged monitor traces and appends explicit
`do(knob:scope) -> R` edges from paired counterfactual rollouts. The output is
a `CausalGraphEstimate` with JSON-safe nodes, edge weights, confidence scores,
lags, and evidence labels for the audit trail.

```python
from scpn_phase_orchestrator.supervisor import learn_causal_graph

graph = learn_causal_graph(
    {"R_good": good_trace, "R_bad": bad_trace},
    [rollout],
    lag=1,
    min_abs_weight=1e-4,
)
audit_graph = graph.to_audit_record()
```

`build_temporal_causal_hypergraph_experiment()` is the research-screening layer
for temporal-causal hypergraph candidates. It compares each proposed
time-symmetric hyperedge against a deterministic family of conventional
baselines before any claim can be made:

- lagged-linear graph edge score from `learn_causal_graph()`;
- lagged Pearson correlation between source and future target;
- lagged-delta Pearson correlation between source and target increment;
- pairwise Granger-style residual improvement over target history;
- target-persistence null correlation.

Candidate hyperedges are accepted for review only when their score beats the
strongest baseline by the configured margin. The manifest stays research-only:
production claims, hot patches, and actuation are disabled, and non-winning
candidates are retained as blocked evidence for audit comparison. Use this for
offline discovery of higher-order temporal coupling hypotheses, not for
real-time causal intervention.

```python
from scpn_phase_orchestrator.supervisor import (
    build_temporal_causal_hypergraph_experiment,
)

manifest = build_temporal_causal_hypergraph_experiment(
    {
        "driver": driver_trace,
        "response": response_trace,
        "distractor": distractor_trace,
    },
    [
        {
            "sources": ["driver", "response"],
            "target": "response",
            "time_offsets": [-1, 0],
            "score": candidate_score,
        }
    ],
    lag=1,
    min_abs_weight=1e-4,
    required_baseline_margin=0.1,
)
assert manifest["production_claim_permitted"] is False
assert manifest["baseline"]["strongest_baseline"] in {
    "lagged_linear_graph",
    "lagged_pearson",
    "lagged_delta_pearson",
    "granger_residual_improvement",
    "target_persistence_null",
}
```

Domainpack demos:

- `domainpacks/cardiac_rhythm/causal_attribution_demo.py` evaluates a
  pacing-drive candidate against a ventricular-disturbance baseline.
- `domainpacks/power_grid/causal_attribution_demo.py` evaluates a governor
  droop coupling candidate against a no-action load-step baseline.
- `domainpacks/traffic_flow/causal_attribution_demo.py` evaluates a signal-cycle
  coupling candidate against a no-action corridor-spillback baseline.
- `domainpacks/network_security/causal_attribution_demo.py` evaluates a
  firewall-coupling candidate against a no-action lateral-movement baseline.

**Backend and cost:** each evaluation performs two UPDE rollouts over the
configured horizon, so work scales with `2 * horizon` engine steps. It uses
the existing `UPDEEngine` backend dispatcher; Rust acceleration is used when
available, otherwise the NumPy path is used.

::: scpn_phase_orchestrator.supervisor.causal

---

## Policy Rules (Declarative)

Declarative rules loaded from YAML/JSON configuration.

### Data model

```python
PolicyCondition(metric: str, layer: int | None, op: str, threshold: float)
CompoundCondition(conditions: list[PolicyCondition], logic: str = "AND")
PolicyAction(knob: str, scope: str, value: float, ttl_s: float)
PolicyRule(
    name: str,
    regimes: list[str],       # active in these regimes
    condition: PolicyCondition | CompoundCondition,
    actions: list[PolicyAction],
    cooldown_s: float = 0.0,  # min seconds between firings
    max_fires: int = 0,       # 0 = unlimited
)
```

### STL Monitors In Policy YAML

Policy files may also declare reviewable Signal Temporal Logic monitors under
top-level `stl_monitors`. These monitors do not emit control actions directly;
they evaluate scalar traces and return audit records that can be used by the
runtime gate or safety review job.

```yaml
rules: []
stl_monitors:
  - name: keep_sync
    spec: always (R >= 0.3)
    severity: hard
  - name: eventual_recovery
    spec: eventually (R >= 0.8)
```

```python
from scpn_phase_orchestrator.supervisor.policy_rules import (
    evaluate_policy_stl_specs,
    load_policy_stl_specs,
)

specs = load_policy_stl_specs("policy.yaml")
results = evaluate_policy_stl_specs(specs, {"R": [0.2, 0.4, 0.9]})
audit_payloads = [result.to_audit_record() for result in results]
```

### PolicyEngine

```python
engine = PolicyEngine(rules)
engine.advance_clock(dt)
actions = engine.evaluate(regime, upde_state, good_layers, bad_layers)
```

Rules are evaluated in list order. Each rule fires if:
1. Current regime is in `rule.regimes`
2. Condition evaluates True against UPDEState metrics
3. Cooldown has expired since last firing
4. `max_fires` not exceeded

`load_policy_rules(path)` loads rules from YAML/JSON file.

::: scpn_phase_orchestrator.supervisor.policy_rules

## Policy Diagnostics

Dry-run helpers for validating policy reachability, overlap, cooldown, and
action output before a rule set is allowed into a live supervisor path.

::: scpn_phase_orchestrator.supervisor.policy_diagnostics

## Formal Export

Export helpers translate Petri-net, policy-rule, and policy-declared STL
surfaces into model-checker artefacts for independent safety analysis. PRISM
exports remain the default; TLA+ modules are available for protocol and policy
transition-system checks.

The CLI supports:

```bash
spo formal-export domainpacks/my_domain/binding_spec.yaml --export protocol
spo formal-export domainpacks/my_domain/binding_spec.yaml --export policy
spo formal-export domainpacks/my_domain/binding_spec.yaml --export stl
spo formal-export domainpacks/my_domain/binding_spec.yaml --export protocol-tla
spo formal-export domainpacks/my_domain/binding_spec.yaml --export policy-tla
spo formal-export domainpacks/my_domain/binding_spec.yaml --export policy-smt
spo formal-export domainpacks/my_domain/binding_spec.yaml --export package
```

`--export stl` reads `stl_monitors` from the sibling `policy.yaml` by default
and emits signal constants plus satisfied/violated labels for the builtin STL
subset. This is a model-checker linkage surface; full temporal automata
synthesis remains future work. `--export protocol-tla` emits a bounded TLA+
module with Petri places as variables, transition guards as constants, `Init`,
`Next`, `Spec`, and `Safety == TypeOK`. `--export policy-tla` emits bounded
rule-fire counters plus reachability predicates for fired rules and emitted
actions. `--export policy-smt` emits an SMT-LIB v2 feasibility model for Z3:
the model declares the active regime, metric inputs, bounded rule-fire counters,
rule firing predicates, emitted-action predicates, and a final `check-sat`
envelope asking whether at least one rule can fire under the declared guards.
`--export package` emits a JSON formal verification package manifest that binds
protocol PRISM/TLA, policy PRISM, and generated policy SMT-LIB artefact hashes
to named safety properties and external PRISM/TLC/Z3 command records. The
package API also accepts reviewed Promela and SMT-LIB text artefacts through
`FormalTextArtifact`, linking them to non-executing SPIN and Z3
command/readiness manifests under the same hash and disabled-execution
contract. The package does not run model checkers; all command records keep
`execution_permitted=false`. Add
`--include-checker-readiness` to append non-executing checker availability
records to that JSON; `--checker-path executable=/path` can make CI readiness
evidence deterministic, and `--checker-path executable=` forces a missing
checker record without invoking anything.
`build_runtime_control_certificate()` turns a package, checker readiness
records, externally reviewed checker result records, and finite runtime bounds
into a deterministic `FormalRuntimeCertificate`. The certificate is the runtime
handoff contract for verifiable control: every required property must have a
matching available checker and a passed result bound to the exact package hash.
Missing, failed, stale, or unavailable evidence produces `status="blocked"`.
Even `status="verified_non_actuating"` keeps `actuation_permitted=false`; it is
an auditable precondition for operator review or a separate runtime monitor,
not permission to execute hardware controls.
Remote CI owns the first external execution lane through
`formal-model-checkers.yml`, which installs SPIN and Z3, materialises reviewed
Promela/SMT-LIB smoke artefacts, validates disabled package/readiness metadata,
and runs those external checkers only under the CI-only execution guard.
The same lane now also materialises safety-domain packages for
`cardiac_rhythm`, `chemical_reactor`, `power_grid`, `pll_clock`,
`autonomous_vehicles`, `satellite_constellation`, `power_safety_nchannel`,
`traffic_flow`, `swarm_robotics`, `manufacturing_spc`, and `robotic_cpg`. Each
domain package binds a SPIN operator-approval gate and a Z3 hard-bound
feasibility artefact derived from the domainpack safety boundaries, preserving
the disabled runtime-execution contract while allowing remote CI to execute the
external checker commands in an isolated environment.

For builtin STL automata, `synthesise_stl_controller_candidates()` provides a
non-actuating controller-synthesis bridge. It proposes signal-level candidate
actions from the weakest violated predicate and records `actuating=False`; the
proposal is an audit artefact, not a live controller or bypass around policy and
actuation safety gates. `project_stl_controller_candidates()` can then map
those candidates through explicit policy-approved projection templates and the
standard `ActionProjector`, yielding bounded `ControlAction` proposals while
still recording `actuating=False`.
`synthesise_stl_closed_loop_plan()` combines those two stages into an offline
closed-loop review artefact: it records the feedback signals, trace length,
future review horizon, projected actions, and fail-closed blockers without
mutating runtime state or enabling actuation.

::: scpn_phase_orchestrator.supervisor.formal_export

::: scpn_phase_orchestrator.supervisor.formal_export.smt_export

---

## Rust Supervisor Backend Probe

The Python supervisor remains the default runtime-control surface. The optional
Rust `spo-supervisor` PyO3 bindings are validated separately through
`audit_rust_supervisor_backend()`, which checks required `spo_kernel` symbols
and runs deterministic, non-actuating smoke checks for regime classification,
boundary observation, and coherence monitoring. `spo doctor` reports this as
the optional `rust-supervisor` backend so operators can diagnose a missing or
malformed Rust supervisor FFI without changing live-control behavior.

::: scpn_phase_orchestrator.supervisor.rust_backend

---

## Petri Net FSM

Formal Petri net state machine enabling formal verification of
safety properties: deadlock freedom, liveness, bounded token counts.

### Components

| Class | Fields | Description |
|-------|--------|-------------|
| `Place` | `name: str` | Token container (regime state) |
| `Arc` | `place: str, weight: int` | Token flow edge |
| `Guard` | `metric: str, op: str, threshold: float` | Firing condition |
| `Transition` | `name, inputs, outputs, guard` | Guarded state change |
| `Marking` | `tokens: dict[str, int]` | Current token distribution |

### Guard operators

Guards support five comparison operators: `>`, `>=`, `<`, `<=`, `==`.
`Guard.evaluate(ctx)` checks the condition against a context dictionary.

### PetriNet methods

| Method | Description |
|--------|-------------|
| `enabled(marking, ctx)` | Returns transitions whose guards pass |
| `fire(marking, transition)` | Moves tokens and returns new marking |
| `step(marking, ctx)` | Fires first enabled transition |

`parse_guard("R < 0.3")` parses a string into a Guard object.

**Performance:** `enabled_transitions()` < 10 μs.

::: scpn_phase_orchestrator.supervisor.petri_net

---

## Petri Net Adapter

Bridge between UPDEState and the Petri net FSM.

```python
PetriNetAdapter(
    net: PetriNet,
    initial_marking: Marking,
    place_to_regime: dict[str, str],  # maps place names to regime names
    event_bus: EventBus | None = None,
)
```

`adapter.step(ctx)` evaluates the Petri net with the given context and
returns the current `Regime` based on which place holds the token.

::: scpn_phase_orchestrator.supervisor.petri_adapter

---

## Event Bus

Publish-subscribe system for supervisor events.

### RegimeEvent (frozen dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `str` | `"regime_transition"` or `"boundary_violation"` |
| `step` | `int` | Step number when event occurred |
| `detail` | `str` | Human-readable description |

### EventBus

```python
bus = EventBus(maxlen=200)
bus.subscribe(callback)
bus.post(RegimeEvent(kind="regime_transition", step=42, detail="nominal->degraded"))
bus.history  # list of all events
bus.count    # total events posted
```

Events are stored in a bounded deque (default 200). Subscribers are
called synchronously on `post()`.

::: scpn_phase_orchestrator.supervisor.events

---

## Model-Predictive Controller (MPC)

Anticipatory control using Ott-Antonsen mean-field reduction.

### Prediction (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `R_predicted` | `list[float]` | Predicted R trajectory (horizon steps) |
| `will_degrade` | `bool` | R predicted to cross DEGRADED threshold |
| `will_critical` | `bool` | R predicted to cross CRITICAL threshold |
| `steps_to_degradation` | `int` | Steps until predicted degradation |

### PredictiveSupervisor

```python
PredictiveSupervisor(
    n_oscillators: int,
    dt: float,
    horizon: int = 10,              # prediction steps ahead
    divergence_threshold: float = 0.3,  # OA model trust threshold
)
```

**Methods:**

- `predict(phases, omegas, knm, alpha) → Prediction` — runs OA forward
  model for `horizon` steps, returns trajectory
- `decide(phases, omegas, knm, alpha, upde_state, boundary_state)
  → list[ControlAction]` — predicts then acts if degradation imminent

`phases`, `omegas`, `knm`, and `alpha` are finite real-valued arrays. Boolean
aliases and complex/object-complex payloads are rejected before OA prediction so
the forward model cannot silently reinterpret non-physical inputs as real
oscillator states, frequencies, coupling, or phase-lag matrices.

### Safety fallback

When |R_predicted - R_measured| > `divergence_threshold`, the MPC
**discards** its prediction and falls back to reactive control. This
prevents acting on a forward model that has lost accuracy.

### Computational advantage

The OA reduction is O(1) per step (single complex ODE) versus O(N)
for the full Kuramoto model. For N=1000 oscillators with horizon=10,
MPC prediction costs ~10 ODE steps versus 10000 Euler steps.

::: scpn_phase_orchestrator.supervisor.predictive

---

## FEP Predictive Supervisor

`FEPPredictiveSupervisor` is the first Python supervisor mode that uses the
existing `VariationalPredictor` as an auditable free-energy signal. It observes
the current phase vector, updates the variational predictor, and emits bounded
`zeta` / `Psi` actions only when free energy, prediction error, or stability
proxy thresholds indicate a pre-emptive correction is needed.

```python
from scpn_phase_orchestrator.supervisor import (
    FEPPredictiveSupervisor,
    assess_fep_hierarchy,
)

fep = FEPPredictiveSupervisor(
    n_oscillators=len(phases),
    dt=0.01,
    target_R=0.8,
    free_energy_threshold=1.0,
)
assessment = fep.assess(phases, omegas)
actions = fep.decide(phases, omegas, upde_state, boundary_state)
audit_payload = assessment.to_audit_record()
```

`FEPPredictionAssessment` records free energy, complexity, mean absolute
prediction error, precision statistics, observed and predicted order
parameters, target `R`, and a scalar surprise proxy. This keeps the FEP path
reviewable in the same audit trail as policy, causal, STL, and topology
decisions.

This slice is intentionally conservative: it is a FEP-Kuramoto correspondence
controller over the existing variational predictor, not a claim of a complete
biological active-inference agent.

`assess_fep_hierarchy()` is the reusable hierarchy primitive. It runs one child
`FEPPredictiveSupervisor` per named child observation, reduces each child's
observed coherence into a parent phase vector, then runs a parent
`FEPPredictiveSupervisor` over the reduced child state. The returned
`FEPHierarchyAssessment` records child assessments, child actions, parent
assessment, parent actions, child `R` values, and parent phase encoding.
Child phase and frequency observations use the same finite real-valued boundary
contract as the single-supervisor path.

```python
hierarchy = assess_fep_hierarchy(
    {
        "generation_area": (generation_phases, generation_omegas),
        "demand_area": (demand_phases, demand_omegas),
    },
    dt=0.01,
    parent_dt=0.1,
)
audit_hierarchy = hierarchy.to_audit_record()
```

Domainpack hierarchy proofs:

- `domainpacks/power_grid/fep_hierarchy_demo.py` runs generation and
  demand/renewable child regions into a parent grid supervisor.
- `domainpacks/cardiac_rhythm/fep_hierarchy_demo.py` runs pacemaker/atrial and
  ventricular/recovery child axes into a parent cardiac supervisor.

---

## Performance summary

| Operation | Budget | Notes |
|-----------|--------|-------|
| `RegimeManager.evaluate()` | < 10 μs | Pure Python comparison |
| `SupervisorPolicy.decide()` | < 50 μs | Rule evaluation + action construction |
| `PetriNet.enabled()` | < 10 μs | Guard evaluation |
| `PredictiveSupervisor.predict()` | < 1 ms | OA mean-field (10 complex ODE steps) |
| `EventBus.post()` | < 5 μs | Synchronous dispatch |

---

## Active Inference Agent

The `ActiveInferenceAgent` provides a **predictive control framework** based on
Karl Friston's Variational Free Energy Principle. It moves beyond static YAML
rules into self-adaptive state-space models.

### Mathematical Model

The agent maintains a low-dimensional internal state $ and minimizes the
**Variational Free Energy** $ between its prediction $\hat{R}$ and the
observed coherence {obs}$:

936875 F \approx \int q(x) \ln \frac{q(x)}{p(R_{obs}, x)} dx 936875

The controller outputs the optimal forcing strength $\zeta$ and reference phase
$\Psi$ to drive the network toward a target coherence level {target}$
(often set to the **metastability threshold**  \approx 0.6$).

### Features
- **Adaptive Suppression:** Spontaneously discovers anti-phase driving ($\Psi = \psi + \pi$) to break harmful phase-locking.
- **Sub-microsecond Control:** Fully implemented in the `spo-kernel` Rust backend for real-time high-frequency response.
- **Emergent Resilientness:** Naturally handles non-stationary frequency drifts by integrating prediction errors into the internal state.

!!! note "Rust-only module"
    `ActiveInferenceAgent` is implemented in `spo-kernel` (Rust crate `spo-supervisor::active_inference`).
    Python access via `spo_kernel.PyActiveInferenceAgent` when the FFI is installed.

## Evolutionary Review Surfaces

Offline-only evolutionary search, grammar, policy DSL, topology mutation, and
example builders used for non-actuating supervisor review workflows.

::: scpn_phase_orchestrator.supervisor.evolutionary_examples

::: scpn_phase_orchestrator.supervisor.evolutionary_petri_grammar

::: scpn_phase_orchestrator.supervisor.evolutionary_policy_dsl

::: scpn_phase_orchestrator.supervisor.evolutionary_search

::: scpn_phase_orchestrator.supervisor.evolutionary_topology_grammar

## Federated Review Surfaces

Federated orchestration, differential-privacy noise service, secure
aggregation, and transport manifests. These APIs produce audit material and
deployment preflight evidence without exporting raw local data.

::: scpn_phase_orchestrator.supervisor.federated

::: scpn_phase_orchestrator.supervisor.federated_dp_noise_service

::: scpn_phase_orchestrator.supervisor.federated_secure_aggregation

::: scpn_phase_orchestrator.supervisor.federated_transport

## Information Geometry and Lineage

Information-geometric control proposals, static scenario examples, and
autopoietic lineage inheritance helpers for review-only policy evolution.
The information-geometry primitive keeps NumPy as the default audit-stable
backend and exposes explicit `backend="jax"` acceleration with reference-gated
parity for Fisher-Rao distance, Wasserstein distance, curvature proxy, and
natural-gradient proposals. Both paths remain non-actuating review surfaces.
The lineage sandbox generates deterministic child-policy candidates from a
parent policy and replay corpus, records accepted/rejected evidence, hashes the
lineage and replay corpus, and keeps live merge, hot patching, execution, and
actuation disabled. The curated replay corpus spans power-grid recovery,
cardiac-rhythm pacing recovery, traffic-flow platooning, and cyber-industrial
recontainment so operators can compare policy diffs across domains before any
separate inheritance-review workflow.
Intergenerational inheritance then signs accepted child-policy records,
materialises inherited genomes, records multi-objective replay fitness, and can
package deterministic history rows for operator review. The history package
links lineage hashes, inheritance hashes, HMAC signature metadata, replay
domains, and fitness ranges while keeping direct hot patching and actuation
disabled.

::: scpn_phase_orchestrator.supervisor.information_geometry

::: scpn_phase_orchestrator.supervisor.information_geometry_examples

::: scpn_phase_orchestrator.supervisor.lineage

## Multiverse and Topos Review

Counterfactual branch simulation, example manifests, branch-risk gates, and
categorical policy composition checks. The multiverse simulator keeps NumPy as
the default deterministic audit backend and exposes explicit `backend="jax"`
acceleration for larger branch corpora where JAX can place the vectorized
rollout on an available accelerator. Reference benchmarks gate JAX output
against NumPy branch hashes, topology metrics, order-parameter trajectories, and
final phase angles while preserving the non-actuating, execution-disabled review
boundary. Multiverse branch rollouts preserve the Kuramoto graph contract by
requiring zero diagonal baseline coupling, phase-lag, and topology-mask matrices;
matrix branch actions are projected back onto the off-diagonal graph before
simulation. Domain scenario fixtures now cover power-grid, cardiac-rhythm,
cyber-industrial, traffic-flow, manufacturing process-control, and plasma-control
use cases with simulator-compatible `K`, `alpha`, `zeta`, and `Psi` candidate
controls. Studio packages rollout manifests and branch-risk reports through
the public
`scpn_phase_orchestrator.studio.build_multiverse_counterfactual_studio_panel()`
facade, which preserves the non-actuating claim boundaries, joins branch hashes,
renders approval/rejection evidence, and never emits executable actions.

::: scpn_phase_orchestrator.supervisor.multiverse

::: scpn_phase_orchestrator.supervisor.multiverse_examples

::: scpn_phase_orchestrator.supervisor.multiverse_risk

::: scpn_phase_orchestrator.supervisor.topos_policy
